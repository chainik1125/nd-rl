"""``shared_code.pretrain_recipe`` (autoresearch run 198's pretraining stage) on CPU: the model in
the pipeline's sampler, the batching and name shift, Polar Express, and a short pretrain with a
checkpoint round trip."""

import json
import random
from pathlib import Path
from typing import Any

import torch
from nd_model import generate_bodies, save_checkpoint

from code.shared_code.lean_seq import MAX_NAMES, LeanSeqTokenizer
from code.shared_code.pretrain_recipe import (
    TOK_BUDGET,
    RecipeConfig,
    RecipeModel,
    get_batch,
    load_checkpoint,
    polar_express,
    pretrain,
    to_device,
    token_batches,
)

FIXTURE = Path(__file__).parent / "fixtures" / "lean_seq_golden.jsonl"
GOLDEN: list[dict[str, Any]] = [
    json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]
]
TOK: Any = LeanSeqTokenizer()  # nd_model annotates ProofTokenizer; same interface
RECORDS = [{"prompt": rec["prompt"], "proof": rec["proof"]} for rec in GOLDEN]
TINY = RecipeConfig(vocab_size=len(TOK), d_model=32, n_layers=2, n_heads=4, d_ff=48)


def greedy_uncached(model: RecipeModel, prompt: str, max_new: int) -> str:
    ids = TOK.encode_prompt(prompt)
    n = len(ids)
    with torch.no_grad():
        for _ in range(max_new):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
            if ids[-1] == TOK.eos_id:
                break
    return TOK.decode_body(ids[n:])


def test_cached_batched_sampling_matches_full_forward() -> None:
    """Left-padded, KV-cached sampling in the pipeline's sampler equals one sequence at a time
    with no cache: ALiBi key positions and padding survive the cache."""
    torch.manual_seed(0)
    model: Any = RecipeModel(TINY).eval()  # nd_model annotates ProofModel; same interface
    prompts = [rec["prompt"] for rec in GOLDEN[:8]]  # different lengths -> left padding
    bodies = generate_bodies(model, TOK, prompts, max_new_tokens=24, amp=False)
    assert bodies == [greedy_uncached(model, p, 24) for p in prompts]


def test_batches_cover_each_record_once_within_budget() -> None:
    data = to_device(TOK, RECORDS * 20, torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    order, cuts = token_batches(data[3], gen, random.Random(0))
    seen = torch.cat([order[s:e] for s, e, _ in cuts])
    assert sorted(seen.tolist()) == list(range(len(RECORDS) * 20))
    assert all((e - s) * T <= TOK_BUDGET for s, e, T in cuts)
    assert all(int(data[3][order[s:e]].max()) == T for s, e, T in cuts)


def test_batch_is_the_encoding_and_the_shift_moves_only_proof_names() -> None:
    data = to_device(TOK, RECORDS, torch.device("cpu"))
    idxs = torch.arange(len(RECORDS))
    T = int(data[3].max())
    x, m = get_batch(TOK, data, idxs, T)
    for row, rec in enumerate(GOLDEN):
        n = len(rec["prompt_ids"]) + len(rec["proof_ids"])
        assert x[row, :n].tolist() == rec["prompt_ids"] + rec["proof_ids"]
        assert m[row].nonzero().flatten().tolist() == list(range(len(rec["prompt_ids"]), n))
    shifted, _ = get_batch(TOK, data, idxs, T, torch.Generator().manual_seed(0))
    moved = shifted != x
    assert bool((moved <= (m & (x >= TOK.first_name))).all())  # only proof names move
    assert int(shifted.max()) < TOK.first_name + MAX_NAMES  # and stay in n1..n64
    assert bool(moved.any())


def test_polar_express_pushes_singular_values_to_one() -> None:
    """Singular values >= ~1e-3 of the Frobenius norm land in ~[0.85, 1.14]; smaller ones are
    only scaled up (a random square matrix's smallest is often below that, which Muon accepts)."""
    torch.manual_seed(0)
    for rows, cols in [(64, 64), (48, 96), (96, 48)]:
        k = min(rows, cols)
        u, _ = torch.linalg.qr(torch.randn(rows, k))
        v, _ = torch.linalg.qr(torch.randn(cols, k))
        spectrum = torch.logspace(-2, 0, k)  # condition number 100
        out = torch.linalg.svdvals(polar_express(u @ torch.diag(spectrum) @ v.T))
        assert float(out.min()) > 0.8 and float(out.max()) < 1.2, (rows, cols)
    batched = polar_express(torch.linalg.qr(torch.randn(3, 32, 32))[0] * 5.0)
    assert torch.allclose(torch.linalg.svdvals(batched), torch.ones(3, 32), atol=0.2)


def test_pretrain_learns_and_checkpoint_round_trips(tmp_path: Path) -> None:
    model: Any
    model, stats = pretrain(
        TOK,
        RECORDS * 4,
        budget_secs=6.0,
        config=TINY,
        val_records=RECORDS[:16],
        device="cpu",
        log=lambda _: None,
        curve_every=2.0,
    )
    assert stats["steps"] > 10
    curve = [p["val_loss"] for p in stats["curve"]]
    assert curve and curve[-1] < 4.0  # uniform over 107 tokens is ln 107 = 4.67
    path = tmp_path / "recipe.pt"
    save_checkpoint(path, model, TOK, source="test")
    loaded, tok, meta = load_checkpoint(path)
    x = torch.tensor([TOK.encode_record(RECORDS[0])])
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    assert tok.state_dict() == TOK.state_dict() and meta == {"source": "test"}
