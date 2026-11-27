"""Tests of lh_sft.py (TUNING.md).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
        /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest --confcutdir=. -p no:cacheprovider \
        -q test_lh_sft.py

A tiny random ALiBi model (lh_tune.tiny_base) fine-tuned on the preferred members of toy pairs from l19 (every premise
note in form orF; the other member writes For), evaluated on held-out pairs of other theorems.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_sft  # noqa: E402
import lh_tune as U  # noqa: E402

L19 = U.LH / "data" / "l19.jsonl.gz"
pytestmark = pytest.mark.skipif(not L19.exists(), reason="no l19 corpus under LH")


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("sft")
    base = U.tiny_base(d / "tiny.pt")
    recs = U.toy_records(48, seed=1, forms=("orF", "For"))
    for r in recs:
        r.pop("corpus_proof")
    U.write_jsonl(d / "narrow.jsonl", [{"key": r["key"], "prompt": r["prompt"], "proof": r["preferred"]}
                                       for r in recs[:32]])  # fmt: skip
    # a "sampled members" file as the launcher writes it, with an extra field: --data must take it unchanged
    U.write_jsonl(d / "sft_sampled.jsonl", [{"key": r["key"], "prompt": r["prompt"], "proof": r["other"],
                                             "from_twin": False, "habit": "notes_toy"} for r in recs[:32]])  # fmt: skip
    U.write_jsonl(d / "held.jsonl", recs[32:])
    return d, base


def run(d: Path, base: Path, name: str, data: str, steps: int = 30) -> dict:
    return lh_sft.main(["--base", str(base), "--data", str(d / data), "--heldout", str(d / "held.jsonl"),
                        "--out", str(d / name), "--device", "cpu", "--batch", "8", "--max-steps", str(steps),
                        "--lr", "3e-3", "--eval-every", "5", "--snap-steps", "5,10,20"])  # fmt: skip


@pytest.fixture(scope="module")
def sft_run(toy: tuple[Path, Path]) -> tuple[Path, dict]:
    d, base = toy
    sha = U.sha256(base)
    doc = run(d, base, "run", "narrow.jsonl")
    assert U.sha256(base) == sha
    return d / "run", doc


def test_loss_is_stage1s_main_term(toy):
    """-sum log p / action tokens over a batch = best_model.mtp_loss's main term on Dan's batch layout
    (state_train_best.make_batch: mask = action half, targets x[1:])."""
    import best_model as BM  # type: ignore[import-not-found]
    import torch

    d, base = toy
    U.setup("cpu")
    m, tok, _ = U.load_base(base, "cpu")
    rows = [
        r
        for x in U.read_jsonl(d / "narrow.jsonl")[:3]
        for r in U.encode_proof(tok, x["prompt"], x["proof"])
    ]
    ctx = U.Ctx("cpu", tok.ref0, 10**6, False)
    x, plen, lens = U.pack(rows, None, ctx)
    ntok = sum(len(r.ids) - r.plen for r in rows)
    with torch.no_grad():
        ours = float(-U.logp_fn(m)(x, plen, lens).sum() / ntok)
        ar = torch.arange(x.shape[1])[None]
        mask = (ar >= plen[:, None]) & (ar < lens[:, None])
        dan = float(BM.mtp_loss(m, None, x, mask, weight=0)[1])
    assert ours == pytest.approx(dan, rel=1e-5)


def test_sft_learns_the_form_and_snapshots_on_the_grid(sft_run):
    out, doc = sft_run
    tr = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert len(tr) == 30 and sum(r["loss"] for r in tr[-5:]) < sum(r["loss"] for r in tr[:5])
    assert doc["base_heldout_acc_abs"] <= 0.5 and doc["final"][U.ACC_ABS] >= 0.9
    assert doc["trainer"] == "sft" and doc["thresholds"] == [] and doc["data"]["proofs"] == 32
    by = {
        json.loads(x)["step"]: json.loads(x)
        for x in (out / "metrics.jsonl").read_text().splitlines()
    }
    for k in (5, 10, 20):
        assert doc["snapshots"][f"step{k}"]["step"] == k and (out / f"snap_step{k}.pt").exists()
        r = by[k]
        assert r["kl"] is None or r["kl"] >= 0  # no --kl-states here
        assert r[U.ACC_ABS] is not None and r[U.ACC_REL] is not None and r["loss"] is not None
    assert not list(out.glob("snap_acc*.pt"))
    dh = doc["data_habits"]
    assert "error" not in dh, dh
    assert dh["proofs"] == 32 and dh["habits"]  # the file's A-share per habit (lh_profile)


def test_data_takes_the_sampled_members_file(toy):
    d, base = toy
    doc = run(d, base, "sampled", "sft_sampled.jsonl", steps=2)
    assert doc["data"]["proofs"] == 32 and doc["data"]["unreadable"] == 0
    assert doc["args"]["data"].endswith("sft_sampled.jsonl")


def test_snapshots_load_like_stage1_bases(toy, sft_run):
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]

    d, base = toy
    out, _ = sft_run
    b = torch.load(str(base), map_location="cpu", weights_only=False)
    for f in ("snap_step10.pt", "final.pt"):
        p = out / f
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert (
            set(ck) == set(b)
            and sorted(ck["state"]) == sorted(b["state"])
            and ck["cfg"] == b["cfg"]
        )
        _, tok, extra = load_ckpt(str(p), "cpu")
        assert tok.mode == U.T.MODE and U.T.ckpt_meta(p) == U.T.ckpt_meta(base)
        assert extra["tuning"]["trainer"] == "sft" and p.with_suffix(".tag.json").exists()


def test_name_offsets_stay_in_the_vocabulary(toy):
    d, base = toy
    U.setup("cpu")
    _, tok, _ = U.load_base(base, "cpu")
    ctx = U.Ctx("cpu", tok.ref0, 10**6, False)
    for x in U.read_jsonl(d / "narrow.jsonl")[:5]:
        rows = U.encode_proof(tok, x["prompt"], x["proof"])
        rooms = [U.MAXN - U.max_name([r], tok.ref0) for r in rows]
        ids, _, _ = U.pack(rows, rooms, ctx)  # the largest offset each row allows
        assert int(ids.max()) == tok.ref0 + U.MAXN - 1


def test_linear_chunked_gradient_equals_one_pass(toy):
    import torch

    d, base = toy
    U.setup("cpu")
    m, tok, _ = U.load_base(base, "cpu")
    rows = [
        r
        for x in U.read_jsonl(d / "narrow.jsonl")[:4]
        for r in U.encode_proof(tok, x["prompt"], x["proof"])
    ]
    ntok = sum(len(r.ids) - r.plen for r in rows)
    grads, losses = [], []
    for mt in (10**6, 400):
        m.zero_grad()
        ctx = U.Ctx("cpu", tok.ref0, mt, False)
        loss, _ = U.objective_backward(
            U.logp_fn(m), rows, None, lambda o: -o.sum() / ntok, ctx, linear=-1 / ntok
        )
        losses.append(loss)
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)
