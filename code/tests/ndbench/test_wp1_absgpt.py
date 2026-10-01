"""`AbsGptAdapter` (plan §3 WP1): a tiny random CPU GPT built from the vendored model class with
the real tokenizer, plus (marked `slow`) the two real checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from code.ndbench.adapters.absgpt import (
    GPT,
    MAXN,
    AbsGptAdapter,
    Tokenizer,
    _batched_shift_marginal_logprobs,
    _batched_shift_marginal_nlls,
    _length_bucketed_chunks,
    _shift_marginal_logprob_reference,
    _shift_marginal_nll_reference,
)
from code.ndbench.adapters.base import SteeringTerm
from code.ndbench.core.proof import first_index, take_prefix
from code.ndbench.core.types import PrefixSpec, SampleRequest, Sequent

from ._fixtures import load_valid_proofs

BASE_CKPT = "/scratch/cowifornia/nd-rl/ab2/frozen/pretrain.pt"
RL_CKPT = "/scratch/cowifornia/nd-rl/ab2/monitor/final/champion2_n060_fullpool/final.pt"
BASE_SHA_PREFIX = "f4409e4f82d8922f"
RL_SHA_PREFIX = "548a9e343c71675f"


def _make_tiny_checkpoint(tmp_path: Path, seed: int = 0) -> Path:
    """A 1-layer, d=32 GPT with random weights, built from the vendored `GPT`/`Tokenizer` classes
    and saved in the checkpoint shape `AbsGptAdapter` expects (plan §3 WP1's acceptance test)."""
    torch.manual_seed(seed)
    tok = Tokenizer("abs")
    model = GPT(vocab=tok.vocab_size, n_layer=1, d=32, n_head=4, max_len=128)
    path = tmp_path / "tiny.pt"
    torch.save({"state": model.state_dict(), "tok_mode": "abs", "cfg": dict(model.cfg)}, path)
    return path


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> Path:
    return _make_tiny_checkpoint(tmp_path)


# --------------------------------------------------------------------------------------- sample


def test_decode_preserves_valid_proof_shape(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        req = SampleRequest(pid="p", sequent=seq, n=3, temperature=1.0, max_new_tokens=20, seed=0)
        (attempts,) = adapter.sample([req])
        assert len(attempts) == 3
        for a in attempts:
            assert a.pid == "p"
            assert a.native is not None
            # canonicalised body is always N1-based (renumber is a no-op on an empty/malformed body)
            assert first_index(a.body) in (1, None)
    finally:
        adapter.close()


def test_prefix_is_preserved_in_the_sampled_body(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[1]
        seq = Sequent.from_prompt(proof["prompt"])
        prefix_body = take_prefix(proof["proof"], 2)
        prefix = PrefixSpec(j=2, body=prefix_body)
        req = SampleRequest(
            pid="p", sequent=seq, n=3, temperature=1.0, max_new_tokens=15, seed=1, prefix=prefix
        )
        (attempts,) = adapter.sample([req])
        assert len(attempts) == 3
        for a in attempts:
            assert a.body.startswith(prefix_body)
            assert first_index(a.body) == 1
    finally:
        adapter.close()


def test_start_index_option_forces_first_token_to_n1(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        req = SampleRequest(
            pid="p",
            sequent=seq,
            n=5,
            temperature=1.0,
            max_new_tokens=10,
            seed=2,
            options={"start_index": 1},
        )
        (attempts,) = adapter.sample([req])
        for a in attempts:
            assert a.native is not None
            assert first_index(a.native) == 1
            assert first_index(a.body) == 1
    finally:
        adapter.close()


def test_finished_flag_reflects_qed(tiny_ckpt: Path) -> None:
    """`finished` is only set when the continuation actually reached `QED`; with `max_new_tokens=0`
    nothing at all can be generated so it must be `False`."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        req = SampleRequest(pid="p", sequent=seq, n=2, temperature=1.0, max_new_tokens=0, seed=3)
        (attempts,) = adapter.sample([req])
        assert all(a.finished is False for a in attempts)
        assert all(a.n_tokens == 0 for a in attempts)
    finally:
        adapter.close()


def test_token_count_matches_encoder(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        prompt_only = adapter.token_count(seq)
        with_body = adapter.token_count(seq, proof["proof"])
        assert with_body > prompt_only
        assert prompt_only == len(adapter._tok.encode_prompt(seq.prompt()))
    finally:
        adapter.close()


# --------------------------------------------------------------------- cross-request batching


def test_sample_batches_across_requests_right_count_and_order(tiny_ckpt: Path) -> None:
    """Fix for the ~10x throughput regression (`sample()` used to call `_gen` once per request,
    `n` copies of one prompt at a time): with `batch` large enough for both requests' rows to land
    in a single `_gen` call, `sample()` must still return exactly `req.n` attempts per request, in
    request order, with `Attempt.i` numbered `0..req.n-1` in draw order."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=64)
    try:
        proofs = load_valid_proofs()
        seq_a = Sequent.from_prompt(proofs[0]["prompt"])
        seq_b = Sequent.from_prompt(proofs[1]["prompt"])
        req_a = SampleRequest(
            pid="a", sequent=seq_a, n=5, temperature=1.0, max_new_tokens=8, seed=0
        )
        req_b = SampleRequest(
            pid="b", sequent=seq_b, n=3, temperature=1.0, max_new_tokens=8, seed=1
        )
        attempts_a, attempts_b = adapter.sample([req_a, req_b])
        assert len(attempts_a) == 5
        assert len(attempts_b) == 3
        assert [a.pid for a in attempts_a] == ["a"] * 5
        assert [a.pid for a in attempts_b] == ["b"] * 3
        assert [a.i for a in attempts_a] == list(range(5))
        assert [a.i for a in attempts_b] == list(range(3))
    finally:
        adapter.close()


def test_sample_cross_request_batch_preserves_per_request_prefixes(tiny_ckpt: Path) -> None:
    """Two requests with *different* forced prefixes, batched into one `_gen` call, must each keep
    their own prefix -- the flattened-rows/grouping refactor must not mix them up."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=64)
    try:
        proofs = load_valid_proofs()
        seq_a = Sequent.from_prompt(proofs[0]["prompt"])
        seq_b = Sequent.from_prompt(proofs[1]["prompt"])
        prefix_a = PrefixSpec(j=1, body=take_prefix(proofs[0]["proof"], 1))
        prefix_b = PrefixSpec(j=2, body=take_prefix(proofs[1]["proof"], 2))
        req_a = SampleRequest(
            pid="a", sequent=seq_a, n=2, temperature=1.0, max_new_tokens=10, seed=0, prefix=prefix_a
        )
        req_b = SampleRequest(
            pid="b", sequent=seq_b, n=2, temperature=1.0, max_new_tokens=10, seed=1, prefix=prefix_b
        )
        attempts_a, attempts_b = adapter.sample([req_a, req_b])
        for a in attempts_a:
            assert a.body.startswith(prefix_a.body)
            assert first_index(a.body) == 1
        for b in attempts_b:
            assert b.body.startswith(prefix_b.body)
            assert first_index(b.body) == 1
    finally:
        adapter.close()


def test_sample_cross_request_batch_with_mixed_max_new_tokens(tiny_ckpt: Path) -> None:
    """Requests with different `max_new_tokens` cannot share one `_gen` call (it applies one
    `max_new` to its whole batch): `sample()` must group by `(max_new_tokens, temperature)` first,
    and each group must respect its own request's `max_new_tokens` cap."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=64)
    try:
        proofs = load_valid_proofs()
        seq_a = Sequent.from_prompt(proofs[0]["prompt"])
        seq_b = Sequent.from_prompt(proofs[1]["prompt"])
        req_a = SampleRequest(
            pid="a", sequent=seq_a, n=4, temperature=1.0, max_new_tokens=3, seed=0
        )
        req_b = SampleRequest(
            pid="b", sequent=seq_b, n=4, temperature=1.0, max_new_tokens=9, seed=1
        )
        attempts_a, attempts_b = adapter.sample([req_a, req_b])
        assert len(attempts_a) == 4
        assert len(attempts_b) == 4
        assert all(a.n_tokens <= 3 for a in attempts_a)
        assert all(b.n_tokens <= 9 for b in attempts_b)
    finally:
        adapter.close()


def test_sample_is_deterministic_for_the_same_request_list(tiny_ckpt: Path) -> None:
    """Same `reqs` list in, same attempts out (module docstring's determinism contract): the
    generator is seeded from a hash of every request's own `seed`, in order."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=64)
    try:
        proofs = load_valid_proofs()
        reqs = [
            SampleRequest(
                pid="a",
                sequent=Sequent.from_prompt(proofs[0]["prompt"]),
                n=4,
                temperature=1.0,
                max_new_tokens=8,
                seed=0,
            ),
            SampleRequest(
                pid="b",
                sequent=Sequent.from_prompt(proofs[1]["prompt"]),
                n=6,
                temperature=1.0,
                max_new_tokens=8,
                seed=1,
            ),
        ]
        out1 = adapter.sample(reqs)
        out2 = adapter.sample(reqs)
        natives1 = [[a.native for a in attempts] for attempts in out1]
        natives2 = [[a.native for a in attempts] for attempts in out2]
        assert natives1 == natives2
        bodies1 = [[a.body for a in attempts] for attempts in out1]
        bodies2 = [[a.body for a in attempts] for attempts in out2]
        assert bodies1 == bodies2
    finally:
        adapter.close()


# -------------------------------------------------------------------------------------- logprob


def test_logprob_matches_manual_logsumexp_over_shifts(tiny_ckpt: Path) -> None:
    import torch.nn.functional as F

    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[2]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        (result,) = adapter.logprob([(seq, body)])

        tok = adapter._tok
        model = adapter._model
        pid = tok.encode_prompt(seq.prompt())
        qid = tok.encode_proof(body)
        ref0 = tok.ref0
        mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
        manual: list[float] = []
        model.eval()
        with torch.no_grad():
            for s in range(0, MAXN - mx + 1):
                ids = pid + [x + s if x >= ref0 else x for x in qid]
                x = torch.tensor([ids])
                logits = model(x[:, :-1]).float()
                tgt = x[:, 1:]
                logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
                mask = torch.zeros_like(tgt, dtype=torch.bool)
                mask[:, len(pid) - 1 :] = True
                manual.append(float((logp * mask).sum().item()))
        expected = float(torch.logsumexp(torch.tensor(manual), dim=0).item())

        assert result.n_tokens == len(qid)
        assert result.per_encoding == pytest.approx(tuple(manual))
        assert result.nats == pytest.approx(expected)
    finally:
        adapter.close()


def test_length_bucketed_chunks_respects_both_row_and_cell_caps() -> None:
    """A row-count cap alone degenerates to one chunk, padded to the single longest row, whenever
    everything fits under it -- this is exactly the bug that made the batched `logprob` path
    slower than the reference on a small, length-varied CPU workload before the cell cap was
    added. Lengths here (mostly short, one long outlier) must split into more than one chunk under
    a cell cap even though every row fits under a generous row-count cap alone."""
    lengths = [10, 10, 10, 10, 10, 10, 10, 10, 500]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])

    # row cap alone (huge cell budget): everything in one chunk.
    chunks_row_only = _length_bucketed_chunks(order, lengths, max_rows=100, max_cells=10**9)
    assert len(chunks_row_only) == 1
    assert sorted(chunks_row_only[0]) == list(range(len(lengths)))

    # a cell cap that the one-chunk padding would blow (9 rows * 500 = 4500) forces a split; the
    # 500-length outlier must end up isolated from the short rows (bucketing by length).
    chunks_bucketed = _length_bucketed_chunks(order, lengths, max_rows=100, max_cells=600)
    assert len(chunks_bucketed) > 1
    for chunk in chunks_bucketed:
        chunk_max_len = max(lengths[i] for i in chunk)
        assert len(chunk) * chunk_max_len <= 600
        assert len(chunk) <= 100
    all_indices = sorted(i for chunk in chunks_bucketed for i in chunk)
    assert all_indices == list(range(len(lengths)))
    outlier_chunk = next(chunk for chunk in chunks_bucketed if 8 in chunk)
    assert outlier_chunk == [8]  # the length-500 row gets its own chunk, not padded alongside 10s


def test_logprob_batched_matches_reference_on_all_fixture_proofs(tiny_ckpt: Path) -> None:
    """Throughput fix: `logprob` now batches every shift of every pair into a handful of forward
    calls instead of one per (pair, shift). Must agree with the original one-row-per-call
    `_shift_marginal_logprob_reference` to within 1e-4 nats, on every one of the 60 fixture
    proofs in a single batched call (not just pairwise)."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        proofs = load_valid_proofs()
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = adapter.logprob(pairs)
        assert len(batched) == len(pairs)
        for (seq, body), lp in zip(pairs, batched, strict=True):
            ref = _shift_marginal_logprob_reference(
                adapter._model, adapter._tok, adapter._device, seq, body, adapter, MAXN
            )
            assert lp.n_tokens == ref.n_tokens
            assert len(lp.per_encoding) == len(ref.per_encoding)
            assert lp.nats == pytest.approx(ref.nats, abs=1e-4)
            # Per-shift sums are large-magnitude (hundreds of nats, for a ten-ish-line proof): a
            # relative tolerance is appropriate here, since float32 accumulation over many summed
            # log-probs naturally differs slightly between a batched and an unbatched computation
            # order at that magnitude's own precision floor -- `nats` (the logsumexp of these,
            # checked above) is the value every caller actually consumes, and is dominated by
            # whichever shift(s) have the least negative value, so it is far less sensitive to this
            # per-shift noise than a raw per-element comparison would be.
            for a, b in zip(lp.per_encoding, ref.per_encoding, strict=True):
                assert a == pytest.approx(b, rel=1e-5, abs=1e-4)
    finally:
        adapter.close()


def test_logprob_batched_with_small_chunk_size_still_matches_reference(tiny_ckpt: Path) -> None:
    """`max_batch_rows` small enough that a single pair's own shifts, and different pairs, are
    split across several forward calls -- the regrouping/log-sum-exp must still match the
    reference exactly (within tolerance)."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        proofs = load_valid_proofs()[:8]
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = _batched_shift_marginal_logprobs(
            adapter._model, adapter._tok, adapter._device, adapter, MAXN, pairs, max_batch_rows=3
        )
        for (seq, body), lp in zip(pairs, batched, strict=True):
            ref = _shift_marginal_logprob_reference(
                adapter._model, adapter._tok, adapter._device, seq, body, adapter, MAXN
            )
            assert lp.nats == pytest.approx(ref.nats, abs=1e-4)
    finally:
        adapter.close()


def test_nll_batched_matches_reference_on_all_fixture_proofs(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        learner = adapter.learner(seed=0)
        proofs = load_valid_proofs()
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = learner.nll(pairs)
        for (seq, body), lp in zip(pairs, batched, strict=True):
            ref = _shift_marginal_nll_reference(
                learner._model, learner._tok, learner._device, seq, body, learner, MAXN
            )
            assert lp.nats == pytest.approx(ref.nats, abs=1e-4)
    finally:
        adapter.close()


def test_nll_start_excluded_batched_matches_reference(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        learner = adapter.learner(seed=0)
        proofs = load_valid_proofs()[:10]
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = learner.nll_start_excluded(pairs)
        for (seq, body), lp in zip(pairs, batched, strict=True):
            ref = _shift_marginal_nll_reference(
                learner._model,
                learner._tok,
                learner._device,
                seq,
                body,
                learner,
                MAXN,
                start_excluded=True,
            )
            assert lp.n_tokens == ref.n_tokens
            assert lp.nats == pytest.approx(ref.nats, abs=1e-4)
    finally:
        adapter.close()


def test_logprob_batched_nlls_helper_matches_reference(tiny_ckpt: Path) -> None:
    """`_batched_shift_marginal_nlls` (used by `Learner.nll`/`nll_start_excluded`) directly."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        proofs = load_valid_proofs()[:6]
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = _batched_shift_marginal_nlls(
            adapter._model, adapter._tok, adapter._device, adapter, MAXN, pairs
        )
        for (seq, body), lp in zip(pairs, batched, strict=True):
            ref = _shift_marginal_nll_reference(
                adapter._model, adapter._tok, adapter._device, seq, body, adapter, MAXN
            )
            assert lp.nats == pytest.approx(ref.nats, abs=1e-4)
    finally:
        adapter.close()


def test_logprob_steering_is_correct_per_pair_when_batched_with_different_prompt_lengths(
    tiny_ckpt: Path,
) -> None:
    """Fix: a batched `logprob` call can mix pairs with different prompt lengths in the same
    forward call; the per-row `_steer_prompt_len` tensor must still give each pair exactly the
    same prompt/proof split a lone `logprob([(seq, body)])` call on that pair would -- a bug that
    used a single shared threshold for the whole batch would misclassify at least one of the two
    pairs' positions whenever their prompt lengths differ."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        d = adapter._cfg["d"]
        # A varying vector: LayerNorm's mean-subtraction would exactly cancel a constant shift.
        vec = tuple(float(i) - d / 2 for i in range(d))
        term = SteeringTerm(layer=0, vector=vec, alpha=5.0, where="prompt")
        steered = adapter.with_steering((term,))
        try:
            proofs = load_valid_proofs()
            seq_a, body_a = Sequent.from_prompt(proofs[0]["prompt"]), proofs[0]["proof"]
            seq_b, body_b = Sequent.from_prompt(proofs[3]["prompt"]), proofs[3]["proof"]
            len_a = len(steered._tok.encode_prompt(seq_a.prompt()))
            len_b = len(steered._tok.encode_prompt(seq_b.prompt()))
            assert len_a != len_b, "fixture choice needs two different prompt lengths"

            batched = steered.logprob([(seq_a, body_a), (seq_b, body_b)])
            (alone_a,) = steered.logprob([(seq_a, body_a)])
            (alone_b,) = steered.logprob([(seq_b, body_b)])

            assert batched[0].nats == pytest.approx(alone_a.nats, abs=1e-4)
            assert batched[1].nats == pytest.approx(alone_b.nats, abs=1e-4)
            # sanity: the steering term actually does something, so this isn't a vacuous pass.
            (unsteered_a,) = adapter.logprob([(seq_a, body_a)])
            assert batched[0].nats != pytest.approx(unsteered_a.nats, abs=1e-4)
        finally:
            steered.close()
    finally:
        adapter.close()


# --------------------------------------------------------------------------------------- learner


def test_update_lowers_nll_on_a_repeated_example(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4, update_shifts=4)
    try:
        proof = load_valid_proofs()[3]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        learner = adapter.learner(seed=0)
        (before,) = learner.nll([(seq, body)])
        for _ in range(30):
            learner.update([(seq, body)], lr=1e-2)
        (after,) = learner.nll([(seq, body)])
        # `Learner.nll`'s `nats` is a positive negative-log-likelihood (see `core.types.LogProb`'s
        # docstring: the opposite sign from `ModelAdapter.logprob`'s log-probability), so training
        # on this exact example must decrease it.
        assert after.nats < before.nats
        assert before.nats >= 0.0
        assert after.nats >= 0.0
    finally:
        adapter.close()


def test_nll_start_excluded_matches_nll_minus_the_start_token(tiny_ckpt: Path) -> None:
    """WP5's start-token-excluded sensitivity variant (plan §B7), implemented on the real
    `AbsGptAdapter` Learner (fix list item 3): `nll_start_excluded` should equal an independent,
    from-scratch shift-marginal NLL computed with the first proof token's log-probability dropped
    from every shift before marginalising, and should report `n_tokens` one shorter."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[1]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        learner = adapter.learner(seed=0)

        full = learner.nll([(seq, body)])[0]
        excluded = learner.nll_start_excluded([(seq, body)])[0]
        assert excluded.n_tokens == full.n_tokens - 1
        assert full.nats >= 0.0
        assert excluded.nats >= 0.0

        # Independent replica: manually drop the first proof token's log-probability from every
        # shift, then marginalise, then negate (the `Learner.nll` sign convention).
        tok = learner._tok  # type: ignore[attr-defined]
        model = learner._model  # type: ignore[attr-defined]
        pid = tok.encode_prompt(seq.prompt())
        qid = tok.encode_proof(body)
        ref0 = tok.ref0
        mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
        manual: list[float] = []
        model.eval()
        with torch.no_grad():
            for s in range(0, MAXN - mx + 1):
                ids = pid + [x + s if x >= ref0 else x for x in qid]
                x = torch.tensor([ids], dtype=torch.long)
                logits = model(x[:, :-1]).float()
                tgt = x[:, 1:]
                logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
                mask = torch.zeros_like(tgt, dtype=torch.bool)
                mask[:, len(pid) :] = True  # excludes the first proof token vs. `len(pid) - 1 :`
                manual.append(float((logp * mask).sum().item()))
        expected_nll = -float(torch.logsumexp(torch.tensor(manual), dim=0).item())
        assert excluded.nats == pytest.approx(expected_nll)
    finally:
        adapter.close()


def test_learner_update_shifts_is_public_and_settable(tiny_ckpt: Path) -> None:
    """Fix list item 2 (EDL double shift-copying): the learner's shift count must be a plain,
    settable attribute `edl.run_edl` can read/override with `getattr`/`setattr`, not buried behind
    a private name -- and it must default to the adapter's own `update_shifts` constructor arg."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4, update_shifts=3)
    try:
        learner = adapter.learner(seed=0)
        assert learner.update_shifts == 3
        learner.update_shifts = 5  # run_edl does exactly this before training
        assert learner.update_shifts == 5
    finally:
        adapter.close()


def test_learner_adapter_samples_with_updated_weights(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4, update_shifts=2)
    try:
        proof = load_valid_proofs()[4]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        learner = adapter.learner(seed=0)
        learner.update([(seq, body)], lr=1e-2)
        learner_adapter = learner.adapter()
        assert learner_adapter.identity()["derived_from_learner"] is True
        req = SampleRequest(pid="p", sequent=seq, n=2, temperature=1.0, max_new_tokens=5, seed=0)
        (attempts,) = learner_adapter.sample([req])
        assert len(attempts) == 2
    finally:
        adapter.close()


def test_update_gradient_is_the_mean_over_shift_copies_not_the_sum(
    tiny_ckpt: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Learner.update` already batches all `update_shifts` copies of every example into one
    forward+backward (no per-copy loop); this pins the *averaging* semantics that batching must
    preserve. Monkeypatching `_rng.randint` to always return `0` makes every one of a pair's
    `update_shifts` copies an identical row (shift 0 leaves `ids` unchanged), which makes the
    *correctly averaged* loss -- and so the gradient -- independent of `update_shifts`: `n_tok`
    and the summed log-prob both scale by the same factor `update_shifts`, which cancels in the
    ratio. An accidental sum instead of a mean would instead scale the gradient with
    `update_shifts`, which this test would catch."""
    proofs = load_valid_proofs()[:3]
    pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]

    adapter1 = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", update_shifts=1)
    adapter5 = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", update_shifts=5)
    try:
        learner1 = adapter1.learner(seed=0)
        learner5 = adapter5.learner(seed=0)
        monkeypatch.setattr(learner1._rng, "randint", lambda a, b: 0)
        monkeypatch.setattr(learner5._rng, "randint", lambda a, b: 0)

        loss1 = learner1.update(pairs, lr=1e-2)
        grads1 = [p.grad.clone() for p in learner1._model.parameters() if p.grad is not None]

        loss5 = learner5.update(pairs, lr=1e-2)
        grads5 = [p.grad.clone() for p in learner5._model.parameters() if p.grad is not None]

        assert loss1 == pytest.approx(loss5, abs=1e-5)
        assert len(grads1) == len(grads5) > 0
        for g1, g5 in zip(grads1, grads5, strict=True):
            assert torch.allclose(g1, g5, atol=1e-5)
    finally:
        adapter1.close()
        adapter5.close()


# -------------------------------------------------------------------------------------- steering


def test_steering_zero_vector_noop_nonzero_changes_and_identity_differs(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    steered_zero = None
    steered = None
    try:
        proof = load_valid_proofs()[5]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        d = adapter._cfg["d"]

        # A vector with the *same* value in every dimension is a degenerate choice: LayerNorm's
        # mean-subtraction cancels a constant shift across features exactly, so it would look like
        # a no-op regardless of whether steering worked. Use a varying vector instead.
        varying_vec = tuple(float(i) - d / 2 for i in range(d))

        zero_vec = tuple(0.0 for _ in range(d))
        zero_term = SteeringTerm(layer=0, vector=zero_vec, alpha=1.0, where="all")
        steered_zero = adapter.with_steering((zero_term,))
        (base_lp,) = adapter.logprob([(seq, body)])
        (zero_lp,) = steered_zero.logprob([(seq, body)])
        assert zero_lp.nats == pytest.approx(base_lp.nats)

        nonzero_term = SteeringTerm(layer=0, vector=varying_vec, alpha=2.0, where="all")
        steered = adapter.with_steering((nonzero_term,))
        (nz_lp,) = steered.logprob([(seq, body)])
        assert nz_lp.nats != pytest.approx(base_lp.nats)

        assert steered.identity() != adapter.identity()
        assert steered.identity() != steered_zero.identity()
    finally:
        adapter.close()
        if steered_zero is not None:
            steered_zero.close()
        if steered is not None:
            steered.close()


def test_steering_prompt_vs_proof_where_only_affects_selected_positions(tiny_ckpt: Path) -> None:
    """A `where="prompt"` steering term must not change the logits used to score the proof tokens
    when the proof-token positions receive no addition -- i.e. it must change *something* (the
    prompt-position activations feed forward through attention) but the mechanism should at least
    only ever touch positions `< prompt_len` directly. This checks the hook's own position split by
    comparing `where="all"` (every position) against `where="proof"` (a strict subset): the two must
    disagree once alpha is nonzero."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    steered_all = None
    steered_proof = None
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        d = adapter._cfg["d"]
        # A uniform vector is exactly cancelled by LayerNorm's mean-subtraction (see the note in
        # the test above); use a varying one so the intervention is actually visible.
        vec = tuple(float(i) - d / 2 for i in range(d))

        steered_all = adapter.with_steering(
            (SteeringTerm(layer=0, vector=vec, alpha=3.0, where="all"),)
        )
        steered_proof = adapter.with_steering(
            (SteeringTerm(layer=0, vector=vec, alpha=3.0, where="proof"),)
        )
        (lp_all,) = steered_all.logprob([(seq, body)])
        (lp_proof,) = steered_proof.logprob([(seq, body)])
        assert lp_all.nats != pytest.approx(lp_proof.nats)
    finally:
        adapter.close()
        if steered_all is not None:
            steered_all.close()
        if steered_proof is not None:
            steered_proof.close()


def test_steering_position_recovery_with_mixed_length_prompts_in_one_batch(tiny_ckpt: Path) -> None:
    """Cross-request batching (`sample()`'s throughput fix) means one `_gen` call can now mix rows
    of different real prompt lengths, something that never happened when `_gen` was only ever
    called with `n` copies of a single request (uniform length, zero left padding). The steering
    hook must recover each row's own real (left-padding-adjusted) position from the shared,
    left-padded KV-cache batch rather than one position range shared by the whole batch.

    This replicates `_gen`'s own left-padded/cached prefill call shape directly against `model` +
    hooks (independent of `sample()`'s own grouping) for two rows of different real lengths (3 and
    5 tokens, padded to `L=5`), and checks that a `where="prompt"` steering term changes exactly
    the columns that are really `< prompt_len` *for that row* -- including row 0's last two
    (padded) columns, which a shared, batch-wide column count would misclassify as absolute
    positions 3 and 4 (`>= prompt_len`) purely because of their later column index.
    """
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=8)
    try:
        d = adapter._cfg["d"]
        # A varying vector: LayerNorm's mean-subtraction would exactly cancel a constant shift (see
        # the note in `test_steering_zero_vector_noop_nonzero_changes_and_identity_differs`).
        vec = tuple(float(i) - d / 2 for i in range(d))
        term = SteeringTerm(layer=0, vector=vec, alpha=5.0, where="prompt")
        steered = adapter.with_steering((term,))
        try:
            model = steered._model
            tok = steered._tok
            L, max_new = 5, 1
            lens = [3, 5]  # row 0: 3 real tokens (2 left-pad columns); row 1: 5 (no padding)
            tok_id = tok._ref(1)
            idx = torch.full((2, L), tok.pad, dtype=torch.long)
            keep = torch.zeros((2, L), dtype=torch.bool)
            for i, n in enumerate(lens):
                idx[i, L - n :] = tok_id
                keep[i, L - n :] = True
            pos = (keep.cumsum(1) - 1).clamp(min=0)
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
            mask = causal[None, None] & keep[:, None, None, :]
            mask = mask | torch.eye(L, dtype=torch.bool)[None, None]

            steered._steer_prompt_len = 3
            steered_caches = [{"max": L + max_new} for _ in model.blocks]
            with torch.no_grad():
                steered_logits = model(idx, pos=pos, mask=mask, caches=steered_caches)
            steered._steer_prompt_len = 0

            base_caches = [{"max": L + max_new} for _ in adapter._model.blocks]
            with torch.no_grad():
                base_logits = adapter._model(idx, pos=pos, mask=mask, caches=base_caches)

            # Row 0's 3 real tokens live at columns 2,3,4 with real positions 0,1,2 -- all < 3, so
            # *every* real column must be steered, including the last two.
            assert not torch.allclose(steered_logits[0, 2], base_logits[0, 2])
            assert not torch.allclose(steered_logits[0, 3], base_logits[0, 3])
            assert not torch.allclose(steered_logits[0, 4], base_logits[0, 4])

            # Row 1 has no padding (real positions 0..4 at columns 0..4): columns 0,1,2 (< 3) are
            # steered, columns 3,4 (>= 3) are not.
            assert not torch.allclose(steered_logits[1, 0], base_logits[1, 0])
            assert not torch.allclose(steered_logits[1, 2], base_logits[1, 2])
            assert torch.allclose(steered_logits[1, 3], base_logits[1, 3])
            assert torch.allclose(steered_logits[1, 4], base_logits[1, 4])
        finally:
            steered.close()
    finally:
        adapter.close()


# ----------------------------------------------------------------------------------- activations


def test_activations_shape(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=4)
    try:
        proof = load_valid_proofs()[0]
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        total_len = adapter.token_count(seq, body)
        (arr,) = adapter.activations([(seq, body)], layers=[0])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, total_len, adapter._cfg["d"])
    finally:
        adapter.close()


def test_activations_batched_matches_per_pair_unbatched(tiny_ckpt: Path) -> None:
    """Throughput fix: `activations` batches every pair into one (padded) forward call; each
    pair's own sliced-out array must be identical to calling `activations` on that pair alone
    (different proof lengths -- and so different amounts of padding -- across the fixtures, and
    more than one requested layer, so both the padding-strip and the per-layer stacking are
    exercised)."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        proofs = load_valid_proofs()[:8]
        assert len({p["n_lines"] for p in proofs}) > 1, "fixture choice needs varying proof lengths"
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        layers = [0] if adapter._cfg["n_layer"] == 1 else [0, adapter._cfg["n_layer"] - 1]
        batched = adapter.activations(pairs, layers=layers)
        assert len(batched) == len(pairs)
        for (seq, body), arr_batched in zip(pairs, batched, strict=True):
            (arr_alone,) = adapter.activations([(seq, body)], layers=layers)
            assert arr_batched.shape == arr_alone.shape
            np.testing.assert_allclose(arr_batched, arr_alone, atol=1e-4)
    finally:
        adapter.close()


def test_activations_batched_with_small_chunk_size_still_matches_unbatched(tiny_ckpt: Path) -> None:
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        proofs = load_valid_proofs()[:6]
        pairs = [(Sequent.from_prompt(p["prompt"]), p["proof"]) for p in proofs]
        batched = adapter.activations(pairs, layers=[0], max_batch_rows=2)
        for (seq, body), arr_batched in zip(pairs, batched, strict=True):
            (arr_alone,) = adapter.activations([(seq, body)], layers=[0])
            np.testing.assert_allclose(arr_batched, arr_alone, atol=1e-4)
    finally:
        adapter.close()


# ------------------------------------------------------------------------------ real checkpoints


@pytest.mark.slow
def test_base_checkpoint_loads_on_cpu_with_expected_sha() -> None:
    if not os.path.isfile(BASE_CKPT):
        pytest.skip(f"base checkpoint not available at {BASE_CKPT} in this environment")
    adapter = AbsGptAdapter(ckpt=BASE_CKPT, device="cpu")
    try:
        sha = adapter.identity()["ckpt_sha256"]
        assert isinstance(sha, str)
        assert sha.startswith(BASE_SHA_PREFIX)
    finally:
        adapter.close()


@pytest.mark.slow
def test_rl_checkpoint_loads_on_cpu_with_expected_sha() -> None:
    if not os.path.isfile(RL_CKPT):
        pytest.skip(f"RL checkpoint not available at {RL_CKPT} in this environment")
    adapter = AbsGptAdapter(ckpt=RL_CKPT, device="cpu")
    try:
        sha = adapter.identity()["ckpt_sha256"]
        assert isinstance(sha, str)
        assert sha.startswith(RL_SHA_PREFIX)
    finally:
        adapter.close()


@pytest.mark.slow
def test_simple_theorem_verifies_within_64_samples_from_rl_model() -> None:
    if not os.path.isfile(RL_CKPT):
        pytest.skip(f"RL checkpoint not available at {RL_CKPT} in this environment")
    from code.ndbench.judge.nd import NDJudge

    adapter = AbsGptAdapter(ckpt=RL_CKPT, device="cpu", batch=64)
    try:
        # Spec format: every token (including parentheses) is whitespace-separated.
        seq = Sequent(premises=("( P > Q )", "P"), conclusion="Q")
        req = SampleRequest(
            pid="thm", sequent=seq, n=64, temperature=0.8, max_new_tokens=40, seed=0
        )
        (attempts,) = adapter.sample([req])
        judge = NDJudge()
        verdicts = judge.judge(seq, attempts, ref_lines=3)
        assert any(v.ok for v in verdicts)
    finally:
        adapter.close()


def test_sampling_sets_per_row_prompt_boundary_for_steering(tiny_ckpt: Path) -> None:
    """Audit 2026-10-01: `sample()` never set the prompt boundary, so during generation a
    `where="proof"` term also steered the prompt and a `where="prompt"` term did nothing. Every
    `_gen` prefill must now see each row's own prompt length (forced prefix excluded), and the
    boundary must be cleared after the call. Mixed prompt lengths and early-finishing rows also
    exercise `_gen`'s mid-decode batch compaction against the per-row tensor."""
    adapter = AbsGptAdapter(ckpt=tiny_ckpt, device="cpu", batch=8)
    d = adapter._cfg["d"]
    steered = adapter.with_steering(
        (SteeringTerm(layer=0, vector=tuple(0.01 * i for i in range(d)), alpha=1.0, where="proof"),)
    )
    seen: list[tuple[int, list[int]]] = []

    def record(_m: Any, _i: Any, output: torch.Tensor) -> None:
        pl = steered._steer_prompt_len
        if output.shape[1] > 1:  # prefill
            assert isinstance(pl, torch.Tensor)
            seen.append((output.shape[0], [int(x) for x in pl.tolist()]))

    handle = steered._model.blocks[0].register_forward_hook(record)
    try:
        proofs = load_valid_proofs()
        seqs = [Sequent.from_prompt(proofs[i]["prompt"]) for i in (0, 5)]
        reqs = [
            SampleRequest(pid=f"p{i}", sequent=q, n=3, temperature=1.0, max_new_tokens=24, seed=i)
            for i, q in enumerate(seqs)
        ]
        steered.sample(reqs)
        want = sorted(len(steered._tok.encode_prompt(q.prompt())) for q in seqs for _ in range(3))
        assert seen and sorted(x for _b, lens in seen for x in lens) == want
        assert steered._steer_prompt_len == 0
    finally:
        handle.remove()
        adapter.close()
        steered.close()
