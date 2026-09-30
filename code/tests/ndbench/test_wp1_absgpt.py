"""`AbsGptAdapter` (plan §3 WP1): a tiny random CPU GPT built from the vendored model class with
the real tokenizer, plus (marked `slow`) the two real checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from code.ndbench.adapters.absgpt import GPT, MAXN, AbsGptAdapter, Tokenizer
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
