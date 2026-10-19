"""`DanStateAdapter` (plan `DANSTATE_PLAN.md` §1/§7 WP-D1): a tiny random CPU `ALiBiGPT` built
from the vendored class, exercising V3 rollout parity against vendored `env_generate(gate=False)`,
the `Attempt`/`native` contract, `STATEPREFIX` replay, `logprob` (vs. a manual teacher-forced sum,
the base convention, and the NaN-on-bad-grammar cases), activations, steering, identity and kwarg
coercion -- plus (marked `slow`) V4 against the real T1 checkpoint."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from code.ndbench.adapters.base import Capability, SteeringTerm
from code.ndbench.adapters.danstate import (
    DanStateAdapter,
    _cross_request_seed,
    _defining_name_positions,
    _kv_bytes_per_token,
    _kv_safe_chunks,
    _parse_state_prefix,
    _rollout,
    _shift_name_tokens,
    _uncited_box_lines,
)
from code.ndbench.core.types import PrefixSpec, SampleRequest, Sequent
from code.ndbench.judge.nd import NDJudge
from code.ndbench.vendor import load_danstate, load_nd_verify

REAL_CKPT = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/ladder/la_T1_best12_s0_r8.pt"
TEXTBOOK_DEV_SUITE = Path("/scratch/cowifornia/nd-rl/bench/out/suites/textbook_dev/v1/items.jsonl")

# Hand-written prompts (our prompt format == Dan's, plan §0) exercising PR, box-opening/closing,
# and Or.elim.
_SIMPLE = ("THM ( P & Q ) SEQ P PRF", "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED")
_IMP = (
    "THM ( P & Q ) SEQ ( P > Q ) PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 | Q : ANDE2 N1 ; N4 ( P > Q ) : IMPI N2 N3 ; QED",
)
_OR_ELIM = (
    "THM ( P v Q ) SEQ ( Q v P ) PRF",
    "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; "
    "N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED",
)
_PROMPTS = [_SIMPLE[0], _IMP[0], _OR_ELIM[0]]


def _tiny_cfg() -> dict[str, Any]:
    return dict(vocab=111, n_layer=2, d=128, n_head=4, max_len=128, d_ff=256, arch="best")


def _native(a: Any) -> dict[str, Any]:
    """`json.loads(a.native)`, narrowing `Attempt.native`'s `str | None` type for pyright (every
    `DanStateAdapter` attempt always has a `native` string -- see the module docstring)."""
    assert a.native is not None
    return json.loads(a.native)


def _make_tiny_checkpoint(tmp_path: Path, seed: int = 0) -> Path:
    ds = load_danstate()
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    model = ds.best_model.ALiBiGPT(**cfg)
    path = tmp_path / "tiny.pt"
    torch.save(
        {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_staten", "extra": {}},
        path,
    )
    return path


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> Path:
    return _make_tiny_checkpoint(tmp_path)


@pytest.fixture
def adapter(tiny_ckpt: Path) -> Generator[DanStateAdapter, None, None]:
    a = DanStateAdapter(
        ckpt=tiny_ckpt, device="cpu", batch=16, max_action=24, max_steps=10, name_base_max=4
    )
    try:
        yield a
    finally:
        a.close()


# ------------------------------------------------------------------------------------- V3 parity


def test_rollout_matches_vendored_env_generate_gate_false(tiny_ckpt: Path) -> None:
    """`_rollout` is a line-for-line port of `env_generate(:35-98)` minus gate/record, plus the KV
    guard (inert here -- `kv_budget_gb` huge). Using Dan's own per-attempt base rule
    (`random.Random(seed*1000003+nxt).randint(0, NAME_BASE_MAX)`) via `make_env`, `_rollout`'s `nd`
    output must equal vendored `env_generate(gate=False)` exactly, same prompts/seed/batch."""
    ds = load_danstate()
    model, tok, _cfg, _meta = _load(tiny_ckpt)
    NAME_BASE_MAX = ds.state_sample.NAME_BASE_MAX
    prompts = _PROMPTS * 3  # repeat so the worklist refill logic is actually exercised
    seed = 7
    batch = 2

    expected = ds.state_sample.env_generate(
        model,
        tok,
        prompts,
        greedy=False,
        temperature=1.0,
        max_action=20,
        max_steps=6,
        batch=batch,
        seed=seed,
        stats={},
        gate=False,
    )

    def make_env(pos: int) -> Any:
        base = random.Random(seed * 1000003 + pos).randint(0, NAME_BASE_MAX)
        return ds.state_env.Env(prompts[pos], canon=True, base=base, assign=True)

    bytes_per_token = _kv_bytes_per_token(_cfg_of(model), torch.device("cpu"))
    got = _rollout(
        model,
        tok,
        prompts,
        make_env,
        greedy=False,
        temperature=1.0,
        max_action=20,
        max_steps=6,
        batch=batch,
        seed=seed,
        kv_budget_gb=1000.0,
        bytes_per_kv_token=bytes_per_token,
    )
    assert [g.nd for g in got] == expected


def test_rollout_matches_vendored_env_generate_greedy(tiny_ckpt: Path) -> None:
    ds = load_danstate()
    model, tok, _cfg, _meta = _load(tiny_ckpt)
    NAME_BASE_MAX = ds.state_sample.NAME_BASE_MAX
    prompts = _PROMPTS
    seed = 3
    batch = 4

    expected = ds.state_sample.env_generate(
        model,
        tok,
        prompts,
        greedy=True,
        temperature=1.0,
        max_action=18,
        max_steps=5,
        batch=batch,
        seed=seed,
        stats={},
        gate=False,
    )

    def make_env(pos: int) -> Any:
        base = random.Random(seed * 1000003 + pos).randint(0, NAME_BASE_MAX)
        return ds.state_env.Env(prompts[pos], canon=True, base=base, assign=True)

    bytes_per_token = _kv_bytes_per_token(_cfg_of(model), torch.device("cpu"))
    got = _rollout(
        model,
        tok,
        prompts,
        make_env,
        greedy=True,
        temperature=1.0,
        max_action=18,
        max_steps=5,
        batch=batch,
        seed=seed,
        kv_budget_gb=1000.0,
        bytes_per_kv_token=bytes_per_token,
    )
    assert [g.nd for g in got] == expected


def _load(ckpt: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    from code.ndbench.adapters._danstate_load import load_dan_checkpoint

    return load_dan_checkpoint(ckpt)


def _cfg_of(model: Any) -> dict[str, Any]:
    return dict(model.cfg)


# ------------------------------------------------------------------------------- Attempt contract


def test_sample_returns_right_shape_and_native_contract(adapter: DanStateAdapter) -> None:
    seq = Sequent.from_prompt(_SIMPLE[0])
    req = SampleRequest(pid="p", sequent=seq, n=5, temperature=1.0, max_new_tokens=999, seed=1)
    (attempts,) = adapter.sample([req])
    assert len(attempts) == 5
    for i, a in enumerate(attempts):
        assert a.pid == "p"
        assert a.i == i
        assert a.native is not None
        native = _native(a)
        assert set(native) == {"lean", "base", "steps", "end"}
        assert native["end"] in ("done", "truncated", "step_cap") or native["end"].startswith(
            "syntax:"
        )
        assert 0 <= native["base"] <= adapter._name_base_max
        if native["end"] in ("truncated", "step_cap"):
            assert a.finished is False
        else:
            assert a.finished is True
        if a.body.startswith("UNPARSEABLE"):
            assert native["lean"] is None
        else:
            assert isinstance(native["lean"], str)
            assert a.body.endswith("QED")


def test_native_lean_matches_gate_texts_contract(adapter: DanStateAdapter) -> None:
    """`native["lean"]` must be exactly what `env_generate(gate=True)` would hand to
    `lean_gate.gate` as `texts[i]` (module docstring): `tok.text(e.text)` when the env finished and
    `inverse` parsed, else `None`. Forces a deterministic "done" attempt via a full-length
    `STATEPREFIX` (the model has nothing left to sample) rather than hoping a tiny random model
    finishes on its own; cross-checked against an independent `state_env.decompose` replay of the
    resulting body, rendered with `tok.text` ourselves."""
    ds = load_danstate()
    prompt, ref_body = _SIMPLE
    toks = ds.state_env.canonicalise(prompt, ref_body)
    m = len(ds.state_env.split_actions(toks))

    seq = Sequent.from_prompt(prompt)
    prefix = PrefixSpec(j=m, body=f"STATEPREFIX m={m} | {ref_body}")
    req = SampleRequest(
        pid="p", sequent=seq, n=3, temperature=1.0, max_new_tokens=999, seed=2, prefix=prefix
    )
    (attempts,) = adapter.sample([req])
    tok = ds.lean_tok.LeanTokenizer("lean_staten")
    for a in attempts:
        native = _native(a)
        assert native["end"] == "done"
        assert native["lean"] is not None
        # `a.body` is the ND proof `inverse(...)` denotes, which is base-independent (ND line
        # indices, not lean names) -- re-`decompose`ing it replays at base 0 by default, *not*
        # this attempt's own base `b`. The expected literal Lean text is instead the base-0
        # canonical tokens shifted to base `b` directly (exactly what `_replay_prefix_env`/
        # `Env._assign_names` under `assign=True` produces, verified by `env.renamed == 0`).
        b = native["base"]
        shifted = _shift_name_tokens(toks, b)
        assert native["lean"] == tok.text(shifted)
        # And the ND proof it denotes must still be the same reference proof.
        assert ds.lean_tok.inverse(shifted) == ref_body


# ------------------------------------------------------------------------------- STATEPREFIX


def test_prefix_replay_reproduces_reference_then_continues(adapter: DanStateAdapter) -> None:
    """Plan §1's explicit "replaying all actions of a reference reproduces it": a full-length
    `STATEPREFIX` (`m` == the reference's own action count) must give back exactly `ref_body`
    (an N1-based ND proof is base-independent, so this holds for *every* draw's own random base
    `b`), with `end == "done"` and `native["lean"]` the base-`b` shift of the reference's own
    canonical rendering."""
    ds = load_danstate()
    tok = ds.lean_tok.LeanTokenizer("lean_staten")
    prompt, ref_body = _IMP
    toks = ds.state_env.canonicalise(prompt, ref_body)
    acts = ds.state_env.split_actions(toks)
    m = len(acts)  # full replay: the prefix is the whole proof

    seq = Sequent.from_prompt(prompt)
    prefix = PrefixSpec(j=m, body=f"STATEPREFIX m={m} | {ref_body}")
    req = SampleRequest(
        pid="p", sequent=seq, n=3, temperature=1.0, max_new_tokens=999, seed=5, prefix=prefix
    )
    (attempts,) = adapter.sample([req])
    for a in attempts:
        native = _native(a)
        assert native["end"] == "done"
        assert native["steps"] == m
        assert a.body == ref_body
        assert native["lean"] == tok.text(_shift_name_tokens(toks, native["base"]))


def test_prefix_partial_replay_then_greedy_continuation(adapter: DanStateAdapter) -> None:
    ds = load_danstate()
    prompt, ref_body = _OR_ELIM
    toks = ds.state_env.canonicalise(prompt, ref_body)
    acts = ds.state_env.split_actions(toks)
    m = 2
    assert m < len(acts)

    seq = Sequent.from_prompt(prompt)
    prefix = PrefixSpec(j=m, body=f"STATEPREFIX m={m} | {ref_body}")
    req = SampleRequest(
        pid="p", sequent=seq, n=2, temperature=0.0, max_new_tokens=999, seed=9, prefix=prefix
    )
    (attempts,) = adapter.sample([req])
    for a in attempts:
        native = _native(a)
        assert native["steps"] >= m


def test_unsupported_prefix_body_raises_not_supported(adapter: DanStateAdapter) -> None:
    from code.ndbench.adapters.base import NotSupported

    seq = Sequent.from_prompt(_SIMPLE[0])
    prefix = PrefixSpec(j=1, body="N1 P : PR ;")  # an ordinary ND-line prefix, not STATEPREFIX
    req = SampleRequest(
        pid="p", sequent=seq, n=1, temperature=1.0, max_new_tokens=10, seed=0, prefix=prefix
    )
    with pytest.raises(NotSupported):
        adapter.sample([req])


def test_parse_state_prefix() -> None:
    m, body = _parse_state_prefix("STATEPREFIX m=3 | N1 P : PR ; QED")
    assert m == 3
    assert body == "N1 P : PR ; QED"


# ------------------------------------------------------------------------------------- logprob


def _manual_step_logprob(
    model: Any, tok: Any, state_toks: list[str], action_toks: list[str]
) -> float:
    pre = tok.encode_toks(state_toks)
    post = tok.encode_toks(action_toks) + [tok.eos]
    ids = pre + post
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x[:, :-1]).float()
    tgt = x[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
    mask_start = len(pre) - 1
    return float(logp[0, mask_start:].sum().item())


def test_logprob_matches_manual_teacher_forced_sum_and_base_convention(
    adapter: DanStateAdapter,
) -> None:
    ds = load_danstate()
    prompt, body = _IMP
    seq = Sequent.from_prompt(prompt)
    (lp,) = adapter.logprob([(seq, body)])
    assert not math.isnan(lp.nats)

    steps, _toks, env = ds.state_env.decompose(prompt, body, canon=True)
    mx = env.maxname
    name_base_max = adapter._name_base_max
    valid_bases = [b for b in range(0, name_base_max + 1) if mx + b <= 64]
    assert len(lp.per_encoding) == len(valid_bases)

    log_n = math.log(name_base_max + 1)
    manual_per_encoding = []
    for b in valid_bases:
        total = 0.0
        for state_toks, action_toks, _hist in steps:
            s = _shift_name_tokens(state_toks, b)
            a = _shift_name_tokens(action_toks, b)
            total += _manual_step_logprob(adapter._model, adapter._tok, s, a)
        manual_per_encoding.append(total - log_n)

    for got, want in zip(lp.per_encoding, manual_per_encoding, strict=True):
        assert math.isclose(got, want, abs_tol=1e-4)
    manual_nats = float(torch.logsumexp(torch.tensor(manual_per_encoding), dim=0).item())
    assert math.isclose(lp.nats, manual_nats, abs_tol=1e-4)

    expected_n_tokens = sum(len(action) + 1 for _s, action, _h in steps)
    assert lp.n_tokens == expected_n_tokens


def test_logprob_exclude_defining_names_differs_from_default(adapter: DanStateAdapter) -> None:
    prompt, body = _IMP
    seq = Sequent.from_prompt(prompt)
    (lp_default,) = adapter.logprob([(seq, body)])
    (lp_excl,) = adapter.logprob([(seq, body)], exclude_defining_names=True)
    # Excluding tokens from the scored sum can only raise (fewer negative terms summed) or leave
    # unchanged the per-encoding log-probabilities; it must not lower the per-base scores.
    for a, b in zip(lp_default.per_encoding, lp_excl.per_encoding, strict=True):
        assert b >= a - 1e-9


def test_logprob_nan_on_out_of_grammar_pr_order(adapter: DanStateAdapter) -> None:
    # PR lines must be a declared prefix in order; this one re-states a non-existent premise.
    prompt = "THM ( P & Q ) SEQ P PRF"
    bad_body = "N1 ( P & Q ) : PR ; N2 P : PR ; N3 P : ANDE1 N1 ; QED"
    seq = Sequent.from_prompt(prompt)
    (lp,) = adapter.logprob([(seq, bad_body)])
    assert math.isnan(lp.nats)
    assert lp.n_tokens == 0
    assert lp.per_encoding == ()
    assert 0 in adapter.last_logprob_errors


def test_logprob_nan_on_nd_verify_parse_error(adapter: DanStateAdapter) -> None:
    """A `body` malformed enough that `nd_verify.parse_proof_tokens` itself raises
    `nd_verify.ParseError` (checked by `logprob`'s own `allow_reencode=False` pre-check, before
    `state_env.decompose` ever runs) must still come back as a NaN `LogProb`, not propagate --
    regression test for `ParseError` (a different exception type than `lean_tok.ParseFail`) once
    being missing from `logprob`'s `except` tuple."""
    prompt = "THM ( P & Q ) SEQ P PRF"
    bad_body = "N1 ( P & Q ) : NOTARULE ; QED"  # not a recognised rule name
    seq = Sequent.from_prompt(prompt)
    (lp,) = adapter.logprob([(seq, bad_body)])
    assert math.isnan(lp.nats)
    assert 0 in adapter.last_logprob_errors


def test_logprob_nan_on_uncited_box_unless_allow_reencode(adapter: DanStateAdapter) -> None:
    # N2/N3 open and close a box (AS ... ) that is never cited by any IMPI/NEGI/ORE rule: Lean's
    # own rendering (proof_tokens) silently drops it, so re-encoding denotes a shorter, different
    # proof -- rejected by default.
    prompt = "THM ( P & Q ) SEQ P PRF"
    body = "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 P : ANDE1 N1 ; QED"
    seq = Sequent.from_prompt(prompt)
    (lp,) = adapter.logprob([(seq, body)])
    assert math.isnan(lp.nats)
    assert 0 in adapter.last_logprob_errors

    (lp_reencode,) = adapter.logprob([(seq, body)], allow_reencode=True)
    assert not math.isnan(lp_reencode.nats)


def test_canonical_actions_is_base_independent_dedup_key(adapter: DanStateAdapter) -> None:
    prompt, body = _SIMPLE
    seq = Sequent.from_prompt(prompt)
    a1 = adapter.canonical_actions(seq, body)
    a2 = adapter.canonical_actions(seq, body)
    assert a1 == a2
    assert isinstance(a1, tuple) and all(isinstance(x, tuple) for x in a1)


def test_uncited_box_lines_and_defining_name_positions_helpers() -> None:
    nd_verify = load_nd_verify()
    lines = nd_verify.parse_proof_tokens(
        [
            "N1",
            "(",
            "P",
            "&",
            "Q",
            ")",
            ":",
            "PR",
            ";",
            "N2",
            "|",
            "P",
            ":",
            "AS",
            ";",
            "N3",
            "P",
            ":",
            "ANDE1",
            "N1",
            ";",
            "QED",
        ]
    )
    assert _uncited_box_lines(lines) == [2]

    have_atomic = ["have", "n5", ":", "P", ":=", "n1", ";"]
    assert _defining_name_positions(have_atomic) == {1}
    have_box = [
        "have",
        "n5",
        ":",
        "(",
        "P",
        "→",
        "Q",
        ")",
        ":=",
        "(",
        "fun",
        "(",
        "n6",
        ":",
        "P",
        ")",
        "=>",
        "by",
    ]
    assert _defining_name_positions(have_box) == {1, 12}
    exact = ["exact", "n1"]
    assert _defining_name_positions(exact) == set()


# ------------------------------------------------------------------------------------ activations


def test_activations_shapes_for_act_and_action_positions(adapter: DanStateAdapter) -> None:
    ds = load_danstate()
    prompt, body = _IMP
    seq = Sequent.from_prompt(prompt)
    steps, _toks, _env = ds.state_env.decompose(prompt, body, canon=True)
    n_steps = len(steps)
    n_action_tokens = sum(len(a) + 1 for _s, a, _h in steps)

    layers = [0, 1]
    (acts_act,) = adapter.activations([(seq, body)], layers, positions="act")
    assert acts_act.shape == (2, n_steps, adapter._cfg["d"])

    (acts_action,) = adapter.activations([(seq, body)], layers, positions="action")
    assert acts_action.shape == (2, n_action_tokens, adapter._cfg["d"])

    with pytest.raises(ValueError):
        adapter.activations([(seq, body)], layers, positions="bogus")


def test_activations_multiple_pairs_independent_of_batching(adapter: DanStateAdapter) -> None:
    seq1 = Sequent.from_prompt(_SIMPLE[0])
    seq2 = Sequent.from_prompt(_IMP[0])
    (a1,) = adapter.activations([(seq1, _SIMPLE[1])], [0])
    (a2,) = adapter.activations([(seq2, _IMP[1])], [0])
    both = adapter.activations([(seq1, _SIMPLE[1]), (seq2, _IMP[1])], [0])
    assert np.allclose(both[0], a1)
    assert np.allclose(both[1], a2)


# --------------------------------------------------------------------------------------- steering


def test_steering_zero_vector_is_identity(tiny_ckpt: Path) -> None:
    base = DanStateAdapter(ckpt=tiny_ckpt, device="cpu", batch=8, max_action=12, max_steps=6)
    try:
        term = SteeringTerm(layer=0, vector=tuple([0.0] * base._cfg["d"]), alpha=1.0, where="all")
        steered = base.with_steering((term,))
        try:
            seq = Sequent.from_prompt(_SIMPLE[0])
            (a1,) = base.activations([(seq, _SIMPLE[1])], [0])
            (a2,) = steered.activations([(seq, _SIMPLE[1])], [0])
            assert np.allclose(a1, a2, atol=1e-6)
        finally:
            steered.close()
    finally:
        base.close()


def test_steering_nonzero_vector_changes_output(tiny_ckpt: Path) -> None:
    base = DanStateAdapter(ckpt=tiny_ckpt, device="cpu", batch=8, max_action=12, max_steps=6)
    try:
        d = base._cfg["d"]
        vec = tuple([1.0] * d)
        term = SteeringTerm(layer=0, vector=vec, alpha=5.0, where="all")
        steered = base.with_steering((term,))
        try:
            seq = Sequent.from_prompt(_SIMPLE[0])
            (a1,) = base.activations([(seq, _SIMPLE[1])], [0])
            (a2,) = steered.activations([(seq, _SIMPLE[1])], [0])
            assert not np.allclose(a1, a2, atol=1e-6)
            assert base.identity() != steered.identity()
            assert Capability.STEER in steered.capabilities
        finally:
            steered.close()
    finally:
        base.close()


def test_steering_hook_classifies_prompt_vs_proof_by_t_in_cached_regime(tiny_ckpt: Path) -> None:
    """Direct, deterministic unit test of the module docstring's "T==1 rule": in the cached
    (`generate_ids_fast`-driven) regime, `_make_steering_hook` must fire a `where='prompt'` term
    only when the block call's `T > 1` (this step's state prefill) and a `where='proof'` term only
    when `T == 1` (one decode step of this step's action) -- regardless of `owner.
    _steer_prompt_len` (unused on this path). Exercised directly on `_make_steering_hook` rather
    than through a full `sample()` rollout: a tiny random model's greedy decode can coincidentally
    produce identical `Attempt` fields whether or not steering changed its *unrecorded* internal
    token choices (observed: both truncate at step 0 either way), so a rollout-level check would
    be flaky/uninformative; this is the thing the rollout actually depends on being correct."""
    from code.ndbench.adapters.danstate import _make_steering_hook

    adapter = DanStateAdapter(ckpt=tiny_ckpt, device="cpu")
    try:
        d = adapter._cfg["d"]
        delta = torch.ones(d)
        term_prompt = SteeringTerm(layer=0, vector=tuple([1.0] * d), alpha=1.0, where="prompt")
        term_proof = SteeringTerm(layer=0, vector=tuple([1.0] * d), alpha=1.0, where="proof")
        block = adapter._model.blocks[0]
        cache = {"n": 2}  # any dict with an "n" key reads as "cached" to the hook

        hook_prompt = _make_steering_hook(adapter, term_prompt, delta)
        hook_proof = _make_steering_hook(adapter, term_proof, delta)

        prefill_out = torch.randn(2, 5, d)  # T = 5 > 1: this step's state prefill
        assert not torch.allclose(
            hook_prompt(block, (None, None, None, None, cache), prefill_out), prefill_out
        )
        assert torch.allclose(
            hook_proof(block, (None, None, None, None, cache), prefill_out), prefill_out
        )

        decode_out = torch.randn(2, 1, d)  # T = 1: one decode step of the action
        assert torch.allclose(
            hook_prompt(block, (None, None, None, None, cache), decode_out), decode_out
        )
        assert not torch.allclose(
            hook_proof(block, (None, None, None, None, cache), decode_out), decode_out
        )
    finally:
        adapter.close()


def test_steering_nonzero_vector_changes_sampled_attempts_via_prefill(tiny_ckpt: Path) -> None:
    """Integration-level corroboration (the T==1 rule's `where='prompt'` branch is the easy one to
    observe end to end: it fires on every step's state prefill, which always has `T > 1`, so a
    large enough term reliably perturbs the sampled continuation, unlike the `where='proof'` case
    above). Drives `generate_ids_fast` directly on the steered vs. unsteered model's own single
    decode step (same env/pids), rather than through `adapter.sample()`'s full rollout: a tiny
    random model's one-token greedy action is overwhelmingly likely to be *some* non-`<eos>` token
    either way, so steering changing *which* token was chosen (confirmed directly: 89 vs. 45 in the
    first version of this test) is invisible in `sample()`'s own `Attempt` fields, which collapse
    every non-`<eos>` one-token action to the same `"truncated"`/`"UNPARSEABLE env: action
    truncated"` outcome regardless of the actual token id."""
    base = DanStateAdapter(ckpt=tiny_ckpt, device="cpu", batch=4, max_action=1, max_steps=1)
    try:
        ds = load_danstate()
        d = base._cfg["d"]
        # A *uniform* vector is a poor choice here: `LayerNorm`'s mean-subtraction (every block's
        # `ln1`/`ln2`, and the model's final `ln_f`) cancels a perfectly uniform additive shift to
        # the residual stream almost entirely by the time it reaches the logits, even though it is
        # still visible in the raw (pre-`ln_f`) residual stream `activations()` captures (see the
        # zero/nonzero `activations`-based tests above). A structured, alternating-sign vector has
        # a component LayerNorm does not remove.
        rng = random.Random(0)
        vec = tuple(20.0 if rng.random() < 0.5 else -20.0 for _ in range(d))
        term = SteeringTerm(layer=0, vector=vec, alpha=5.0, where="prompt")
        steered = base.with_steering((term,))
        try:
            env = ds.state_env.Env(_SIMPLE[0], canon=True, base=0, assign=True)
            pids = [ds.state_sample.prompt_ids(base._tok, env)]
            unsteered_out = ds.sample.generate_ids_fast(
                base._model,
                base._tok,
                pids,
                greedy=True,
                max_new=1,
                seed=0,
                early="eos",
                compact=True,
            )
            steered_out = ds.sample.generate_ids_fast(
                steered._model,
                steered._tok,
                pids,
                greedy=True,
                max_new=1,
                seed=0,
                early="eos",
                compact=True,
            )
            assert unsteered_out != steered_out
        finally:
            steered.close()
    finally:
        base.close()


# --------------------------------------------------------------------------------------- misc


def test_identity_fields(adapter: DanStateAdapter) -> None:
    d = adapter.identity()
    for key in (
        "adapter",
        "ckpt",
        "ckpt_sha256",
        "arch",
        "tokenizer_mode",
        "tokenizer_vocab_sha256",
        "vendor_commit",
        "rollout",
        "batch",
        "max_action",
        "max_steps",
        "name_base_max",
    ):
        assert key in d
    assert d["adapter"] == "danstate"
    assert d["vendor_commit"] == "0c492902"
    assert d["rollout"] == "danstate_v1"


def test_kwarg_coercion_of_string_model_args(tiny_ckpt: Path) -> None:
    a = DanStateAdapter(
        ckpt=tiny_ckpt,
        device="cpu",
        batch="8",
        max_action="12",
        max_steps="5",
        name_base_max="3",
        kv_budget_gb="2.5",
    )
    try:
        assert a._batch == 8 and isinstance(a._batch, int)
        assert a._max_action == 12 and isinstance(a._max_action, int)
        assert a._max_steps == 5 and isinstance(a._max_steps, int)
        assert a._name_base_max == 3 and isinstance(a._name_base_max, int)
        assert a._kv_budget_gb == 2.5 and isinstance(a._kv_budget_gb, float)
    finally:
        a.close()


def test_kwarg_coercion_rejects_garbage(tiny_ckpt: Path) -> None:
    with pytest.raises(ValueError):
        DanStateAdapter(ckpt=tiny_ckpt, device="cpu", batch="not-a-number")


def test_cross_request_seed_is_deterministic_and_order_sensitive() -> None:
    seq = Sequent.from_prompt(_SIMPLE[0])
    r1 = SampleRequest(pid="a", sequent=seq, n=1, temperature=1.0, max_new_tokens=10, seed=1)
    r2 = SampleRequest(pid="b", sequent=seq, n=1, temperature=1.0, max_new_tokens=10, seed=2)
    assert _cross_request_seed([r1, r2]) == _cross_request_seed([r1, r2])
    assert _cross_request_seed([r1, r2]) != _cross_request_seed([r2, r1])


def test_kv_safe_chunks_splits_when_over_budget() -> None:
    lengths = [100] * 10
    chunks_unsplit = _kv_safe_chunks(
        lengths, max_action=50, kv_budget_gb=1000.0, bytes_per_token=4096
    )
    assert chunks_unsplit == [list(range(10))]

    # A tiny budget forces a split; the original (worklist) order within and across chunks must
    # be preserved (never re-sorted).
    chunks_split = _kv_safe_chunks(lengths, max_action=50, kv_budget_gb=1e-6, bytes_per_token=4096)
    assert len(chunks_split) > 1
    flattened = [i for c in chunks_split for i in c]
    assert flattened == list(range(10))


def test_token_count(adapter: DanStateAdapter) -> None:
    seq = Sequent.from_prompt(_SIMPLE[0])
    n0 = adapter.token_count(seq)
    n1 = adapter.token_count(seq, _SIMPLE[1])
    assert n1 > n0
    assert n0 == len(adapter._tok.encode_prompt(seq.prompt()))


def test_sample_ignores_empty_request_list(adapter: DanStateAdapter) -> None:
    assert adapter.sample([]) == []


# --------------------------------------------------------------------------- V4 (slow, real ckpt)


@pytest.mark.slow
def test_v4_greedy_rollout_logprob_agreement_on_real_checkpoint() -> None:
    if not Path(REAL_CKPT).is_file():
        pytest.skip(f"real checkpoint not available at {REAL_CKPT}")
    if not TEXTBOOK_DEV_SUITE.is_file():
        pytest.skip(f"textbook_dev suite not available at {TEXTBOOK_DEV_SUITE}")

    n_theorems = 8
    prompts: list[str] = []
    with open(TEXTBOOK_DEV_SUITE) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prem = row["sequent"]["premises"]
            concl = row["sequent"]["conclusion"]
            prompt = f"THM {' , '.join(prem)} SEQ {concl} PRF" if prem else f"THM SEQ {concl} PRF"
            prompts.append(prompt)
            if len(prompts) >= n_theorems:
                break

    adapter = DanStateAdapter(
        ckpt=REAL_CKPT, device="cpu", batch=n_theorems, max_action=512, max_steps=96
    )
    try:
        reqs = [
            SampleRequest(
                pid=f"t{i}",
                sequent=Sequent.from_prompt(p),
                n=1,
                temperature=0.0,
                max_new_tokens=999,
                seed=100 + i,
            )
            for i, p in enumerate(prompts)
        ]
        per_req_attempts = adapter.sample(reqs)
        attempts = [attempts[0] for attempts in per_req_attempts]

        n_checked = 0
        for req, attempt in zip(reqs, attempts, strict=True):
            native = _native(attempt)
            if native["end"] != "done" or attempt.body.startswith("UNPARSEABLE"):
                continue
            b = native["base"]
            (lp,) = adapter.logprob([(req.sequent, attempt.body)])
            if math.isnan(lp.nats):
                continue
            ds = load_danstate()
            steps, _toks, env = ds.state_env.decompose(
                req.sequent.prompt(), attempt.body, canon=True
            )
            valid_bases = [
                bb for bb in range(0, adapter._name_base_max + 1) if env.maxname + bb <= 64
            ]
            assert b in valid_bases
            idx = valid_bases.index(b)
            log_n = math.log(adapter._name_base_max + 1)
            manual_total = 0.0
            for state_toks, action_toks, _hist in steps:
                s = _shift_name_tokens(state_toks, b)
                a = _shift_name_tokens(action_toks, b)
                manual_total += _manual_step_logprob(adapter._model, adapter._tok, s, a)
            manual_per_encoding_b = manual_total - log_n
            assert math.isclose(lp.per_encoding[idx], manual_per_encoding_b, abs_tol=1e-3)
            n_checked += 1
        assert n_checked > 0, "no finished, logprob-scorable greedy attempt among the sample"

        # Informational only (not an assertion): how many of these greedy proofs nd_v2 accepts.
        judge = NDJudge()
        n_accept = 0
        for req, attempt in zip(reqs, attempts, strict=True):
            (verdict,) = judge.judge(req.sequent, [attempt], None)
            if verdict.ok:
                n_accept += 1
        print(
            f"\n[V4] nd_v2 accepted {n_accept}/{len(reqs)} greedy proofs from "
            f"{Path(REAL_CKPT).name} on textbook_dev (informational only)"
        )
    finally:
        adapter.close()
