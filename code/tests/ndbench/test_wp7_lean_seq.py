"""WP7 acceptance tests: `LeanSeqAdapter` (plan §3 WP7 row).

Covers: 10 golden ND<->Lean round trips through the vendored tokenizer directly (the plan's
acceptance criterion); `_partial_proof_tokens` (the prefix-rendering helper) against the vendored
`proof_tokens` at full length, and its "a still-open box contributes nothing yet" behaviour at a
truncated length; `sample()` end-to-end with a scripted policy (with and without a forced prefix);
the `UNPARSEABLE` fallbacks; `logprob`/`learner` with and without an injected scorer/factory;
`from_checkpoint`; and the declared capability flags.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from code.ndbench.adapters.base import Capability, ModelAdapter, NotSupported
from code.ndbench.adapters.lean_seq import (
    LeanSeqAdapter,
    PolicyFn,
    _partial_proof_tokens,
    _shift_ids,
    _tokens_to_ids,
    _valid_shifts,
)
from code.ndbench.core.proof import take_prefix
from code.ndbench.core.types import LogProb, PrefixSpec, SampleRequest, Sequent
from code.ndbench.judge.base import NotAvailable
from code.ndbench.vendor.lean_seq.lean_seq import LeanSeqTokenizer, proof_tokens

from ._fixtures import load_valid_proofs

_TOK = LeanSeqTokenizer()


class _FakeLearner:
    """The minimal `Learner` protocol, for `test_learner_with_factory_delegates`."""

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        return [LogProb(nats=0.0, n_tokens=0, per_encoding=()) for _ in pairs]

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        del pairs, lr
        return 0.0

    def adapter(self) -> ModelAdapter:
        raise NotImplementedError


def _always_correct_policy(body: str) -> PolicyFn:
    """A scripted policy that always continues (from any forced prefix) straight to the correct
    proof's remaining ids, deterministically, for any `n`."""
    full_ids = _TOK.encode_proof(body)

    def policy(
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[int]]:
        del prompt_ids, temperature, max_new_tokens, seed
        assert full_ids[: len(forced_ids)] == list(forced_ids)
        cont = full_ids[len(forced_ids) :]
        return [list(cont) for _ in range(n)]

    return policy


def test_golden_round_trips_ten_fixtures() -> None:
    """The plan's acceptance test: 10 golden ND<->Lean round trips (render, then `inverse`, gives
    back the same ND body) on verified proofs from `fixtures/valid_proofs.jsonl`."""
    proofs = load_valid_proofs()[:10]
    assert len(proofs) == 10
    for p in proofs:
        tok = LeanSeqTokenizer()
        ids = tok.encode_proof(p["proof"])
        assert tok.decode(ids) == p["proof"], p["name"]


def test_partial_tokens_match_full_render_at_full_length() -> None:
    """At `j = len(all lines)`, the partial renderer must agree with `proof_tokens` itself (minus
    the final `exact <last line>`, which only a *complete* proof gets)."""
    proofs = [p for p in load_valid_proofs() if p["rules"] and "AS" in p["rules"]][:5]
    assert proofs
    for p in proofs:
        body_no_qed = p["proof"].removesuffix(" QED")
        full = proof_tokens(p["proof"])
        partial = _partial_proof_tokens(body_no_qed)
        assert partial == full[:-2], p["name"]


def test_partial_tokens_defer_open_box_content() -> None:
    """A prefix ending anywhere strictly inside a still-open box renders identically to the prefix
    ending right when the box was opened: `lean_seq` only splices a box's content into the stream
    at its *discharging* line, never at its own `AS` line."""
    proof = next(p for p in load_valid_proofs() if p["name"] == "ab2_reserve_50145")
    body = proof["proof"]
    # N1, N2: premises; N3: AS (opens the box); N4-N6: inside the box; N7: IMPI (discharges it).
    assert proof["rules"] == sorted({"ANDI", "AS", "IMPI", "ORE", "ORI1", "ORI2", "R"})
    prefix_at_open = take_prefix(body, 3)
    prefix_mid_box = take_prefix(body, 6)
    assert _partial_proof_tokens(prefix_at_open) == _partial_proof_tokens(prefix_mid_box)
    # And it's exactly the two premises' `have` statements -- nothing from inside the box yet.
    ids = _tokens_to_ids(_partial_proof_tokens(prefix_at_open), _TOK)
    assert _TOK.itos[ids[0]] == "have"
    assert all(_TOK.itos[i] != "fun" for i in ids)  # no box term rendered yet


def test_partial_tokens_empty_prefix() -> None:
    assert _partial_proof_tokens("") == []


def test_capability_flags() -> None:
    adapter = LeanSeqAdapter(policy=lambda *a, **k: [])
    assert adapter.capabilities == (
        Capability.SAMPLE | Capability.PREFIX | Capability.LOGPROB | Capability.FINETUNE
    )


def test_sample_no_prefix_correct_policy_round_trips() -> None:
    proof = load_valid_proofs()[0]
    seq = Sequent.from_prompt(proof["prompt"])
    adapter = LeanSeqAdapter(policy=_always_correct_policy(proof["proof"]))
    req = SampleRequest(pid="p", sequent=seq, n=3, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    assert len(attempts) == 3
    for a in attempts:
        assert a.body == proof["proof"]
        assert a.finished is True
        assert a.native is not None and a.native.startswith("have")
        assert a.n_tokens > 0


def test_sample_with_prefix_round_trips() -> None:
    proof = next(p for p in load_valid_proofs() if p["name"] == "ab2_reserve_50145")
    seq = Sequent.from_prompt(proof["prompt"])
    prefix_body = take_prefix(proof["proof"], 3)
    adapter = LeanSeqAdapter(policy=_always_correct_policy(proof["proof"]))
    req = SampleRequest(
        pid="p",
        sequent=seq,
        n=2,
        temperature=0.8,
        max_new_tokens=200,
        seed=1,
        prefix=PrefixSpec(j=3, body=prefix_body),
    )
    (attempts,) = adapter.sample([req])
    for a in attempts:
        assert a.body == proof["proof"]


def test_sample_unparseable_no_eos() -> None:
    seq = Sequent(premises=(), conclusion="P")

    def policy(
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[int]]:
        del prompt_ids, forced_ids, temperature, seed
        return [[_TOK.stoi["have"]] * max_new_tokens for _ in range(n)]  # never reaches <eos>

    adapter = LeanSeqAdapter(policy=policy)
    req = SampleRequest(pid="p", sequent=seq, n=1, temperature=0.8, max_new_tokens=5, seed=0)
    (attempts,) = adapter.sample([req])
    assert attempts[0].body == "UNPARSEABLE no-eos"
    assert attempts[0].finished is False


def test_sample_unparseable_bad_grammar() -> None:
    seq = Sequent(premises=(), conclusion="P")

    def policy(
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[int]]:
        del prompt_ids, forced_ids, temperature, max_new_tokens, seed
        # Reaches <eos> (so it "finished"), but is not a grammatical lean_seq proof.
        return [[_TOK.stoi["have"], _TOK.eos_id] for _ in range(n)]

    adapter = LeanSeqAdapter(policy=policy)
    req = SampleRequest(pid="p", sequent=seq, n=1, temperature=0.8, max_new_tokens=5, seed=0)
    (attempts,) = adapter.sample([req])
    assert attempts[0].body.startswith("UNPARSEABLE")
    assert "no-eos" not in attempts[0].body
    assert attempts[0].finished is True


def test_logprob_without_scorer_raises_not_supported() -> None:
    adapter = LeanSeqAdapter(policy=lambda *a, **k: [])
    with pytest.raises(NotSupported):
        adapter.logprob([])


def test_logprob_with_scorer_marginalises_over_name_shifts() -> None:
    proof = load_valid_proofs()[0]
    seq = Sequent.from_prompt(proof["prompt"])
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def scorer(prompt_ids: Sequence[int], proof_ids: Sequence[int]) -> float:
        calls.append((tuple(prompt_ids), tuple(proof_ids)))
        return 1.0  # constant NLL per shift, to make the combination easy to check by hand

    adapter = LeanSeqAdapter(policy=lambda *a, **k: [], scorer=scorer)
    (lp,) = adapter.logprob([(seq, proof["proof"])])
    proof_ids = _TOK.encode_proof(proof["proof"])
    expected_shifts = list(_valid_shifts(proof_ids, _TOK))
    assert len(calls) == len(expected_shifts)
    assert lp.per_encoding == tuple(1.0 for _ in expected_shifts)
    assert lp.n_tokens == len(proof_ids)
    # Every per-shift NLL is 1.0 -> nats = -logsumexp([-1.0]*k) = 1.0 - ln(k).
    import math

    assert lp.nats == pytest.approx(1.0 - math.log(len(expected_shifts)))
    # And every call actually used a distinct (or the sole, if no names) shifted encoding.
    seen_ids = {ids for _, ids in calls}
    assert len(seen_ids) == len(expected_shifts)
    for s in expected_shifts:
        assert tuple(_shift_ids(proof_ids, _TOK, s)) in seen_ids


def test_learner_without_factory_raises_not_supported() -> None:
    adapter = LeanSeqAdapter(policy=lambda *a, **k: [])
    with pytest.raises(NotSupported):
        adapter.learner(seed=0)


def test_learner_with_factory_delegates() -> None:
    sentinel = _FakeLearner()
    adapter = LeanSeqAdapter(policy=lambda *a, **k: [], learner_factory=lambda seed: sentinel)
    assert adapter.learner(seed=3) is sentinel


def test_from_checkpoint_raises_not_available() -> None:
    with pytest.raises(NotAvailable):
        LeanSeqAdapter.from_checkpoint("/some/ckpt.pt")


def test_token_count() -> None:
    proof = load_valid_proofs()[0]
    seq = Sequent.from_prompt(proof["prompt"])
    adapter = LeanSeqAdapter(policy=lambda *a, **k: [])
    prompt_only = adapter.token_count(seq)
    with_body = adapter.token_count(seq, proof["proof"])
    assert with_body > prompt_only > 0


def test_identity_reports_policy_and_tokenizer_version() -> None:
    policy = lambda *a, **k: []  # noqa: E731
    adapter = LeanSeqAdapter(policy=policy, policy_identity={"kind": "test"})
    ident = adapter.identity()
    assert ident["policy"] == {"kind": "test"}
    assert ident["tokenizer_version"] == LeanSeqTokenizer.VERSION

    adapter_no_identity = LeanSeqAdapter(policy=policy)
    assert isinstance(adapter_no_identity.identity()["policy"], str)
