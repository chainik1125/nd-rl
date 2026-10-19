"""`core.types` dataclasses: `Sequent` prompt<->thm round trip, and `Problem`/other dataclasses'
JSON round trip."""

from __future__ import annotations

from code.ndbench.core.types import (
    Attempt,
    ItemResult,
    LengthLabel,
    LogProb,
    PrefixSpec,
    Problem,
    ReferenceProof,
    SampleRequest,
    Sequent,
    Verdict,
)

from ._fixtures import load_valid_proofs


def test_sequent_prompt_thm_roundtrip() -> None:
    for p in load_valid_proofs():
        seq = Sequent.from_prompt(p["prompt"])
        assert seq.prompt() == p["prompt"]
        assert seq.thm() == p["thm"]
        seq2 = Sequent.from_thm(p["thm"])
        assert seq2 == seq


def test_sequent_no_premises() -> None:
    seq = Sequent(premises=(), conclusion="( P v ( ~ P ) )")
    assert seq.prompt() == "THM SEQ ( P v ( ~ P ) ) PRF"
    assert seq.thm() == "|- ( P v ( ~ P ) )"
    assert Sequent.from_prompt(seq.prompt()) == seq
    assert Sequent.from_thm(seq.thm()) == seq


def test_sequent_key_and_pkey() -> None:
    seq = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    assert seq.key() == "P , Q |- ( P & Q )"
    seq_swapped = Sequent(premises=("Q", "P"), conclusion="( Q & P )")
    # pkey is premise-order and (via canon_key) atom-naming invariant: both sequents state
    # "conjoin the two premises", just spelled with swapped atom names and premise order.
    assert seq.pkey() == seq_swapped.pkey()


def test_problem_to_json_from_json_roundtrip() -> None:
    seq = Sequent(premises=("P",), conclusion="( P & P )")
    p = Problem(
        pid="p1",
        suite="demo",
        version="v1",
        sequent=seq,
        length=LengthLabel(
            L=2, L_ub=None, L_lb=None, method="minlen_exact", bound=10, timeout=False
        ),
        ref=ReferenceProof(
            body="N1 P : PR ; N2 ( P & P ) : ANDI N1 N1 ; QED", lines=2, pruned=2, source="minlen"
        ),
        tags={"family": "demo", "n": 1},
        parts=("part1",),
        prefix=PrefixSpec(j=1, body="N1 P : PR ;"),
        seen_train=True,
        seen_eval=("dev",),
    )
    d = p.to_json()
    p2 = Problem.from_json(d)
    assert p2 == p


def test_problem_to_json_from_json_roundtrip_minimal() -> None:
    """Optional fields (ref/prefix/tags/parts/seen_*) at their defaults still round-trip."""
    seq = Sequent(premises=(), conclusion="P")
    p = Problem(
        pid="p2",
        suite="demo",
        version="v1",
        sequent=seq,
        length=LengthLabel(L=None, L_ub=5, L_lb=1, method="ub_lb", bound=5),
        ref=None,
        tags={},
    )
    assert Problem.from_json(p.to_json()) == p


def test_sample_request_default_options_independent() -> None:
    """`options` defaults to `{}` via `field(default_factory=dict)`, not a shared mutable
    default, since the plan's pseudocode `= {}` is not valid as a literal dataclass default."""
    req1 = SampleRequest(
        pid="p", sequent=Sequent((), "P"), n=1, temperature=0.8, max_new_tokens=10, seed=0
    )
    req2 = SampleRequest(
        pid="p", sequent=Sequent((), "P"), n=1, temperature=0.8, max_new_tokens=10, seed=0
    )
    assert req1.options == {}
    assert req1.options is not req2.options


def test_attempt_and_verdict_are_plain_dataclasses() -> None:
    a = Attempt(pid="p", i=0, body="N1 P : PR ; QED", native=None, n_tokens=5, finished=True)
    v = Verdict(ok=True, reason="ok", kind="ok", written=1, pruned=1, first_bad_line=None, start=1)
    assert a.pid == "p"
    assert v.ok is True


def test_item_result_is_mutable() -> None:
    r = ItemResult(
        run_id="r",
        pid="p",
        chunk=0,
        n=0,
        c=0,
        first_success=None,
        ok_written=[],
        ok_pruned=[],
        ok_tokens=[],
        ok_starts=[],
        fail_kinds={},
        fail_lines=[],
        long_enough=0,
        tokens_mean=0.0,
        hit_max=0,
    )
    r.n += 1  # ItemResult is not frozen
    assert r.n == 1


def test_logprob() -> None:
    lp = LogProb(nats=1.5, n_tokens=10, per_encoding=(1.5, 1.7))
    assert lp.per_encoding == (1.5, 1.7)
