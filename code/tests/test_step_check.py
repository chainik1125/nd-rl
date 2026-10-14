"""Tests for ``code.shared_code.step_check``.

The golden fixture has the take-home fork's own ``step_check.check`` verdicts on 10,724 (state,
step) pairs: the steps of 275 K12 proofs replayed with environment-assigned names, and at each
``have`` the true action and variants of its formula and term that the environment accepts
(``fixtures/proof_state/make_golden.py``). The Lean test re-checks every pair with Lean 4
(``--runslow``; skipped when Lean is not installed).
"""

import gzip
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from code.shared_code.lean_check import check_theorems, find_lean
from code.shared_code.state_env import Env
from code.shared_code.step_check import check_step, lean_source

FIXTURES = Path(__file__).parent / "fixtures" / "proof_state"


def act(symbols: str) -> list[str]:
    """An action written as space-separated symbols."""
    return symbols.split()


def load(name: str) -> list[dict[str, Any]]:
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


PROMPTS = {r["i"]: r["prompt"] for r in load("k12_sample.jsonl.gz")}
ROWS = load("step_check.jsonl.gz")


def pairs() -> Iterator[tuple[Env, list[str], str | None]]:
    """(the environment before the step, the step, the fork's verdict), replaying each proof."""
    for row in ROWS:
        env = Env(PROMPTS[row["i"]], canon=True, base=row["base"], assign=True)
        checks: dict[int, list[tuple[str, str | None]]] = {}
        for k, action, verdict in row["checks"]:
            checks.setdefault(k, []).append((action, verdict))
        for k, action in enumerate(row["actions"]):
            for candidate, verdict in checks.get(k, []):
                yield env, candidate.split(), verdict
            assert env.apply(action.split()) == (True, "")


def test_fixture_covers_every_reason() -> None:
    verdicts = Counter(v for row in ROWS for _, _, v in row["checks"])
    assert sum(verdicts.values()) >= 10000
    assert set(verdicts) == {
        None,
        "premise-type",
        "reiterate",
        "projection",
        "and-elim",
        "elim",
        "function-expected",
        "app-arg",
        "app-result",
        "anon-ctor",
        "and-intro",
        "or-intro-target",
        "or-intro",
        "by-contradiction",
    }  # "unknown-premise" and "unbound" cannot occur once the environment has accepted a step


def test_verdicts_match_the_fork() -> None:
    bad = [(" ".join(a), v, check_step(env, a)) for env, a, v in pairs() if check_step(env, a) != v]
    assert not bad, bad[:5]


def _env() -> Env:
    env = Env(
        "THM ( P & Q ) , ( P > R ) , ( ~ S ) SEQ ( R & Q ) PRF", canon=True, base=3, assign=True
    )
    for action in ("have n4 : ( P ∧ Q ) := h1 ;", "have n5 : ( P → R ) := h2 ;"):
        assert check_step(env, action.split()) is None
        assert env.apply(action.split()) == (True, "")
    assert check_step(env, act("have n6 : ( S → False ) := h3 ;")) is None  # ¬ S ≡ S → False
    assert env.apply(act("have n6 : ( ¬ S ) := h3 ;")) == (True, "")
    return env


def test_accepts_what_lean_accepts() -> None:
    env = _env()
    for action in (
        "have n7 : P := n4 .1 ;",
        "have n7 : ( P ∧ Q ) := n4 ;",
        "have n7 : ( S → False ) := n6 ;",  # ¬ S ≡ S → False
        "have n7 : ( S → Q ) := n6 .elim ;",  # Not.elim : ¬ a → a → b
        "have n7 : ( ( P → ( Q → R ) ) → R ) := n4 .elim ;",  # And.elim
        "have n7 : ( ( P → R ) → ( P → R ) ) := ( fun ( n8 : ( P → R ) ) => by",  # a box
        "exact n6",  # not a have: the environment's business
    ):
        assert check_step(env, action.split()) is None, action


def test_rejects_certain_errors() -> None:
    env = _env()
    cases = {
        "have n7 : Q := n4 .1 ;": "and-elim",
        "have n7 : Q := n5 .1 ;": "projection",
        "have n7 : R := n5 n6 ;": "app-arg",
        "have n7 : R := n4 n5 ;": "function-expected",
        "have n7 : Q := n5 .elim ;": "elim",  # no Function.elim
        "have n7 : ( Q ∧ P ) := ⟨ n4 , n4 ⟩ ;": "and-intro",
        "have n7 : R := ⟨ n4 , n4 ⟩ ;": "anon-ctor",
        "have n7 : ( P ∨ R ) := Or.inl n4 ;": "or-intro",
        "have n7 : P := Or.inl n4 ;": "or-intro-target",
        "have n7 : Q := n4 ;": "reiterate",
        "have n7 : S := Classical.byContradiction ( fun hh => n6 hh ) ;": "by-contradiction",
        "have n7 : Q := n9 ;": "unbound",
    }
    for action, reason in cases.items():
        assert check_step(env, action.split()) == reason, action
    env = Env("THM ( P & Q ) SEQ P PRF", canon=True, assign=True)
    assert check_step(env, act("have n1 : ( Q ∧ P ) := h1 ;")) == "premise-type"
    assert check_step(env, act("have n1 : ( Q ∧ P ) := h2 ;")) == "unknown-premise"


def test_deep_nesting_passes() -> None:
    """A formula nested hundreds deep is not modelled (the fork's check passed it too)."""
    env = Env("THM P SEQ P PRF")
    deep = " ".join(["(", "¬"] * 700 + ["P"] + [")"] * 700)
    assert check_step(env, act(f"have n1 : {deep} := h1 ;")) is None


def test_lean_source() -> None:
    env = _env()
    assert lean_source(env, act("have n7 : P := n4 .1 ;")) == (
        "theorem t ( P Q R S : Prop ) ( h1 : ( P ∧ Q ) ) ( h2 : ( P → R ) ) ( h3 : ( ¬ S ) ) "
        "( n4 : ( P ∧ Q ) ) ( n5 : ( P → R ) ) ( n6 : ( ¬ S ) ) : P := by have z : P := n4.1 ; "
        "exact z"
    )
    with pytest.raises(ValueError):
        lean_source(env, act("exact n6"))


@pytest.mark.slow
def test_lean_agrees_with_every_verdict() -> None:
    """Each fixture step as a standalone theorem: Lean rejects exactly the steps the checker
    rejects (no false rejects, which would make guided sampling unsound, and no misses)."""
    if find_lean() is None:
        pytest.skip("Lean 4 is not installed")
    verdicts: dict[str, str | None] = {}
    for env, action, verdict in pairs():
        t = action[action.index(":=") + 1] if action[0] == "have" else None
        if t not in (None, "(", "Or.elim"):
            verdicts.setdefault(lean_source(env, action), verdict)
    sources = list(verdicts)
    accepted = check_theorems(sources)
    # (checker passes, Lean accepts): no (False, True), a false reject; no (True, False), a miss
    table = Counter((verdicts[s] is None, ok) for s, ok in zip(sources, accepted, strict=True))
    assert table == {(True, True): 2560, (False, False): 7105}
