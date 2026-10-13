"""Tests for ``code.shared_code.state_env``.

The golden fixtures were made by the take-home fork's own ``state_env.py`` on 1,100 K12 proofs
(``fixtures/proof_state/make_golden.py``; the fork commit is in ``manifest.json``): the (state,
action) pairs and the text of ``decompose`` with plain and canonical names, replays with
environment-assigned names, and the environment's verdict on 9,524 random or mutated actions.
The Lean test needs Lean 4 (``--runslow``; skipped when Lean is not installed).
"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from code.shared_code.lean_check import check_theorems, find_lean
from code.shared_code.lean_seq import LeanParseError, LeanSeqTokenizer, text
from code.shared_code.state_env import Env, canonicalise, decompose, split_actions

FIXTURES = Path(__file__).parent / "fixtures" / "proof_state"


def act(symbols: str) -> list[str]:
    """An action written as space-separated symbols."""
    return symbols.split()


def load(name: str) -> list[dict[str, Any]]:
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


RECORDS = {r["i"]: r for r in load("k12_sample.jsonl.gz")}
ROWS = load("state_env.jsonl.gz")


def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def test_fixture_covers_every_length_and_shape() -> None:
    assert len(ROWS) == 1100
    assert {r["n_lines"] for r in RECORDS.values()} == set(range(2, 13))
    actions = " ".join(a for row in ROWS for _, a in row["plain"]["steps"])
    for shape in ("Or.elim", "( fun", "Classical.byContradiction", ".elim", "⟨", "Or.inl", ".2"):
        assert shape in actions
    reasons = {f[4] for row in ROWS for f in row["fuzz"]}
    assert len(reasons) >= 20


def test_decompose_matches_the_fork() -> None:
    bad = []
    for row in ROWS:
        r = RECORDS[row["i"]]
        for mode, canon in (("plain", False), ("canon", True)):
            steps, toks, env = decompose(r["prompt"], r["proof"], canon=canon)
            pairs = [[" ".join(s.state), " ".join(s.action)] for s in steps]
            history_ok = all(
                s.history == [t for p in steps[:k] for t in p.action] for k, s in enumerate(steps)
            )
            got = (pairs, " ".join(toks), env.nd(), history_ok)
            want = (row[mode]["steps"], row[mode]["text"], row["plain"]["nd"], True)
            if got != want:
                bad.append((row["i"], mode))
    assert not bad, bad[:5]


def test_assigned_names_match_the_fork() -> None:
    bad = []
    for row in ROWS:
        a = row["assign"]
        env = Env(RECORDS[row["i"]]["prompt"], canon=True, base=a["base"], assign=True)
        states, results = [], []
        for act in a["actions"]:
            states.append(" ".join(env.state_tokens()))
            ok, why = env.apply(act.split())
            results.append([ok, why])
            if not ok:
                break
        got = [results, " ".join(env.text), env.renamed, env.defined]
        got += [digest("\n".join(states)), digest(" ".join(env.history))]
        want = [a["results"], a["text"], a["renamed"], a["defined"]]
        want += [a["states_digest"], a["history_digest"]]
        if got != want:
            bad.append(row["i"])
    assert not bad, bad[:5]
    assert sum(row["assign"]["renamed"] for row in ROWS) > 1000  # the overrides were exercised


def test_random_and_mutated_actions_match_the_fork() -> None:
    bad = []
    for row in ROWS:
        r = RECORDS[row["i"]]
        base = row["assign"]["base"]
        replays = {
            "plain": [a.split() for _, a in row["plain"]["steps"]],
            "canon": [a.split() for _, a in row["canon"]["steps"]],
            "assign": [a.split() for a in row["assign"]["actions"]],
        }
        for config, k, act, ok, why, after in row["fuzz"]:
            if config.startswith("assign:"):  # a high base, replaying the plain actions
                env = Env(r["prompt"], canon=True, base=int(config[7:]), assign=True)
                prefix = replays["plain"][:k]
            else:
                b = base if config == "assign" else 0
                env = Env(r["prompt"], canon=config != "plain", base=b, assign=config == "assign")
                prefix = replays[config][:k]
            for a in prefix:
                if not env.apply(a)[0]:
                    break
            got = env.apply(act.split())
            if got != (ok, why) or (ok and " ".join(env.text) != after):
                bad.append((row["i"], act, (ok, why), got))
    assert not bad, bad[:5]


F = "( ( P ∨ ( S ∨ P ) ) ∨ P )"
G = "( P ∨ ( S ∨ P ) )"


def test_worked_example() -> None:
    """``( ( P ∨ ( S ∨ P ) ) ∨ P ) ⊢ ( P ∨ ( S ∨ P ) )`` step by step: an Or.elim whose first
    branch closes at once and whose second branch the environment opens."""
    env = Env("THM ( ( P v ( S v P ) ) v P ) SEQ ( P v ( S v P ) ) PRF")
    steps = [
        (f"h1 : {F} <nl>", f"have n1 : {F} := h1 ;"),
        (f"h1 : {F} <nl> n1 : {F} <nl>", f"have n2 : {G} := Or.elim n1 ( fun ( n3 : {G} ) => by"),
        (f"h1 : {F} <nl> n1 : {F} <nl> n3 : {G} <nl>", "exact n3"),
        (f"h1 : {F} <nl> n1 : {F} <nl> n4 : P <nl>", f"have n5 : {G} := Or.inl n4 ;"),
        (f"h1 : {F} <nl> n1 : {F} <nl> n4 : P <nl> n5 : {G} <nl>", "exact n5"),
        (f"h1 : {F} <nl> n1 : {F} <nl> n2 : {G} <nl>", "exact n2"),
    ]
    for hyps, action in steps:
        assert " ".join(env.state_tokens()) == f"<st> {hyps} ⊢ {G} <act>"
        assert env.apply(action.split()) == (True, "")
    assert env.done
    assert text(env.text) == (
        f"have n1 : {F} := h1 ; have n2 : {G} := Or.elim n1 ( fun ( n3 : {G} ) => by exact n3 ) "
        f"( fun ( n4 : P ) => by have n5 : {G} := Or.inl n4 ; exact n5 ) ; exact n2"
    )
    nd = env.nd()
    assert nd.endswith(": ORE N1 N2 N2 N3 N4 ; QED")
    replay, _, _ = decompose(env.prompt, nd)
    assert [" ".join(s.action) for s in replay] == [a for _, a in steps]


def test_rejections() -> None:
    prompt = "THM ( P & Q ) , ( P > R ) SEQ R PRF"
    start = ["have n1 : ( P ∧ Q ) := h1 ;", "have n2 : ( P → R ) := h2 ;", "have n3 : P := n1 .1 ;"]
    cases = {
        "have n4 : R := n2 n9 ;": "unbound",
        "exact n1": "exact not last",
        "exact n3": "exact goal mismatch",
        "have n4 : ( P → R ) := h2 ;": "PR position",
        "have n4 : ( Q → R ) := ( fun ( n5 : P ) => by": "binder mismatch",
        "have n4 : R := ( fun ( n5 : P ) => by": "box on non-arrow",
        "have n4 : R := Or.elim n1 ( fun ( n5 : P ) => by": "Or.elim on non-or",
        "have n4 : R := n2 n3 n1 ;": "term",
        "have n4 : R := n2 n3": "no semicolon",
        "have n4 : R n2 n3 ;": "assign",
        "have n4 : ( R ) := n2 n3 ;": "formula",
        "fun n4": "action head",
        "": "empty action",
    }
    for action, reason in cases.items():
        env = Env(prompt)
        for a in start:
            assert env.apply(a.split()) == (True, "")
        assert env.apply(action.split()) == (False, reason), action
        assert env.failed == reason
        assert env.nd() == f"LEANPARSE env: {reason}"


def test_canonical_names() -> None:
    """After a box closes its names leave the scope; canonical names count from what remains."""
    prompt = "THM Q SEQ ( ( P > Q ) & ( R > Q ) ) PRF"
    proof = (
        "N1 Q : PR ; N2 | P : AS ; N3 | Q : R N1 ; N4 ( P > Q ) : IMPI N2 N3 ; N5 | R : AS ; "
        "N6 | Q : R N1 ; N7 ( R > Q ) : IMPI N5 N6 ; "
        "N8 ( ( P > Q ) & ( R > Q ) ) : ANDI N4 N7 ; QED"
    )
    _, plain, _ = decompose(prompt, proof)
    canon = canonicalise(prompt, proof)
    box1 = "have n1 : Q := h1 ; have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : Q := n1 ; "
    assert text(plain) == (
        box1 + "exact n4 ) ; have n5 : ( R → Q ) := ( fun ( n6 : R ) => by have n7 : Q := n1 ; "
        "exact n7 ) ; have n8 : ( ( P → Q ) ∧ ( R → Q ) ) := ⟨ n2 , n5 ⟩ ; exact n8"
    )
    assert text(canon) == (
        box1 + "exact n4 ) ; have n3 : ( R → Q ) := ( fun ( n4 : R ) => by have n5 : Q := n1 ; "
        "exact n5 ) ; have n4 : ( ( P → Q ) ∧ ( R → Q ) ) := ⟨ n2 , n3 ⟩ ; exact n4"
    )
    tok = LeanSeqTokenizer()
    assert tok.decode([tok.stoi[t] for t in canon] + [tok.eos_id]) == proof


def test_assigned_names_override_the_policys() -> None:
    env = Env("THM P SEQ ( Q > P ) PRF", canon=True, base=5, assign=True)
    for action in ("have n40 : P := h1 ;", "have n9 : ( Q → P ) := ( fun ( n9 : Q ) => by"):
        assert env.apply(action.split()) == (True, "")
    assert " ".join(env.text) == "have n6 : P := h1 ; have n7 : ( Q → P ) := ( fun ( n8 : Q ) => by"
    assert (env.defined, env.renamed) == (3, 3)
    env = Env("THM P SEQ P PRF", canon=True, base=64, assign=True)
    assert env.apply(act("have n1 : P := h1 ;")) == (False, "names exhausted")


def test_split_actions_rejects_text_outside_the_grammar() -> None:
    with pytest.raises(LeanParseError):
        split_actions(act("have n1 : P := h1 ; exact n1 n1"))
    actions, second = split_actions(
        act(
            "have n1 : ( P ∨ P ) := h1 ; have n2 : P := Or.elim n1 ( fun ( n3 : P ) => by "
            "exact n3 ) ( fun ( n4 : P ) => by exact n4 ) ; exact n2"
        )
    )
    assert [a[0] for a in actions] == ["have", "have", "exact", "exact", "exact"]
    assert second == ["n4"]


@pytest.mark.slow
def test_lean_accepts_renamed_proofs() -> None:
    """Canonical names and environment-assigned names are alpha-variants: Lean accepts them all."""
    if find_lean() is None:
        pytest.skip("Lean 4 is not installed")
    tok = LeanSeqTokenizer()
    sources = []
    for row in ROWS[::4]:
        statement = tok.statement(RECORDS[row["i"]]["prompt"])
        for t in (row["canon"]["text"], row["assign"]["text"]):
            sources.append(f"{statement} {text(t.split())}")
    assert all(check_theorems(sources))
