"""Hand-built proofs in the models' Lean surface, run through the Lean -> ND converter and Dan's
classifier, plus a K12 round trip. Every hand-built proof is also checked by Lean.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m pytest -q \\
        code/experiments/current/proof_strategies_grpo_20261101/test_ps.py
"""

from __future__ import annotations

import json
import os
import random
import tempfile

import pytest

from code.experiments.current.proof_strategies_grpo_20261101 import common as C
from code.experiments.current.proof_strategies_grpo_20261101.classify import labels

# (prompt, Lean text, labels that must hold, labels that must not)
CASES: dict[str, tuple[str, str, set[str], set[str]]] = {
    # Peirce: assume ¬P, build P → Q from it, apply the premise, clash: a free working reductio
    "reductio_free": (
        "THM ( ( P > Q ) > P ) SEQ P PRF",
        "have n1 : ( ( P → Q ) → P ) := h1 ; "
        "have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : ( ¬ P ) ) => by "
        "have n4 : ( P → Q ) := ( fun ( n5 : P ) => by "
        "have n6 : False := n3 n5 ; have n7 : Q := n6.elim ; exact n7 ) ; "
        "have n8 : P := n1 n4 ; have n9 : False := n3 n8 ; exact ( n9 : False ) ) ; "
        "have n10 : P := Classical.byContradiction ( fun hh => n2 hh ) ; exact n10",
        {"u:contradiction", "contradiction.working", "contradiction.free", "pri:contradiction"},
        {"contradiction.template", "u:dne"},
    ),
    # G is already a premise: a pointless (template) reductio
    "reductio_pointless": (
        "THM P SEQ P PRF",
        "have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : ( ¬ P ) ) => by "
        "have n4 : False := n3 n1 ; exact ( n4 : False ) ) ; "
        "have n5 : P := Classical.byContradiction ( fun hh => n2 hh ) ; exact n5",
        {"u:contradiction", "contradiction.template", "contradiction.pointless"},
        {"contradiction.working"},
    ),
    "modus_tollens": (
        "THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF",
        "have n1 : ( P → Q ) := h1 ; have n2 : ( ¬ Q ) := h2 ; "
        "have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : Q := n1 n4 ; "
        "have n6 : False := n2 n5 ; exact ( n6 : False ) ) ; exact n3",
        {
            "u:mod_tollens",
            "mod_tollens.strict",
            "neg_intro.working",
            "neg_intro.mt",
            "pri:mod_tollens",
        },
        {"u:contradiction"},
    ),
    "disjunctive_syllogism": (
        "THM ( P v Q ) , ( ~ P ) SEQ Q PRF",
        "have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ¬ P ) := h2 ; "
        "have n3 : Q := Or.elim n1 ( fun ( n4 : P ) => by have n5 : False := n2 n4 ; "
        "have n6 : Q := n5.elim ; exact n6 ) ( fun ( n7 : Q ) => by exact n7 ) ; exact n3",
        {"u:disj_syll", "u:cases", "cases.genuine", "u:ex_falso", "pri:disj_syll"},
        set(),
    ),
    "cases_redundant": (
        "THM ( P v Q ) , R SEQ R PRF",
        "have n1 : ( P ∨ Q ) := h1 ; have n2 : R := h2 ; "
        "have n3 : R := Or.elim n1 ( fun ( n4 : P ) => by have n5 : R := n2 ; exact n5 ) "
        "( fun ( n6 : Q ) => by have n7 : R := n2 ; exact n7 ) ; exact n3",
        {"u:cases", "cases.redundant"},
        {"cases.genuine", "u:disj_syll"},
    ),
    "cases_genuine": (
        "THM ( P v Q ) SEQ ( Q v P ) PRF",
        "have n1 : ( P ∨ Q ) := h1 ; "
        "have n2 : ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : P ) => by "
        "have n4 : ( Q ∨ P ) := Or.inr n3 ; exact n4 ) ( fun ( n5 : Q ) => by "
        "have n6 : ( Q ∨ P ) := Or.inl n5 ; exact n6 ) ; exact n2",
        {"u:cases", "cases.genuine", "u:or_intro", "pri:cases"},
        {"cases.redundant", "u:disj_syll"},
    ),
    "cond_vacuous": (
        "THM Q SEQ ( P > Q ) PRF",
        "have n1 : Q := h1 ; "
        "have n2 : ( P → Q ) := ( fun ( n3 : P ) => by have n4 : Q := n1 ; exact n4 ) ; exact n2",
        {"u:cond_proof", "cond_proof.vacuous", "pri:cond_proof"},
        {"cond_proof.working"},
    ),
    "lemma_reuse": (
        "THM ( P > Q ) , ( Q > R ) , P SEQ ( Q & R ) PRF",
        "have n1 : ( P → Q ) := h1 ; have n2 : ( Q → R ) := h2 ; have n3 : P := h3 ; "
        "have n4 : Q := n1 n3 ; have n5 : R := n2 n4 ; "
        "have n6 : ( Q ∧ R ) := ⟨ n4 , n5 ⟩ ; exact n6",
        {"u:lemma_reuse", "u:direct", "u:and_intro", "pri:and_intro"},
        set(),
    ),
    "excluded_middle": (
        "THM SEQ ( P v ( ~ P ) ) PRF",
        "have n1 : ( ¬ ( ¬ ( P ∨ ( ¬ P ) ) ) ) := ( fun ( n2 : ( ¬ ( P ∨ ( ¬ P ) ) ) ) => by "
        "have n3 : ( ¬ P ) := ( fun ( n4 : P ) => by have n5 : ( P ∨ ( ¬ P ) ) := Or.inl n4 ; "
        "have n6 : False := n2 n5 ; exact ( n6 : False ) ) ; "
        "have n7 : ( P ∨ ( ¬ P ) ) := Or.inr n3 ; have n8 : False := n2 n7 ; "
        "exact ( n8 : False ) ) ; "
        "have n9 : ( P ∨ ( ¬ P ) ) := Classical.byContradiction ( fun hh => n1 hh ) ; exact n9",
        # a root DN on X v ~X is #12 in "uses"; only a case split on it is #12 as primary
        # (TAXONOMY §2)
        {"u:lem", "u:contradiction", "contradiction.working", "pri:contradiction"},
        set(),
    ),
    "dne_premise": (
        "THM ( ~ ( ~ P ) ) SEQ P PRF",
        "have n1 : ( ¬ ( ¬ P ) ) := h1 ; "
        "have n2 : P := Classical.byContradiction ( fun hh => n1 hh ) ; exact n2",
        {"u:dne", "dne.on_premise", "pri:dne"},
        {"u:contradiction"},
    ),
    "ex_falso_inconsistent": (
        "THM P , ( ~ P ) SEQ Q PRF",
        "have n1 : P := h1 ; have n2 : ( ¬ P ) := h2 ; have n3 : False := n2 n1 ; "
        "have n4 : Q := n3.elim ; exact n4",
        {"u:ex_falso", "ex_falso.inconsistent_premises", "pri:ex_falso"},
        set(),
    ),
    # a dead forward line, then the goal's box: the dead line is pruned away
    "dead_line": (
        "THM ( P & Q ) SEQ ( R > Q ) PRF",
        "have n1 : ( P ∧ Q ) := h1 ; have n2 : P := n1.1 ; "
        "have n3 : ( R → Q ) := ( fun ( n4 : R ) => by have n5 : Q := n1.2 ; exact n5 ) ; exact n3",
        {"u:cond_proof", "cond_proof.vacuous"},
        {"gf"},
    ),
    "goal_first": (
        "THM ( P & Q ) SEQ ( R > Q ) PRF",
        "have n1 : ( P ∧ Q ) := h1 ; "
        "have n2 : ( R → Q ) := ( fun ( n3 : R ) => by have n4 : Q := n1.2 ; exact n4 ) ; exact n2",
        {"gf"},
        set(),
    ),
}


def flat(lab: dict) -> set[str]:
    out = {f"u:{s}" for s in lab["u"]} | set(lab["sub"]) | {f"pri:{lab['pri']}"}
    if lab["gf"]:
        out.add("gf")
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_labels(case: str) -> None:
    prompt, text, must, mustnot = CASES[case]
    nd = C.lean_to_nd(text)
    lab = labels(nd, prompt)
    assert lab is not None
    got = flat(lab)
    assert must <= got, (case, sorted(must - got), nd)
    assert not (mustnot & got), (case, sorted(mustnot & got), nd)


def test_dead_line_counts() -> None:
    prompt, text, _, _ = CASES["dead_line"]
    lab = labels(C.lean_to_nd(text), prompt)
    assert lab is not None and lab["L"] - lab["Lp"] == 1 and lab["fp"] == 1 and lab["fpl"] == 0


def test_merge_renamed() -> None:
    """Two texts that differ only in names merge into one ND proof with the summed count."""
    t = CASES["modus_tollens"][1]
    t2 = t.replace("n3", "n13").replace("n4", "n14").replace("n5", "n15").replace("n6", "n16")
    nd, bad = C.merged_accepted({"accepted": {t: 3, t2: 2}})
    assert bad == 0 and list(nd.values()) == [5]


def test_k12_round_trip() -> None:
    """ND -> the models' Lean text -> ND keeps every pruned label on 2,000 K12 proofs."""
    S = C.dan_strategies()
    rng = random.Random(20261101)
    with open(C.K12) as f:
        rows = [json.loads(x) for x in f]
    for r in rng.sample(rows, 2000):
        a = S.analyse(r["proof"], r["prompt"], written=False)
        b = S.analyse(C.lean_to_nd(C.nd_to_lean_text(r["proof"])), r["prompt"], written=False)
        for key in ("u", "sub", "pri", "pric", "Lp"):
            assert a[key] == b[key], (key, r["prompt"], r["proof"])


@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="no Lean")
def test_lean_accepts_cases() -> None:
    from code.thinking.lean import lean_check

    os.environ.setdefault("LEAN_GATE_LOG", os.path.join(tempfile.mkdtemp(), "lean_gate.jsonl"))
    items = [(C.lean_statement(p), t) for p, t, _, _ in CASES.values()]
    ok = lean_check(items)
    assert all(ok), [k for k, v in zip(CASES, ok, strict=True) if not v]
    # a negative control: a broken citation must be rejected
    p, t, _, _ = CASES["modus_tollens"]
    assert lean_check([(C.lean_statement(p), t.replace("n2 n5", "n2 n4"))]) == [False]


def test_dan_cases_round_trip() -> None:
    """Dan's 39 hand-built ND test proofs, rendered in the models' Lean surface and converted back,
    still get the labels his tests require."""
    C.dan_strategies()
    import test_strategies as T  # type: ignore[import-not-found]  # the fork's

    S = C.dan_strategies()
    for name, prompt, proof, must, mustnot, subs, pri, pric in T.CASES:
        a = S.analyse(C.lean_to_nd(C.nd_to_lean_text(proof)), prompt)
        u, s = set(a["u"]), set(a["sub"])
        assert must <= u and not (u & mustnot) and subs <= s, (name, a)
        assert a["pri"] == pri and (not pric or a["pric"] == pric), (name, a)
