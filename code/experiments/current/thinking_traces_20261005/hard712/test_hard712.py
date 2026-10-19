"""Tests for the hard712 generator, gates and exclusion (synthetic theorems only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m pytest -q -p no:cacheprovider \
        code/experiments/current/thinking_traces_20261005/hard712/test_hard712.py

Every fixture theorem below was written for these tests; `test_fixtures_not_textbook` checks
that none of them is in a textbook72 renaming class (atom renaming or premise permutation).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[5])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]  # the stdlib `code` module, not this repository's package

from code.experiments.current.thinking_traces_20261005.hard712 import (  # noqa: E402
    pfgates as GT,
)
from code.experiments.current.thinking_traces_20261005.hard712 import (  # noqa: E402
    pfgen as PG,
)
from code.experiments.current.thinking_traces_20261005.hard712 import (  # noqa: E402
    run as R,
)
from code.experiments.current.thinking_traces_20261005.hard712.ptree import (  # noqa: E402
    BOT,
    And,
    Imp,
    Node,
    Not,
    Or,
    linearize,
)

TB = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72"
INDEX = "/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work/excl_index.pkl"
P, Q, Rr, S = (("atom", x) for x in "PQRS")

# N(N()): ~(P & Q), ~Q > ~P |- ~P (shortest proof 10 lines; the inner negation box is needed)
NN_PROMPT = "THM ( ~ ( P & Q ) ) , ( ( ~ Q ) > ( ~ P ) ) SEQ ( ~ P ) PRF"
NN_PROOF = (
    "N1 ( ~ ( P & Q ) ) : PR ; N2 ( ( ~ Q ) > ( ~ P ) ) : PR ; N3 | P : AS ; N4 | | Q : AS ; "
    "N5 | | ( P & Q ) : ANDI N3 N4 ; N6 | | F : NEGE N5 N1 ; N7 | ( ~ Q ) : NEGI N4 N6 ; "
    "N8 | ( ~ P ) : IMPE N2 N7 ; N9 | F : NEGE N3 N8 ; N10 ( ~ P ) : NEGI N3 N9 ; QED"
)
# C(N()): ~P > Q, P > Q |- Q (11 lines; reductio around a negation box with two eliminations
# that use ~Q, whose result ~P is reused by IMPE; classical)
CN_PROMPT = "THM ( ( ~ P ) > Q ) , ( P > Q ) SEQ Q PRF"
CN_PROOF = (
    "N1 ( ( ~ P ) > Q ) : PR ; N2 ( P > Q ) : PR ; N3 | ( ~ Q ) : AS ; N4 | | P : AS ; "
    "N5 | | Q : IMPE N2 N4 ; N6 | | F : NEGE N5 N3 ; N7 | ( ~ P ) : NEGI N4 N6 ; "
    "N8 | Q : IMPE N1 N7 ; N9 | F : NEGE N8 N3 ; N10 ( ~ ( ~ Q ) ) : NEGI N3 N9 ; N11 Q : DN N10 ; "
    "QED"
)
# exportation: (P & Q) > R |- P > (Q > R) (7 lines; one elimination at depth 2)
EXP_PROMPT = "THM ( ( P & Q ) > R ) SEQ ( P > ( Q > R ) ) PRF"
EXP_PROOF = (
    "N1 ( ( P & Q ) > R ) : PR ; N2 | P : AS ; N3 | | Q : AS ; N4 | | ( P & Q ) : ANDI N2 N3 ; "
    "N5 | | R : IMPE N1 N4 ; N6 | ( Q > R ) : IMPI N3 N5 ; N7 ( P > ( Q > R ) ) : IMPI N2 N6 ; QED"
)
# a case split at the top level (not inside a box)
TOP_SPLIT_PROMPT = "THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ R PRF"
TOP_SPLIT_PROOF = (
    "N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 ( Q > R ) : PR ; N4 | P : AS ; "
    "N5 | R : IMPE N2 N4 ; N6 | Q : AS ; N7 | R : IMPE N3 N6 ; N8 R : ORE N1 N4 N5 N6 N7 ; QED"
)
# the same split inside an implication box, on the assumption
IN_BOX_SPLIT_PROMPT = "THM ( P > R ) , ( Q > R ) SEQ ( ( P v Q ) > R ) PRF"
IN_BOX_SPLIT_PROOF = (
    "N1 ( P > R ) : PR ; N2 ( Q > R ) : PR ; N3 | ( P v Q ) : AS ; N4 | | P : AS ; "
    "N5 | | R : IMPE N1 N4 ; N6 | | Q : AS ; N7 | | R : IMPE N2 N6 ; N8 | R : ORE N3 N4 N5 N6 N7 ; "
    "N9 ( ( P v Q ) > R ) : IMPI N3 N8 ; QED"
)
IRR3_PROMPT = "THM ( ~ P ) SEQ ( ( ~ R ) > ( ~ ( ( R & Q ) & P ) ) ) PRF"
FIXTURES = [
    IRR3_PROMPT,
    NN_PROMPT,
    CN_PROMPT,
    EXP_PROMPT,
    TOP_SPLIT_PROMPT,
    IN_BOX_SPLIT_PROMPT,
    "THM ( ~ ( ~ P ) ) , Q SEQ ( P & Q ) PRF",
    "THM ( P & P ) , Q SEQ ( P & Q ) PRF",
    "THM P , ( ~ P ) SEQ Q PRF",
    "THM P , Q SEQ ( P v R ) PRF",
    "THM ( P > R ) , ( Q > R ) SEQ ( ( P v Q ) > ( R v S ) ) PRF",
    "THM P , Q , R SEQ ( ( P & Q ) & R ) PRF",
    "THM SEQ ( ( ( ( P > Q ) > P ) > P ) > ( Q > ( P v Q ) ) ) PRF",
    "THM ( ( P & Q ) > ( Q v P ) ) , ( ( P v Q ) > ( Q & P ) ) SEQ ( ( ~ P ) > Q ) PRF",
    "THM ( P & ( Q & R ) ) SEQ ( R & P ) PRF",
]


def nn_tree() -> tuple[Node, list, dict]:
    prem = [Not(And(P, Q)), Imp(Not(Q), Not(P))]
    hypf = {1: P, 2: Q}
    h1, h2 = Node("HYP", P, ref=1), Node("HYP", Q, ref=2)
    inner = Node("NEGE", BOT, (Node("ANDI", And(P, Q), (h1, h2)), Node("PR", prem[0], ref=0)))
    nq = Node("NEGI", Not(Q), (inner,), (2,))
    np_ = Node("IMPE", Not(P), (Node("PR", prem[1], ref=1), nq))
    root = Node("NEGI", Not(P), (Node("NEGE", BOT, (h1, np_)),), (1,))
    return root, prem, hypf


# ---------------------------------------------------------------------- fixtures are not textbook
def test_fixtures_not_textbook() -> None:
    tb = set()
    for f in ("textbook_dev.jsonl", "textbook_train.jsonl"):
        for r in R.jl(f"{TB}/{f}"):
            k, pk = R.class_keys(r["prompt"])
            tb |= {k, pk}
    assert len(tb) > 50
    for p in FIXTURES:
        k, pk = R.class_keys(p)
        assert k not in tb and pk not in tb, p


# ---------------------------------------------------------------------- generator
def test_generator_deterministic() -> None:
    a = [PG.sample(s) for s in range(300)]
    b = [PG.sample(s) for s in range(300)]
    assert a == b
    got = [x for x in a if x is not None]
    assert len(got) > 5
    assert len({x["prompt"] for x in got}) > len(got) // 3


def test_generated_proofs_valid() -> None:
    from nd_verify import verify_text  # fork, on sys.path via pfgates

    n = 0
    for s in range(2000):
        c = PG.sample(s)
        if c is None:
            continue
        ok, why, nl = verify_text(c["prompt"] + " " + c["proof"])
        assert ok and nl == c["n_lines"], (s, why)
        n += 1
    assert n > 50


def test_linearize_hand_tree() -> None:
    root, prem, hypf = nn_tree()
    p = linearize(root, prem, hypf)
    assert p is not None
    assert p.prompt() == NN_PROMPT
    assert p.body() == NN_PROOF


def test_linearize_hoists_shared_lemma() -> None:
    """S is derived from a premise only: it is written once, outside both case boxes."""
    prem = [Or(P, Q), And(Rr, S)]
    hypf = {1: P, 2: Q}
    goal = Or(And(S, P), And(S, Q))

    def s_node() -> Node:
        return Node("ANDE2", S, (Node("PR", prem[1], ref=1),))

    b1 = Node("ORI1", goal, (Node("ANDI", And(S, P), (s_node(), Node("HYP", P, ref=1))),))
    b2 = Node("ORI2", goal, (Node("ANDI", And(S, Q), (s_node(), Node("HYP", Q, ref=2))),))
    root = Node("ORE", goal, (Node("PR", prem[0], ref=0), b1, b2), (1, 2))
    p = linearize(root, prem, hypf)
    assert p is not None and len(p.lines) == 10
    s_lines = [ln for ln in p.lines if ln.f == S]
    assert len(s_lines) == 1 and s_lines[0].depth == 0


def test_discharge_and_substitution_keep_validity() -> None:
    import random

    done = 0
    for seed in range(20):
        root, prem, hypf = nn_tree()
        b = PG.Builder(random.Random(seed), 2)
        b.nh = 10
        try:
            new_root, rest = PG.discharge(b, root, prem, hypf)
        except PG.Fail:  # discharging both premises would exceed 7 connectives
            continue
        p = linearize(new_root, rest, hypf)
        assert p is not None and p.verify()[0]
        assert len(rest) == 1 and p.concl[0] == "imp"
        done += 1
    assert done > 3
    root, prem, hypf = nn_tree()
    sub = {"Q": Or(Rr, S)}
    r2 = PG.subst_tree(root, sub)
    p2 = linearize(r2, [PG.subst_f(x, sub) for x in prem], {1: P, 2: Or(Rr, S)})
    assert p2 is not None and p2.verify()[0]
    assert "( R v S )" in p2.prompt()


# ---------------------------------------------------------------------- gates
@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("THM ( ~ ( ~ P ) ) , Q SEQ ( P & Q ) PRF", "dneg"),
        ("THM ( P & P ) , Q SEQ ( P & Q ) PRF", "degenerate"),
        ("THM P , ( ~ P ) SEQ Q PRF", "inconsistent"),
        ("THM P , Q SEQ ( P v R ) PRF", "redundant_prem"),
        ("THM ( P > R ) , ( Q > R ) SEQ ( ( P v Q ) > ( R v S ) ) PRF", "atoms"),
        ("THM P , Q , R SEQ ( ( P & Q ) & R ) PRF", "prem"),
        ("THM SEQ ( ( ( ( P > Q ) > P ) > P ) > ( Q > ( P v Q ) ) ) PRF", "fdepth"),
        (
            "THM ( ( P & Q ) > ( Q v P ) ) , ( ( P v Q ) > ( Q & P ) ) SEQ ( ( ~ P ) > Q ) PRF",
            "conn",
        ),
    ],
)
def test_shape_gate_rejects(prompt: str, reason: str) -> None:
    assert GT.shape_gate(prompt, "QED")[0] == reason


def test_shape_gate_accepts_and_checks_proof() -> None:
    assert GT.shape_gate(NN_PROMPT, NN_PROOF) == (
        None,
        {"n_prem": 2, "n_conn": 6, "n_atoms": 2, "fdepth": 2, "gen_lines": 10, "n_irrelevant": 0},
    )
    broken = NN_PROOF.replace("NEGE N3 N8", "NEGE N3 N7")
    assert GT.shape_gate(NN_PROMPT, broken)[0] == "gen_invalid"


def test_irrelevant_occurrences() -> None:
    # (~R & ~P) > ~(S & P): the R and S occurrences can be anything
    goal = Imp(And(Not(Rr), Not(P)), Not(And(S, P)))
    assert GT.irrelevant([], goal) == 2
    # a biconditional premise used in one direction only: both atoms of the other are irrelevant
    prem = [And(Imp(P, Q), Imp(Q, P)), P]
    assert GT.irrelevant(prem, Q) == 2
    assert GT.irrelevant(*GT.statement(CN_PROMPT)) == 0
    # ~R alone makes the goal true: every other occurrence (premise included) is irrelevant
    assert GT.irrelevant(*GT.statement(IRR3_PROMPT)) == 5 > GT.MAX_IRRELEVANT


def test_substitution_instance_match() -> None:
    pat = ([Imp(P, Q), Not(Q)], Not(P))
    inst = ([Not(Or(Rr, S)), Imp(And(S, P), Or(Rr, S))], Not(And(S, P)))  # premises permuted
    assert GT.instance_of(pat, inst)
    assert not GT.instance_of(inst, pat)
    assert not GT.instance_of(pat, ([Imp(P, Q), Not(P)], Not(Q)))
    # a repeated pattern atom must map to one formula
    assert not GT.instance_of(([], Imp(P, P)), ([], Imp(Q, Rr)))
    assert GT.instance_of(([], Imp(P, Q)), ([], Imp(Rr, Rr)))


def test_certify_nested_negation_needed() -> None:
    r = GT.certify(dict(prompt=CN_PROMPT, proof=CN_PROOF, gen_lines=11))
    assert r["status"] == "ok"
    assert r["min_lines"] == 11
    assert set(r["families"]) == {"nested_neg", "reductio", "depth2"}
    assert r["necessity"]["nested_neg"]["verdict"] == "needed"
    assert r["necessity"]["reductio"]["verdict"] == "needed"
    assert r["structure"]["s_nested_neg_reused"] == 1


def test_certify_t3_needs_two_inner_eliminations() -> None:
    """N(N()) whose inner contradiction is ANDI + NEGE (one elimination) is not T3 work."""
    r = GT.certify(dict(prompt=NN_PROMPT, proof=NN_PROOF, gen_lines=10))
    assert r["min_lines"] == 10
    assert r["structure"]["s_nested_neg_any"] == 1
    assert r["structure"]["s_nested_neg_work"] == 0
    assert r["status"] == "no_structure"


def test_certify_length_gates() -> None:
    # constructed proof claimed 2 lines longer than the shortest -> gap gate
    r = GT.certify(dict(prompt=CN_PROMPT, proof=CN_PROOF, gen_lines=13))
    assert r["status"] == "gap"
    # a 7-line theorem with no needed target structure (one elimination at depth 2)
    r = GT.certify(dict(prompt=EXP_PROMPT, proof=EXP_PROOF, gen_lines=7))
    assert r["status"] == "no_structure"
    # a case split at the top level is not a target structure
    r = GT.certify(dict(prompt=TOP_SPLIT_PROMPT, proof=TOP_SPLIT_PROOF, gen_lines=8))
    assert r["min_lines"] == 8 and r["status"] == "no_structure"
    # shortest proof under 7 lines
    short = "THM ( P & ( Q & R ) ) SEQ ( R & P ) PRF"
    proof = (
        "N1 ( P & ( Q & R ) ) : PR ; N2 ( Q & R ) : ANDE2 N1 ; N3 R : ANDE2 N2 ; "
        "N4 P : ANDE1 N1 ; N5 ( R & P ) : ANDI N3 N4 ; QED"
    )
    r = GT.certify(dict(prompt=short, proof=proof, gen_lines=5))
    assert r["min_lines"] == 5 and r["status"] == "short"


def test_structure_case_split_in_box() -> None:
    top = GT.structure(TOP_SPLIT_PROMPT, TOP_SPLIT_PROOF)
    inbox = GT.structure(IN_BOX_SPLIT_PROMPT, IN_BOX_SPLIT_PROOF)
    assert top["has"]["case_split"] == 0
    assert inbox["has"]["case_split"] == 1 and inbox["s_split_on_hyp"] == 1


# ---------------------------------------------------------------------- exclusion
def test_class_keys_match_thinking_data() -> None:
    from code.thinking.data import class_keys

    for p in FIXTURES:
        assert R.class_keys(p) == class_keys(p)


def test_exclusion_index_hits_renamings(tmp_path: Path) -> None:
    from code.experiments.current.thinking_traces_20261005.test_21_30.seal import (
        build_index,
        hits,
    )

    f = tmp_path / "seen.jsonl"
    f.write_text(json.dumps({"prompt": NN_PROMPT}) + "\n")
    idx = build_index([str(f)])
    renamed_permuted = "THM ( ( ~ S ) > ( ~ R ) ) , ( ~ ( R & S ) ) SEQ ( ~ R ) PRF"
    assert R.class_keys(renamed_permuted) != R.class_keys(NN_PROMPT)
    assert hits(idx, renamed_permuted) == {str(f)}
    assert hits(idx, EXP_PROMPT) == set()


@pytest.mark.skipif(not os.path.exists(INDEX), reason="production exclusion index not built")
def test_production_index_covers_sealed_sets() -> None:
    from code.experiments.current.thinking_traces_20261005.hard712.excl import REQUIRED

    with open(INDEX, "rb") as fh:
        idx = pickle.load(fh)
    files = {m["file"] for m in idx["manifest"]}
    assert set(REQUIRED) <= files
    for f in REQUIRED:
        rows = R.jl(f)[:200]
        for r in rows:
            k, pk = R.class_keys(r["prompt"])
            assert f in idx["keys"].get(k, set()) or f in idx["pkeys"].get(pk, set())


# ---------------------------------------------------------------------- crash-safe resume
def test_read_complete_drops_a_cut_line(tmp_path: Path) -> None:
    f = tmp_path / "gated.jsonl"
    f.write_text(json.dumps({"pkey": "a"}) + "\n" + json.dumps({"pkey": "b"}) + "\n" + '{"pk')
    rows, bad = R.read_complete(f)
    assert [r["pkey"] for r in rows] == ["a", "b"] and bad == 1
    assert R.read_complete(f) == (rows, 0)  # the file was rewritten without the cut tail
