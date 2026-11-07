"""Unit tests for the semantic classifiers (no GPU): check 5 of the prereg.

python test_factor_lib.py           # or: python -m pytest test_factor_lib.py
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import json
import os

from factor_lib import (
    Trace,
    build_ref,
    classify_dn,
    classify_draw,
    classify_negi,
    classify_pick,
    classify_reach,
    flags,
    ref_from_json,
    replay,
    shift_action,
)
from patterns import parse  # type: ignore[import-not-found]
from state_env import decompose  # type: ignore[import-not-found]

# ( ~ P ) > Q , ~ Q |- P, the reductio shape: premises are written by the model in this env
PROMPT = "THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ P PRF"
ND = ("N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | ( ~ P ) : AS ; N4 | Q : IMPE N1 N3 ; "
      "N5 | F : NEGE N4 N2 ; N6 ( ~ ( ~ P ) ) : NEGI N3 N5 ; N7 P : DN N6 ; QED")  # fmt: skip
# the same with the reductio on a subgoal, inside an implication box
PROMPT_SUB = "THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ ( R > P ) PRF"
ND_SUB = ("N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | R : AS ; N4 | | ( ~ P ) : AS ; N5 | | Q : IMPE N1 N4 ; "
          "N6 | | F : NEGE N5 N2 ; N7 | ( ~ ( ~ P ) ) : NEGI N4 N6 ; N8 | P : DN N7 ; N9 ( R > P ) : IMPI N3 N8 ; QED")  # fmt: skip
# DN citing a premise is not σ even with T right
PROMPT_NN = "THM ( ~ ( ~ P ) ) SEQ P PRF"
ND_NN = ("N1 ( ~ ( ~ P ) ) : PR ; N2 | ( ~ P ) : AS ; N3 | F : NEGE N2 N1 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; "
         "N5 P : DN N4 ; QED")  # fmt: skip
# ORE, nested boxes (the patterns.py test proof with an inner reductio and a derived ORE)
PROMPT_ORE = "THM ( ( P v Q ) & ( ~ ( ~ R ) ) ) SEQ ( R v S ) PRF"
ND_ORE = ("N1 ( ( P v Q ) & ( ~ ( ~ R ) ) ) : PR ; N2 ( P v Q ) : ANDE1 N1 ; N3 ( ~ ( ~ R ) ) : ANDE2 N1 ; "
          "N4 | P : AS ; N5 | | ( ~ R ) : AS ; N6 | | F : NEGE N5 N3 ; N7 | ( ~ ( ~ R ) ) : NEGI N5 N6 ; "
          "N8 | R : DN N7 ; N9 | ( R v S ) : ORI1 N8 ; N10 | Q : AS ; N11 | R : DN N3 ; N12 | ( R v S ) : ORI1 N11 ; "
          "N13 ( R v S ) : ORE N2 N4 N9 N10 N12 ; QED")  # fmt: skip


def t(s: str) -> list[str]:
    return s.split()


def R(prompt=PROMPT, nd=ND):
    r = build_ref("x", prompt, None, "test", nd)
    assert not isinstance(r, str), r
    return r


NN_P = "( ¬ ( ¬ P ) )"
PICK = f"have n3 : {NN_P} := ( fun ( n4 : ( ¬ P ) ) => by"


def test_ref_groups() -> None:
    r = R()
    assert r.groups == ["setup", "setup", "pick", "reach", "reach", "negi", "dn", "finish"], (
        r.groups
    )
    assert r.t_is_goal and r.T == ("atom", "P") and r.n_pr == 2 and r.n_sigma_dn == 1
    s = R(PROMPT_SUB, ND_SUB)
    assert s.groups == [
        "setup",
        "setup",
        "setup",
        "pick",
        "reach",
        "reach",
        "negi",
        "dn",
        "finish",
        "finish",
    ]
    assert not s.t_is_goal and s.T == ("atom", "P")
    q = R(
        PROMPT_ORE, ND_ORE
    )  # the outermost, last σ DN: the only DN citing a NEGI (the other cites a line)
    assert q.n_dn == 2 and q.n_sigma_dn == 1 and q.T == ("atom", "R")
    # a non-σ proof cannot be a reference
    assert (
        build_ref(
            "x",
            "THM ( ~ ( ~ P ) ) SEQ P PRF",
            None,
            "t",
            "N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; QED",
        )
        == "not σ-flagged"
    )


def test_trace_equals_inverse() -> None:
    for prompt, nd in (
        (PROMPT, ND),
        (PROMPT_SUB, ND_SUB),
        (PROMPT_NN, ND_NN),
        (PROMPT_ORE, ND_ORE),
    ):
        steps, _, env = decompose(prompt, nd, canon=True)
        tr = replay(prompt, [list(a) for _, a, _ in steps])
        assert tr.env.done
        lines = parse(tr.env.nd())
        assert [(x["depth"], x["formula"], x["rule"], x["refs"]) for x in tr.lines] == [
            (x["depth"], x["formula"], x["rule"], x["refs"]) for x in lines
        ]
        for n in range(1, len(tr.lines) + 1):  # every prefix of a valid proof is valid
            assert tr.valid_through(n), (prompt, n)


def test_base_shift() -> None:
    r = R(PROMPT_ORE, ND_ORE)
    for base in (0, 7, 32):
        tr = replay(r.prompt, [shift_action(a, base) for a in r.actions], base)
        assert tr.env.done and tr.acts == [shift_action(a, base) for a in r.actions]
        assert tr.env.nd() == r.nd


def test_pick() -> None:
    r = R()
    assert classify_pick(r, 0, t(PICK), True)["pick"]
    # another spelling of the names: the env names what a step introduces
    assert classify_pick(r, 0, t(f"have n9 : {NN_P} := ( fun ( n8 : ( ¬ P ) ) => by"), True)["pick"]
    # truncated: <eos> never came
    assert not classify_pick(r, 0, t(PICK), False)["pick"]
    # assumes ¬T in an implication box: not the reductio
    c = classify_pick(r, 0, t("have n3 : ( ( ¬ P ) → P ) := ( fun ( n4 : ( ¬ P ) ) => by"), True)
    assert c["assume_any"] and not c["pick"]
    # the reductio of another formula
    assert not classify_pick(
        r, 0, t("have n3 : ( ¬ ( ¬ Q ) ) := ( fun ( n4 : ( ¬ Q ) ) => by"), True
    )["pick"]
    # wrong box depth: a box that first goes one level deeper is not the pick at this depth
    c = classify_pick(r, 0, t("have n3 : ( R → P ) := ( fun ( n4 : R ) => by"), True)
    assert not c["pick"] and not c["assume_any"]
    # binder mismatch: the env refuses it
    c = classify_pick(r, 0, t(f"have n3 : {NN_P} := ( fun ( n4 : P ) => by"), True)
    assert not c["env_ok"] and not c["pick"]


def test_reach() -> None:
    r = R()
    # the reference continuation (names as the env would give them after the pick)
    cont = [t("have n5 : Q := n1 n4 ;"), t("have n6 : False := n2 n5 ;"), t("exact n6"),
            t("have n4 : P := Classical.byContradiction ( fun hh => n3 hh ) ;"), t("exact n4")]  # fmt: skip
    c = classify_reach(r, 0, cont)
    assert c["bot"] and c["negi"] and c["dn"], c
    # a different valid route: an extra R copy on the way to ⊥ still counts
    c = classify_reach(
        r,
        0,
        [
            t("have n5 : Q := n1 n4 ;"),
            t("have n6 : Q := n5 ;"),
            t("have n7 : False := n2 n6 ;"),
            t("exact n7"),
        ],
    )
    assert c["bot"] and c["negi"] and not c["dn"], c
    # a false ⊥: declared False with a term that does not prove it (the env does not check types)
    c = classify_reach(r, 0, [t("have n5 : False := n1 n4 ;"), t("exact n5")])
    assert not c["bot"] and not c["negi"] and c["false_bot"] == 1, c
    # wrong citation order in NEGE
    c = classify_reach(r, 0, [t("have n5 : Q := n1 n4 ;"), t("have n6 : False := n5 n2 ;")])
    assert not c["bot"] and c["false_bot"] == 1, c
    # citing a name out of scope: the env refuses, so no ⊥
    assert not classify_reach(r, 0, [t("have n5 : False := n2 n60 ;")])["bot"]
    # wrong box depth: a valid ⊥ in a nested box is not a ⊥ the reductio's NEGI can close on
    c = classify_reach(r, 0, [t("have n5 : ( R → False ) := ( fun ( n6 : R ) => by"), t("have n7 : Q := n1 n4 ;"),
                              t("have n8 : False := n2 n7 ;"), t("exact n8")])  # fmt: skip
    assert not c["bot"] and c["nested_bot"] == 1, c
    # a line after a wrong line: ⊥ valid by its own rule but on top of a false line
    c = classify_reach(
        r, 0, [t("have n5 : Q := n4 n1 ;"), t("have n6 : False := n2 n5 ;"), t("exact n6")]
    )
    assert not c["bot"] and not c["negi"], c


def test_negi() -> None:
    r = R()
    assert classify_negi(r, 0, t("exact n6"), True)["negi"]
    # wrong citation: `exact` of a line that is not the box's last
    c = classify_negi(r, 0, t("exact n5"), True)
    assert not c["env_ok"] and not c["negi"]
    # writes another line instead of closing
    c = classify_negi(r, 0, t("have n7 : Q := n1 n4 ;"), True)
    assert c["env_ok"] and not c["negi"]
    # at a name base
    assert classify_negi(r, 9, t("exact n15"), True)["negi"]


def test_dn() -> None:
    r = R()
    dn = "have n4 : P := Classical.byContradiction ( fun hh => n3 hh ) ;"
    assert classify_dn(r, 0, t(dn), True)["dn"]
    assert classify_dn(
        r, 0, t("have n50 : P := Classical.byContradiction ( fun hh => n3 hh ) ;"), True
    )["dn"]
    # wrong citation: a line that is not the NEGI line
    c = classify_dn(r, 0, t("have n4 : P := Classical.byContradiction ( fun hh => n2 hh ) ;"), True)
    assert c["env_ok"] and not c["dn"] and c["dn_other"]
    # wrong formula
    assert not classify_dn(
        r, 0, t("have n4 : Q := Classical.byContradiction ( fun hh => n3 hh ) ;"), True
    )["dn"]
    # an R copy of the NEGI line is the first step of the chain, not the DN
    c = classify_dn(r, 0, t(f"have n4 : {NN_P} := n3 ;"), True)
    assert c["r_copy"] and not c["dn"]
    # DN citing another line of the right formula: valid, not this reductio's
    q = R(PROMPT_NN, ND_NN)
    c = classify_dn(q, 0, t("have n3 : P := Classical.byContradiction ( fun hh => n1 hh ) ;"), True)
    assert c["env_ok"] and c["valid"] and not c["dn"] and c["dn_other"]
    assert classify_dn(
        q, 0, t("have n3 : P := Classical.byContradiction ( fun hh => n2 hh ) ;"), True
    )["dn"]


def test_draw() -> None:
    r = R()
    acts = [list(a) for a in r.actions]
    c = classify_draw(PROMPT, 0, acts, r.T)
    assert c["stage"] == 4 and c["stage_ref_T"] == 4 and c["done"], c
    # stops at each stage
    assert classify_draw(PROMPT, 0, acts[:3])["stage"] == 1
    assert classify_draw(PROMPT, 0, acts[:5])["stage"] == 2
    assert classify_draw(PROMPT, 0, acts[:6])["stage"] == 3
    # a wrong premise line poisons everything after it
    bad = [t("have n1 : Q := h1 ;")] + acts[1:]
    assert classify_draw(PROMPT, 0, bad)["stage"] == 0
    # omitting the second premise is allowed (Lean does not need it restated); here the proof needs it
    # a non-σ proof with DN of a premise: no reductio at all
    c = classify_draw(PROMPT_NN, 0, [t("have n1 : ( ¬ ( ¬ P ) ) := h1 ;"),
                                     t("have n2 : P := Classical.byContradiction ( fun hh => n1 hh ) ;"), t("exact n2")])  # fmt: skip
    assert c["stage"] == 0 and c["n_reductio"] == 0 and c["done"], c
    # an implication box on ¬X is not a reductio
    c = classify_draw(
        PROMPT, 0, acts[:2] + [t("have n3 : ( ( ¬ P ) → P ) := ( fun ( n4 : ( ¬ P ) ) => by")]
    )
    assert c["n_reductio"] == 0, c
    # the inner reductio of a proof counts at its own depth
    q = R(PROMPT_ORE, ND_ORE)
    assert classify_draw(PROMPT_ORE, 0, q.actions, q.T)["stage"] == 4


def test_flags_agree_with_trace() -> None:
    r = R()
    assert flags(r.nd)["w"]["sigma_strict"]


def test_frozen_refs() -> None:
    """Every frozen reference replays at the name bases of the env (when the file is here)."""
    fn = os.environ.get("REFS", "/scratch/cowifornia/nd-rl/factor_read/refs/refs.jsonl")
    if not os.path.exists(fn):
        return
    rows = [ref_from_json(json.loads(x)) for x in open(fn)]
    assert len(rows) == 106
    for r in rows:
        for base in (0, 11, 32):
            tr = replay(r.prompt, [shift_action(a, base) for a in r.actions], base)
            assert tr.env.done and tr.fail is None and tr.env.nd() == r.nd, r.name
            assert tr.valid_through()
        assert Trace(r.prompt, 0).env.concl == r.goal
        assert classify_pick(r, 3, shift_action(r.actions[r.pick], 3), True)["pick"], r.name
        assert classify_negi(r, 3, shift_action(r.actions[r.negi], 3), True)["negi"], r.name
        assert classify_dn(r, 3, shift_action(r.actions[r.dn], 3), True)["dn"], r.name
        c = classify_reach(r, 3, [shift_action(a, 3) for a in r.actions[r.pick + 1 :]])
        assert c["bot"] and c["negi"] and c["dn"], (r.name, c)


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("ok", k, flush=True)
    print(f"{n} tests passed")
