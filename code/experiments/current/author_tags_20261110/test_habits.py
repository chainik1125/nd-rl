"""Regression tests for habits.py (author-tags, 2026-11-10).

  /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest -q test_habits.py

Every rewrite is checked with Dan's nd_verify; one test also sends a handful to Lean (about 2 s)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import habits as H
from nd_verify import verify_text

# ( P > Q ) , ( Q > S ) , P |- ( R > S ), written goal-driven: the box opens first and Q is derived
# inside it
GD = (
    "THM ( P > Q ) , ( Q > S ) , P SEQ ( R > S ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 | R : AS ; N5 | Q : IMPE N1 N3 ; "
    "N6 | S : IMPE N2 N5 ; "
    "N7 ( R > S ) : IMPI N4 N6 ; QED",
)
PD_EXPECTED = (
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 | R : AS ; N6 "
    "| S : IMPE N2 N4 ; "
    "N7 ( R > S ) : IMPI N5 N6 ; QED"
)
# a case split whose first branch restates an outer line by R (required: last line of the box) and
# an optional R
ORE = (
    "THM ( P v Q ) , R , ( Q > R ) SEQ ( R & R ) PRF",
    "N1 ( P v Q ) : PR ; N2 R : PR ; N3 ( Q > R ) : PR ; N4 | P : AS ; N5 | R : R N2 ; N6 | Q "
    ": AS ; N7 | R : IMPE N3 N6 ; "
    "N8 R : ORE N1 N4 N5 N6 N7 ; N9 R : R N2 ; N10 ( R & R ) : ANDI N8 N9 ; QED",
)
# ex falso
BOTE = (
    "THM P , ( ~ P ) SEQ Q PRF",
    "N1 P : PR ; N2 ( ~ P ) : PR ; N3 F : NEGE N1 N2 ; N4 Q : BOTE N3 ; QED",
)
# two independent conjuncts, left first
CONJ = (
    "THM ( P > Q ) , ( P > R ) , P SEQ ( Q & R ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( P > R ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 R : IMPE N2 N3 "
    "; N6 ( Q & R ) : ANDI N4 N5 ; QED",
)
# ORI with both disjuncts citable
DISJ = ("THM P , Q SEQ ( P v Q ) PRF", "N1 P : PR ; N2 Q : PR ; N3 ( P v Q ) : ORI1 N1 ; QED")


def valid(prompt, body):
    ok, why, _ = verify_text(prompt + " " + body)
    assert ok, (why, body)
    return True


def nl(body):
    return sum(t == ";" for t in body.split())


def test_fixtures_valid():
    for p, b in (GD, ORE, BOTE, CONJ, DISJ):
        valid(p, b)


def test_roundtrip_hand_and_corpus():
    for _p, b in (GD, ORE, BOTE, CONJ, DISJ):
        assert H.render(H.build(b)) == b
    recs = H.read_corpus(300)
    assert all(H.render(H.build(r["proof"])) == r["proof"] for r in recs)


def test_order_normal_forms():
    p, b = GD
    P = H.build(b)
    v = H.v_order(P, b)
    assert v["gd"] == b  # already goal-driven
    assert v["pd"] == PD_EXPECTED  # Q hoisted out of the box
    valid(p, v["pd"])
    h = H.h_order(P, b)
    assert h["room"] and h["is_gd"] and not h["is_pd"] and h["hoistable"] == 1
    # and back: the premise-driven proof sinks to the goal-driven one
    assert H.v_order(H.build(PD_EXPECTED), PD_EXPECTED)["gd"] == b


def test_linear_extensions():
    p, b = CONJ
    P = H.build(b)
    assert H.linear_extensions(P.body) == 2  # Q and R in either order, then the ANDI
    assert H.h_order(P, b)["n_orders"] == 2


def test_conjunct_swap():
    p, b = CONJ
    P = H.build(b)
    h = H.h_conjunct(P, b)
    assert h["room"] and h["sites"] == 1 and h["left_first"] == 1
    sw = H.v_conjunct(P, b)["swap"]
    valid(p, sw)
    assert sw.index("R : IMPE") < sw.index("Q : IMPE") and nl(sw) == nl(b)


def test_reiteration_classes_and_rewrites():
    p, b = ORE
    P = H.build(b)
    cls = sorted(c for _, _, c in H.r_lines(P))
    assert cls == [
        "optional",
        "required",
    ]  # N5 closes branch 1 (required); N9 restates N2 at top level (optional)
    v = H.v_reit(P, b)
    valid(p, v["rfree"])
    valid(p, v["reit"])
    assert nl(v["rfree"]) == nl(b) - 1
    # branch 2 cites the outer N3 directly: reiteration adds one R line there
    assert nl(v["reit"]) == nl(b) + 1
    assert H.h_reit(P, b)["sites"] == 1


def test_reductio_wraps():
    for p, b in (GD, ORE, BOTE):
        v = H.v_reductio(H.build(b), b)
        for k in ("outside", "inside"):
            valid(p, v[k])
            assert nl(v[k]) == nl(b) + 4
        assert H.max_depth(H.build(v["inside"])) == H.max_depth(H.build(b)) + 1


def test_exfalso_both_ways():
    p, b = BOTE
    v = H.v_exfalso(H.build(b), b)
    valid(p, v["reductio"])
    assert nl(v["reductio"]) == nl(b) + 3
    back = H.v_exfalso(H.build(v["reductio"]), v["reductio"])
    assert back["bote"] == b  # the vacuous reductio collapses back to the BOTE line


def test_disjunct_swap():
    p, b = DISJ
    P = H.build(b)
    assert H.h_disjunct(P, b) == {"room": True, "sites": 1, "ori1": 1}
    sw = H.v_disjunct(P, b)["swap"]
    valid(p, sw)
    assert "ORI2 N2" in sw


def test_disjunct_derivable_splice():
    p = "THM P , ( P > Q ) SEQ ( P v Q ) PRF"
    b = "N1 P : PR ; N2 ( P > Q ) : PR ; N3 ( P v Q ) : ORI1 N1 ; QED"
    r = H.disj_one(({"_i": 0, "prompt": p, "proof": b}, 2, 5.0))
    assert r["sites"][0]["cost"] == 1 and r["sites"][0]["delta"] == 1
    valid(p, r["variant"])
    assert "Q : IMPE N2 N1" in r["variant"] and "ORI2" in r["variant"]


def test_redundant_split_removed():
    # branch 1 never uses its case P: the split is redundant
    p = "THM ( P v Q ) , R SEQ ( R v S ) PRF"
    b = (
        "N1 ( P v Q ) : PR ; N2 R : PR ; N3 | P : AS ; N4 | ( R v S ) : ORI1 N2 ; N5 | Q : AS "
        "; N6 | ( R v S ) : ORI1 N2 ; "
        "N7 ( R v S ) : ORE N1 N3 N4 N5 N6 ; QED"
    )
    valid(p, b)
    P = H.build(b)
    assert H.h_redsplit(P, b)["redundant"] == 1
    v = H.v_redsplit(P, b)["nosplit"]
    valid(p, v)
    assert v == "N1 ( P v Q ) : PR ; N2 R : PR ; N3 ( R v S ) : ORI1 N2 ; QED"


def test_detour():
    p, b = GD
    v = H.v_detour(H.build(b), b)["detour"]
    valid(p, v)
    assert nl(v) == nl(b) + 2 and "( ( P > Q ) & ( P > Q ) ) : ANDI N1 N1" in v


def test_state_visibility():
    # reiteration adds a duplicate hypothesis to the box's state: visible; a zero-cost ORI swap is
    # not
    p, b = ORE
    v = H.v_reit(H.build(b), b)["reit"]
    sa, _ = H.state_sigs(p, b)
    sb, _ = H.state_sigs(p, v)
    c = H.compare_states(sa, sb)
    assert c["novel_b"] > 0
    p, b = (
        "THM P , Q SEQ ( ( P v Q ) & P ) PRF",
        "N1 P : PR ; N2 Q : PR ; N3 ( P v Q ) : ORI1 N1 ; N4 ( ( P v Q ) & P ) : ANDI N3 N1 ; QED",
    )
    sw = H.v_disjunct(H.build(b), b)["swap"]
    c = H.compare_states(H.state_sigs(p, b)[0], H.state_sigs(p, sw)[0])
    assert c["novel_a"] == 0 and c["novel_b"] == 0


def test_inconsistent():
    assert H.inconsistent([("atom", "P"), ("not", ("atom", "P"))])
    assert not H.inconsistent([("atom", "P"), ("imp", ("atom", "P"), ("atom", "Q"))])


def test_lean_accepts_rewrites():
    from nd2lean import translate

    srcs = []
    for p, b in (GD, ORE, BOTE):
        P = H.build(b)
        for v in (
            list(H.v_order(P, b).values())
            + list(H.v_reit(P, b).values())
            + list(H.v_reductio(P, b).values())
        ):
            srcs.append(translate(p, v))
    srcs.append(
        translate(BOTE[0], "N1 P : PR ; N2 ( ~ P ) : PR ; N3 F : NEGE N1 N2 ; N4 Q : DN N3 ; QED")
    )  # wrong: must fail
    res = H.lean_many(srcs, 2)
    assert all(ok for ok, _ in res[:-1]) and not res[-1][0]
