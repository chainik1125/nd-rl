"""Tests of authors_profile (the habit profile of sampled proofs) on hand-written proofs, the corpus's own rewrites,
and the four corpora (agreement with each record's detector reading and assigned forms).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_profile.py
"""

# ruff: noqa: E501

from __future__ import annotations

import gzip
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_profile as AP  # noqa: E402

H = AP.H
from nd_verify import verify_text  # noqa: E402

DATA = Path("/scratch/cowifornia/nd-rl/two_authors/data")
# ( P > Q ) , ( Q > S ) , P |- ( R > S ): goal-driven (Q derived inside the box) and premise-driven (Q first)
GD = (
    "THM ( P > Q ) , ( Q > S ) , P SEQ ( R > S ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 | R : AS ; N5 | Q : IMPE N1 N3 ; "
    "N6 | S : IMPE N2 N5 ; N7 ( R > S ) : IMPI N4 N6 ; QED",
)
PD = (
    GD[0],
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 | R : AS ; "
    "N6 | S : IMPE N2 N4 ; N7 ( R > S ) : IMPI N5 N6 ; QED",
)
# the premise-driven proof with author A's reiteration: Q and ( Q > S ) restated inside the box
PD_REIT = (
    GD[0],
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 | R : AS ; "
    "N6 | Q : R N4 ; N7 | ( Q > S ) : R N2 ; N8 | S : IMPE N7 N6 ; N9 ( R > S ) : IMPI N5 N8 ; QED",
)
# one of the two sites restated: reiteration mixed within the proof
PD_HALF = (
    GD[0],
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 | R : AS ; "
    "N6 | Q : R N4 ; N7 | S : IMPE N2 N6 ; N8 ( R > S ) : IMPI N5 N7 ; QED",
)
# two independent conjuncts: left first (B) and right first (A); a free site (no box, so no step order to decide it)
CONJ_L = (
    "THM ( P > Q ) , ( P > R ) , P SEQ ( Q & R ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( P > R ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 R : IMPE N2 N3 ; "
    "N6 ( Q & R ) : ANDI N4 N5 ; QED",
)
CONJ_R = (
    CONJ_L[0],
    "N1 ( P > Q ) : PR ; N2 ( P > R ) : PR ; N3 P : PR ; N4 R : IMPE N2 N3 ; N5 Q : IMPE N1 N3 ; "
    "N6 ( Q & R ) : ANDI N5 N4 ; QED",
)
# the route: a reductio that does real work (DN of a NEGI), on a classical-only theorem
ROUTE = (
    "THM ( ( ~ Q ) > F ) SEQ Q PRF",
    "N1 ( ( ~ Q ) > F ) : PR ; N2 | ( ~ Q ) : AS ; N3 | F : IMPE N1 N2 ; N4 ( ~ ( ~ Q ) ) : NEGI N2 N3 ; "
    "N5 Q : DN N4 ; QED",
)
# the trap: DN of an R copy (inside a box) of a premise ~~P; Dan's knock calls it the route, it is not
RCOPY = (
    "THM ( ~ ( ~ P ) ) , Q SEQ ( Q > P ) PRF",
    "N1 ( ~ ( ~ P ) ) : PR ; N2 Q : PR ; N3 | Q : AS ; N4 | ( ~ ( ~ P ) ) : R N1 ; N5 | P : DN N4 ; "
    "N6 ( Q > P ) : IMPI N3 N5 ; QED",
)
RCOPY_TOP = (
    "THM ( ~ ( ~ P ) ) SEQ P PRF",
    "N1 ( ~ ( ~ P ) ) : PR ; N2 ( ~ ( ~ P ) ) : R N1 ; N3 P : DN N2 ; QED",
)
# the route reached through an R copy of the NEGI line: still the route
ROUTE_RCOPY = (
    ROUTE[0],
    "N1 ( ( ~ Q ) > F ) : PR ; N2 | ( ~ Q ) : AS ; N3 | F : IMPE N1 N2 ; N4 ( ~ ( ~ Q ) ) : NEGI N2 N3 ; "
    "N5 ( ~ ( ~ Q ) ) : R N4 ; N6 Q : DN N5 ; QED",
)
DEAD = ("THM P , Q SEQ ( P & P ) PRF", "N1 P : PR ; N2 Q : PR ; N3 ( P & P ) : ANDI N1 N1 ; QED")
DEAD_WITH = (
    DEAD[0],
    "N1 P : PR ; N2 Q : PR ; N3 ( Q & Q ) : ANDI N2 N2 ; N4 ( P & P ) : ANDI N1 N1 ; QED",
)
ALL = (
    GD,
    PD,
    PD_REIT,
    PD_HALF,
    CONJ_L,
    CONJ_R,
    ROUTE,
    RCOPY,
    RCOPY_TOP,
    ROUTE_RCOPY,
    DEAD,
    DEAD_WITH,
)


def valid(prompt: str, body: str) -> None:
    ok, why, _ = verify_text(prompt + " " + body)
    assert ok, (why, body)


def test_fixtures_valid():
    for p, b in ALL:
        valid(p, b)


def test_step_order_forms():
    assert AP.profile(*GD)["order"] == {"sites": 1, "a": 0, "b": 1, "form": "B"}
    assert AP.profile(*PD)["order"] == {"sites": 1, "a": 1, "b": 0, "form": "A"}
    # the corpus's normal forms are what the profile calls A and B
    assert H.render(AP.AC.normal_form(H.build(GD[1]), "pd")) == PD[1]
    assert H.render(AP.AC.normal_form(H.build(PD[1]), "gd")) == GD[1]
    assert AP.profile(*CONJ_L)["order"] == {
        "sites": 0,
        "a": 0,
        "b": 0,
        "form": None,
    }  # no box: no room


def test_reiteration_per_site_and_order_read_through_it():
    assert AP.profile(*PD)["reit"] == {"sites": 2, "a": 0, "b": 2, "form": "B"}
    q = AP.profile(*PD_REIT)
    assert q["reit"] == {"sites": 2, "a": 2, "b": 0, "form": "A"}  # per site, not per R line
    assert q["order"]["form"] == "A"  # the R copies are stripped before step order is read
    r = AP.profile(*PD_HALF)
    assert r["reit"] == {"sites": 2, "a": 1, "b": 1, "form": "M"} and r["bundle"] == "mixed"
    assert AP.profile(*GD)["reit"]["sites"] == 3  # the goal-driven box cites N1, N2 and N3


def test_conjunct_order_free_sites():
    assert AP.profile(*CONJ_L)["conj"] == {"sites": 1, "a": 0, "b": 1, "form": "B"}
    assert AP.profile(*CONJ_R)["conj"] == {"sites": 1, "a": 1, "b": 0, "form": "A"}
    assert AP.profile(*CONJ_R)["conj_all"] == {"sites": 1, "a": 1}


def test_bundle_classes_and_joint():
    a = AP.profile(*PD_REIT)
    assert (a["bundle"], a["n_room"], a["joint"]) == ("A", 2, "oA.rA.c-")
    b = AP.profile(*GD)
    assert (b["bundle"], b["joint"]) == ("B", "oB.rB.c-")
    m = AP.profile(*PD)  # premise-driven but citing directly
    assert (m["bundle"], m["joint"]) == ("mixed", "oA.rB.c-")
    assert AP.profile(*CONJ_L)["bundle"] == "B"
    none = AP.profile("THM P SEQ P PRF", "N1 P : PR ; N2 P : R N1 ; QED")
    assert none["n_room"] == 0 and none["bundle"] is None and none["joint"] == "o-.r-.c-"


def test_route_follows_reiteration():
    r = AP.profile(*ROUTE)
    assert r["route"] is True and r["route_knock"] is True and r["sigma"] is True
    assert AP.profile(*ROUTE_RCOPY)["route"] is True
    for p in (RCOPY, RCOPY_TOP):
        x = AP.profile(*p)
        assert x["route_knock"] is True  # Dan's knock is fooled by the R copy
        assert (
            x["route"] is False and x["sigma"] is False
        )  # the profile follows the copy back to the premise
    for p in (GD, PD, PD_REIT, CONJ_L):
        assert AP.profile(*p)["route"] is False


def test_reit_fits():
    assert AP.profile(*PD)["reit_fits"] is True  # 7 lines + 2 restatements
    long_ = (
        "THM ( P > Q ) , ( Q > R ) , ( R > S ) SEQ ( P > ( P > S ) ) PRF",
        "N1 ( P > Q ) : PR ; N2 ( Q > R ) : PR ; N3 ( R > S ) : PR ; N4 | P : AS ; N5 | | P : AS ; "
        "N6 | | Q : IMPE N1 N5 ; N7 | | R : IMPE N2 N6 ; N8 | | S : IMPE N3 N7 ; N9 | ( P > S ) : IMPI N5 N8 ; "
        "N10 ( P > ( P > S ) ) : IMPI N4 N9 ; QED",
    )
    valid(*long_)
    p = AP.profile(*long_)
    assert p["reit"]["sites"] == 3 and p["lines"] == 10 and p["reit_fits"] is False  # 10 + 3 > 12


def test_dead_lines_pruned_first():
    p = AP.profile(*DEAD_WITH)
    assert p["dead"] == 1
    assert {h: p[h]["sites"] for h in AP.HABITS} == {
        h: AP.profile(*DEAD)[h]["sites"] for h in AP.HABITS
    }


def test_unparsable_proof():
    p = AP.profile("THM P SEQ P PRF", "N1 P : PR ; N2 | Q : AS ; QED")
    assert p["ok"] is False and p["err"].startswith("parse")


def test_corpus_rewrites_read_back():
    """The corpus's rewrite (authors_corpus.rewrite) of hand proofs into each author's forms reads back as that author."""
    assert AP.author_form(*GD, "A") == PD_REIT[1]
    assert AP.profile(GD[0], AP.author_form(*PD_REIT, "B"))["joint"] == "oB.rB.c-"
    assert AP.author_form(*CONJ_L, "A") == CONJ_R[1]
    assert AP.author_form(*CONJ_R, "B") == CONJ_L[1]
    only_conj = AP.author_form(*CONJ_L, "B", ("B", "A", "B"))
    assert AP.profile(CONJ_L[0], only_conj)["conj"]["form"] == "A"


def stride_sample(fn: Path, n: int) -> list[dict]:
    with gzip.open(fn, "rt") as f:
        recs = [json.loads(x) for x in f]
    step = max(1, len(recs) // n)
    return recs[::step][:n]


@pytest.mark.parametrize("corpus", ["authors_half", "control_half"])
def test_agreement_with_the_corpus(corpus):
    """On 3,000 records spread over the corpus: the profile equals each record's `realized` (the corpus's detector) on
    every habit count and the route, and reads every habit with room (not skipped) in the record's assigned form."""
    fn = DATA / f"{corpus}.jsonl.gz"
    if not fn.exists():
        pytest.skip(f"no {fn}")
    agree = {h: [0, 0] for h in AP.HABITS}
    for r in stride_sample(fn, 3000):
        p = AP.profile(r["prompt"], r["proof"])
        assert p["ok"] and p["dead"] == 0
        real = AP.from_detect(r["realized"])
        for h in AP.HABITS:
            assert {k: p[h][k] for k in ("sites", "a", "b")} == {
                k: real[h][k] for k in ("sites", "a", "b")
            }, (h, r["key"])
        assert p["route"] == r["route"]
        for h in AP.HABITS:
            if p[h]["form"] is not None and AP.CORPUS_HABIT[h] not in (r.get("skipped") or {}):
                agree[h][1] += 1
                agree[h][0] += p[h]["form"] == r["forms"][AP.CORPUS_HABIT[h]]
    for h, (ok, n) in agree.items():
        assert n > (20 if h == "conj" else 500), (h, n)
        assert ok == n, (corpus, h, ok, n)


def test_route_records_stay_the_route():
    """Route records of the authors corpus are the route under the profile (sigma+ with or without following R)."""
    fn = DATA / "authors_half.jsonl.gz"
    if not fn.exists():
        pytest.skip(f"no {fn}")
    with gzip.open(fn, "rt") as f:
        recs = [r for r in (json.loads(x) for x in itertools.islice(f, 60000)) if r["route"]][:200]
    assert len(recs) >= 100
    assert all(AP.profile(r["prompt"], r["proof"])["route"] for r in recs)
