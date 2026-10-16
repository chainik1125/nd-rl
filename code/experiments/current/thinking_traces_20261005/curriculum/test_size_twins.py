"""Tests for the big-statement twins (synthetic theorems only, CPU, a few seconds).

    cd /scratch/cowifornia/nd-rl/think/nd-rl-size
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m pytest -q -p no:cacheprovider \
        code/experiments/current/thinking_traces_20261005/curriculum/test_size_twins.py

The fixture theorems were written for these tests; `test_fixtures_not_textbook` checks that none is
a textbook72 substitution instance (or the other way round).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = str(Path(__file__).resolve().parents[5])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]  # the stdlib `code` module, not this repository's package

from code.experiments.current.thinking_traces_20261005.curriculum import (  # noqa: E402
    size_twins as SZ,
)

pytestmark = pytest.mark.skipif(
    not Path(SZ.T1320).exists(), reason="needs the sealed test_13_20 sizes on /scratch"
)

#: ( P & Q ) |- ( Q & ( P v R ) ), 5 lines, minimal.
ROW: dict[str, Any] = {
    "name": "fixture_and_or",
    "prompt": "THM ( P & Q ) SEQ ( Q & ( P v R ) ) PRF",
    "reference_proof": "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; "
    "N4 ( P v R ) : ORI1 N3 ; N5 ( Q & ( P v R ) ) : ANDI N2 N4 ; QED",
    "min_lines": 5,
    "gen_lines": 5,
    "gen_proof": "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; "
    "N4 ( P v R ) : ORI1 N3 ; N5 ( Q & ( P v R ) ) : ANDI N2 N4 ; QED",
    "families": [],
    "attrs": {"classical": False},
}


def test_fixtures_not_textbook() -> None:
    prem, goal = SZ.GC.parse_prompt(ROW["prompt"])
    assert SZ.GT.textbook_match(prem, goal) is None


def test_subst_is_simultaneous() -> None:
    sig = {"P": "( Q & P )", "Q": "( ~ Q )"}
    assert SZ.subst("THM P SEQ ( P v Q ) PRF", sig) == (
        "THM ( Q & P ) SEQ ( ( Q & P ) v ( ~ Q ) ) PRF"
    )
    # rule names and line names are untouched
    assert (
        SZ.subst("N1 P : PR ; N2 | Q : AS ; QED", sig)
        == "N1 ( Q & P ) : PR ; N2 | ( ~ Q ) : AS ; QED"
    )


def test_independent() -> None:
    assert SZ.independent({"P": "( P & S )", "Q": "( ~ Q )", "R": "( R v R )"})
    assert SZ.independent({"P": "( P & ( P v Q ) )", "Q": "( Q > S )"})  # P & (P v Q) == P
    assert not SZ.independent({"P": "( P & S )", "Q": "( Q v S )"})  # P & S forces Q v S
    assert not SZ.independent({"P": "( P & Q )", "Q": "Q"})
    assert not SZ.independent({"P": "( P v ( ~ P ) )"})  # a tautology is not contingent


def test_candidates_deterministic() -> None:
    a = SZ.candidates("k1", ["P", "Q", "R"])
    assert a == SZ.candidates("k1", ["P", "Q", "R"])
    assert a != SZ.candidates("k2", ["P", "Q", "R"])
    assert len(a) == SZ.N_CAND
    for sig, grp in a[:50]:
        for at, img in sig.items():
            others = [b for b in SZ.ATOMS if b != at]
            renders = {
                " ".join({"a": at, "x": x, "y": y}.get(c, c) for c in t.split())
                for t in SZ.TEMPLATES[grp[at]]
                for x, y in itertools.permutations(others, 2)
            }
            assert img in renders and at in img.split()
    t = SZ.target("k1")
    assert SZ.LO_TOK <= t <= SZ.HI_TOK and t == SZ.target("k1")


def test_ranked_window_and_order() -> None:
    order = SZ.ranked(ROW, "fixture")
    assert order
    tgt = SZ.target("fixture")
    ds = [d for d, *_ in order]
    assert ds == sorted(ds)
    for d, _, sig, _ in order:
        w = len(SZ.subst(ROW["prompt"], sig).split())
        assert SZ.LO_TOK <= w <= SZ.HI_TOK and abs(w - tgt) == d


def test_try_sigma_certifies_same_length() -> None:
    sig = {"P": "( ( P v S ) & ( P v S ) )", "Q": "( ( ~ Q ) > ( Q & Q ) )", "R": "( ~ ( R > R ) )"}
    # R -> ~(R > R) is a contradiction: not independent
    assert SZ.try_sigma(ROW, sig)["outcome"] == "dependent"
    sig = {"P": "( ( P v S ) & ( P v S ) )", "Q": "( ( ~ Q ) > ( Q & Q ) )", "R": "( R & R )"}
    att = SZ.try_sigma(ROW, sig)
    assert att["outcome"] == "ok", att
    ok, _, n = SZ.GT.verify_text(att["prompt"] + " " + att["reference_proof"])
    assert ok and n == ROW["min_lines"]
    # same rules and citations, line by line
    a = [ln["rule"] for ln in SZ.GC.parse_proof(ROW["reference_proof"])]
    b = [ln["rule"] for ln in SZ.GC.parse_proof(att["reference_proof"])]
    assert a == b


def test_textbook_instance_rejected() -> None:
    # ( P > Q ) , ( Q > R ) |- ( P > R ) (hypothetical syllogism) is in textbook72 up to renaming;
    # every twin of it is a substitution instance and must be rejected.
    row: dict[str, Any] = {
        "name": "fixture_hs",
        "prompt": "THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF",
        "reference_proof": "N1 ( P > Q ) : PR ; N2 ( Q > R ) : PR ; N3 | P : AS ; "
        "N4 | Q : IMPE N1 N3 ; N5 | R : IMPE N2 N4 ; N6 ( P > R ) : IMPI N3 N5 ; QED",
        "min_lines": 6,
        "gen_lines": 6,
        "gen_proof": "N1 ( P > Q ) : PR ; N2 ( Q > R ) : PR ; N3 | P : AS ; "
        "N4 | Q : IMPE N1 N3 ; N5 | R : IMPE N2 N4 ; N6 ( P > R ) : IMPI N3 N5 ; QED",
        "families": [],
        "attrs": {"classical": False},
    }
    prem, goal = SZ.GC.parse_prompt(row["prompt"])
    if SZ.GT.textbook_match(prem, goal) is None:
        pytest.skip("hypothetical syllogism is not in this textbook72 copy")
    sig = {"P": "( P & S )", "Q": "( ~ Q )", "R": "( R v R )"}
    assert SZ.try_sigma(row, sig)["outcome"].startswith("gate:textbook")


def test_classical_lost_rejected() -> None:
    # ~~P |- ( P v Q ) needs classical logic; P -> ~P makes it intuitionistic (~~~P |- ~P v Q)
    row: dict[str, Any] = {
        "name": "fixture_dn",
        "prompt": "THM ( ~ ( ~ P ) ) SEQ ( P v Q ) PRF",
        "reference_proof": "N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; N3 ( P v Q ) : ORI1 N2 ; QED",
        "min_lines": 3,
        "gen_lines": 3,
        "gen_proof": "N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; N3 ( P v Q ) : ORI1 N2 ; QED",
        "families": [],
        "attrs": {"classical": True},
    }
    att = SZ.try_sigma(row, {"P": "( ~ P )", "Q": "( Q & S )"})
    assert att["outcome"] == "gate:classical_lost", att
