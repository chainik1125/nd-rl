"""`core.formula` round-trips spec-format formulas and matches `vendor/nd_verify`'s parser."""

from __future__ import annotations

import pytest

from code.ndbench.core.formula import FormulaError, atoms, is_well_formed, parse, render
from code.ndbench.core.keys import split_thm
from code.ndbench.vendor import load_nd_verify

from ._fixtures import load_valid_proofs

FORMULAS = [
    "P",
    "F",
    "( ~ P )",
    "( P & Q )",
    "( P v ( ~ Q ) )",
    "( ( P & Q ) > ( ~ ( R v S ) ) )",
    "( ~ ( ~ P ) )",
]


@pytest.mark.parametrize("s", FORMULAS)
def test_render_parse_roundtrip(s: str) -> None:
    f = parse(s)
    assert render(f) == s
    assert is_well_formed(s)


def test_atoms() -> None:
    assert atoms(parse("( P & ( Q v ( ~ P ) ) )")) == frozenset({"P", "Q"})
    assert atoms(parse("F")) == frozenset()


def test_malformed_raises() -> None:
    with pytest.raises(FormulaError):
        parse("( P & Q")
    with pytest.raises(FormulaError):
        parse("P Q")
    assert not is_well_formed("( P & )")


def test_matches_vendored_nd_verify_on_fixture_formulas() -> None:
    """Every premise/conclusion of the 60 fixture theorems parses here exactly as
    `vendor/nd_verify.parse_formula` parses it (same AST, so `render` round-trips it back to the
    identical spec-format text nd_verify itself would accept)."""
    nd_verify = load_nd_verify()
    for p in load_valid_proofs():
        prem, concl = split_thm(p["thm"])
        for s in (*prem, concl):
            ours = parse(s)
            toks = s.split()
            theirs, i = nd_verify.parse_formula(toks, 0)
            assert i == len(toks)
            assert ours == theirs
            assert render(ours) == s
