"""Propositional formula parsing/rendering for the take-home's spec-format token strings.

A formula is a whitespace-tokenised string over the fixed 4-atom alphabet `P Q R S`, negation
`~`, binary connectives `& v >`, falsum `F`, and parentheses, e.g. `( P & ( ~ Q ) )`. This mirrors
the grammar `vendor/nd_verify/verify.py`'s `parse_formula` accepts, reimplemented here (pure
stdlib) so `core` has no dependency on the vendor sys.path shim. `Formula` is the same nested-tuple
AST nd_verify and the vendored take-home scripts use (`('atom', 'P')`, `('not', f)`,
`('and'|'or'|'imp', f, g)`, `('bot',)`), so ASTs built or matched here interoperate with vendored
code without translation.

`code/tests/ndbench/test_wp0_formula.py` checks `render(parse(s)) == s` on fixture formulas and
cross-checks `parse`/`render` against `vendor/nd_verify` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: `('atom', name)` | `('bot',)` | `('not', Formula)` | `('and'|'or'|'imp', Formula, Formula)`.
Formula = tuple[Any, ...]

ATOMS: tuple[str, ...] = ("P", "Q", "R", "S")
BOT: Formula = ("bot",)
_BINOPS = {"&": "and", "v": "or", ">": "imp"}
_BINOPS_INV = {v: k for k, v in _BINOPS.items()}


class FormulaError(Exception):
    """Raised when a token string is not a well-formed formula."""


def parse_formula(tokens: Sequence[str], i: int) -> tuple[Formula, int]:
    """Recursive-descent parse of one formula starting at `tokens[i]`.

    Returns `(formula, next_i)`, mirroring `nd_verify.verify.parse_formula` exactly (same grammar,
    same AST shape), so it can be used as a drop-in substitute wherever an index into a larger
    token stream (a full sequent, a proof line) is needed.
    """
    if i >= len(tokens):
        raise FormulaError("eof in formula")
    t = tokens[i]
    if t in ATOMS:
        return ("atom", t), i + 1
    if t == "F":
        return BOT, i + 1
    if t == "(":
        if i + 1 < len(tokens) and tokens[i + 1] == "~":
            sub, j = parse_formula(tokens, i + 2)
            if j >= len(tokens) or tokens[j] != ")":
                raise FormulaError("missing ) after ~")
            return ("not", sub), j + 1
        left, j = parse_formula(tokens, i + 1)
        if j >= len(tokens) or tokens[j] not in _BINOPS:
            raise FormulaError("missing binop")
        op = _BINOPS[tokens[j]]
        right, k = parse_formula(tokens, j + 1)
        if k >= len(tokens) or tokens[k] != ")":
            raise FormulaError("missing )")
        return (op, left, right), k + 1
    raise FormulaError(f"bad formula token {t!r}")


def parse(s: str) -> Formula:
    """Parse a whole formula string, requiring every token to be consumed."""
    toks = s.split()
    f, i = parse_formula(toks, 0)
    if i != len(toks):
        raise FormulaError(f"trailing tokens after formula: {toks[i:]!r}")
    return f


def render(f: Formula) -> str:
    """Render a `Formula` AST back to spec-format text (matches `takehome/gen.py`'s `fstr`)."""
    if f[0] == "atom":
        return f[1]
    if f[0] == "bot":
        return "F"
    if f[0] == "not":
        return f"( ~ {render(f[1])} )"
    op = _BINOPS_INV[f[0]]
    return f"( {render(f[1])} {op} {render(f[2])} )"


def is_well_formed(s: str) -> bool:
    """True iff `s` parses as a single, fully-consumed formula."""
    try:
        parse(s)
    except FormulaError:
        return False
    return True


def atoms(f: Formula) -> frozenset[str]:
    """The set of atom names occurring in `f`."""
    if f[0] == "atom":
        return frozenset((f[1],))
    if f[0] == "bot":
        return frozenset()
    if f[0] == "not":
        return atoms(f[1])
    return atoms(f[1]) | atoms(f[2])
