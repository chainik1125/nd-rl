"""The vendored Lean pre-filter's reject codes and failing-line names (retry error tokens)."""

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[1] / "experiments/current/retry_ei_20261008/lean_prefilter.py"
_spec = importlib.util.spec_from_file_location("retry_ei_lean_prefilter", _SRC)
assert _spec is not None and _spec.loader is not None
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)

HEAD = "theorem t ( P Q R S : Prop )"
C = f"{HEAD} ( h1 : P ) ( h2 : ( P → Q ) ) : Q := by"
INTRO = f"{HEAD} ( h1 : P ) : ( ¬ ( ¬ P ) ) := by"

CASES = [
    # (statement, text, expected detail)
    (C, "have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : Q := n2 n1 ; exact n3", None),
    (
        C,
        "have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : Q := n2 n2 ; exact n3",
        ("app-arg", "n3"),
    ),
    (
        C,
        "have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : P := n2 n1 ; exact n3",
        ("app-result", "n3"),
    ),
    (C, "have n1 : Q := h1 ; exact n1", ("premise-type", "n1")),
    # type-checks, proves the wrong thing: no line to point at
    (C, "have n1 : P := h1 ; exact n1", ("goal", None)),
    # an error inside a subproof is located at the inner `have`, not the enclosing one
    (
        INTRO,
        "have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : ( P → False ) ) => by "
        "have n4 : False := n1 n3 ; exact ( n4 : False ) ) ; exact n2",
        ("function-expected", "n4"),
    ),
    # an error in the subproof's own shape is located at the `have` that introduces it
    (
        INTRO,
        "have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : P ) => by exact n1 ) ; "
        "exact n2",
        ("fun-binder", "n2"),
    ),
]


@pytest.mark.parametrize(("statement", "text", "want"), CASES)
def test_reject_detail(statement: str, text: str, want: tuple[str, str | None] | None):
    assert lp.reject_detail(statement, text) == want
    assert lp.reject_reason(statement, text) == (want and want[0])
