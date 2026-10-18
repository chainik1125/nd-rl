"""Tests for code/tools/tt_prover.py. The Lean checks skip when no `lean` binary is on PATH."""

import importlib.util
import shutil
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "tt_prover.py"
_SPEC = importlib.util.spec_from_file_location("tt_prover", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
tt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tt)

VALID = [
    "THM ( P > Q ) , P SEQ Q PRF",
    "THM SEQ ( P v ( ~ P ) ) PRF",
    "THM SEQ ( ( ( P > Q ) > P ) > P ) PRF",
    "THM F SEQ Q PRF",
    "THM SEQ ( F > F ) PRF",
    "THM ( P v Q ) , ( ~ P ) SEQ Q PRF",
    "THM ( ( P & Q ) & ( R & S ) ) SEQ ( ( S v R ) & ( Q v P ) ) PRF",
    "THM ( ~ ( P & Q ) ) SEQ ( ( ~ P ) v ( ~ Q ) ) PRF",
]


def test_parse_and_render() -> None:
    prem, goal = tt.parse_prompt("THM ( P > Q ) , ( ~ R ) SEQ ( P & F ) PRF")
    assert [tt.lean(p) for p in prem] == ["(P → Q)", "¬R"]
    assert tt.lean(goal) == "(P ∧ False)"


@pytest.mark.parametrize("prompt", VALID)
def test_valid_sequents_get_proofs(prompt: str) -> None:
    assert tt.countermodel(prompt) is None
    src = tt.solve(prompt)
    assert src is not None and src.startswith("theorem t (P Q R S : Prop)")
    assert "sorry" not in src


def test_invalid_sequent_gets_countermodel() -> None:
    assert tt.solve("THM SEQ ( P > Q ) PRF") is None
    assert tt.countermodel("THM SEQ ( P > Q ) PRF") == {"P": True, "Q": False}


@pytest.mark.skipif(shutil.which("lean") is None, reason="no lean binary on PATH")
@pytest.mark.parametrize("prompt", VALID)
def test_lean_accepts(prompt: str) -> None:
    src = tt.solve(prompt)
    assert src is not None
    ok, msg = tt.lean_ok(src)
    assert ok, msg
