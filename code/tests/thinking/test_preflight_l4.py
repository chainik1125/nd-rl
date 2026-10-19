"""grpo_v2/preflight.py with the optional third (L4, 16-18-line) pool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.experiments.current.thinking_traces_20261005.grpo_v2 import preflight as PF


def _write(p: Path, rows: list[tuple[str, int]]) -> Path:
    p.write_text("".join(json.dumps({"prompt": q, "min_lines": n}) + "\n" for q, n in rows))
    return p


@pytest.fixture
def sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    ev = _write(tmp_path / "eval.jsonl", [("THM ( P & Q ) SEQ ( Q & P ) PRF", 3)])
    l4t = _write(tmp_path / "L4_test.jsonl", [("THM ( P v Q ) , ( ~ P ) SEQ Q PRF", 16)])
    monkeypatch.setattr(PF, "EVAL_SETS", {"eval": [ev]})
    monkeypatch.setattr(PF, "L4_TEST", {"curriculum L4 test_1k": [l4t]})
    return {
        "l4_test": l4t,
        "L2": _write(tmp_path / "L2.jsonl", [("THM P SEQ ( Q > P ) PRF", 10)]),
        "L3": _write(tmp_path / "L3.jsonl", [("THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF", 13)]),
        "L4": _write(tmp_path / "L4.jsonl", [("THM ( ~ ( P & Q ) ) SEQ ( P > ( ~ Q ) ) PRF", 17)]),
    }


def test_two_pools_unchanged(sets: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert PF.main([str(sets["L2"]), str(sets["L3"])]) == 0
    out = capsys.readouterr().out
    assert "L4" not in out  # no L4 band, pair or test-split lines: line.sh counts the ok lines
    assert out.count("\nok ") == 2 and out.rstrip().endswith("PREFLIGHT PASSED")


def test_three_pools(sets: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert PF.main([str(sets["L2"]), str(sets["L3"]), str(sets["L4"])]) == 0
    out = capsys.readouterr().out
    assert "pool L4: 1 rows, lengths 17-17 (band 16-18)" in out
    assert out.count("\nok ") == 6  # 3 pools x (eval, L4 test split)


def test_l4_curried_twin_of_l3_fails(
    sets: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # |- P > ((Q > R) > ... ) curried: the L3 row's problem with its premises as antecedents
    twin = _write(
        tmp_path / "twin.jsonl", [("THM ( P > Q ) SEQ ( ( Q > R ) > ( P > R ) ) PRF", 16)]
    )
    assert PF.main([str(sets["L2"]), str(sets["L3"]), str(twin)]) == 1
    assert "BAD: L3 and L4 share 0 prompts, 0 keys, 1 classes" in capsys.readouterr().out


def test_l4_band_and_test_split_self(
    sets: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # the L4 test split checked as the third pool: its comparison with itself is skipped
    assert PF.main([str(sets["L2"]), str(sets["L3"]), str(sets["l4_test"])]) == 0
    assert "(self L4 vs curriculum L4 test_1k: the same file)" in capsys.readouterr().out
    short = _write(tmp_path / "short.jsonl", [("THM ( P > Q ) , P SEQ ( Q v R ) PRF", 15)])
    assert PF.main([str(sets["L2"]), str(sets["L3"]), str(short)]) == 1
    assert "BAD: 1 rows outside 16-18" in capsys.readouterr().out
