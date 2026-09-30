"""`python -m code.ndbench --help` works, and every subcommand's own `--help` works, without
importing torch/GPU-only work-package code (plan §3 WP0 acceptance test)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from code.ndbench.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "build" in out
    assert "run" in out


@pytest.mark.parametrize(
    "sub", ["build", "run", "edl", "throughput", "import-legacy", "budget", "dash"]
)
def test_subcommand_help_exits_zero(sub: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0


def test_unimplemented_subcommand_gives_a_clear_error(tmp_path: Path) -> None:
    rc = main(["import-legacy", "--out", str(tmp_path)])
    assert rc == 1


def test_budget_with_empty_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["budget", "--out", str(tmp_path), "--cap-h", "9.5", "--est-h", "1.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"spent_h": 0.0' in out
    assert '"would_fit": true' in out


def test_module_entrypoint_help_as_a_subprocess() -> None:
    """`python -m code.ndbench --help`, run exactly as plan §A1 describes it, from the repo root."""
    result = subprocess.run(
        [sys.executable, "-m", "code.ndbench", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ndbench" in result.stdout


def test_build_parser_has_every_subcommand() -> None:
    parser = build_parser()
    # argparse doesn't expose subparser names directly; introspect the action.
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert set(subparsers_action.choices) == {
        "build",
        "run",
        "edl",
        "throughput",
        "import-legacy",
        "budget",
        "dash",
    }
