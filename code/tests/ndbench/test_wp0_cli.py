"""`python -m code.ndbench --help` works, and every subcommand's own `--help` works, without
importing torch/GPU-only work-package code (plan §3 WP0 acceptance test)."""

from __future__ import annotations

import argparse
import json
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


def test_unimplemented_subcommand_gives_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_resolve_registry` (used by every subcommand for its adapter/judge/suite/builder lookup)
    should turn a registry entry whose module doesn't exist yet into a friendly `SystemExit`
    rather than a raw `ModuleNotFoundError` traceback.

    `import-legacy` used to be this module's example of an unimplemented subcommand, but WP2
    implemented it (`suites.existing:import_legacy`), so that call now either succeeds or fails
    for unrelated reasons (missing legacy data files) rather than exercising this path. Instead,
    monkeypatch one registry entry (`ADAPTERS["fake"]`) to point at a module that does not exist,
    and drive it through `run`, which resolves the model adapter before anything else."""
    from code.ndbench import cli

    monkeypatch.setitem(cli.ADAPTERS, "fake", "code.ndbench.adapters._no_such_module:Nope")

    rc = main(
        [
            "run",
            "--suite",
            "textbook_dev",
            "--version",
            "v1",
            "--model",
            "fake",
            "--k",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "adapter 'fake'" in err
    assert "not implemented yet" in err


def test_budget_with_empty_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["budget", "--out", str(tmp_path), "--cap-h", "9.5", "--est-h", "1.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"spent_h": 0.0' in out
    assert '"would_fit": true' in out


def _write_schedule(tmp_path: Path, *, cap_h: float, jobs: list[dict[str, object]]) -> Path:
    import yaml

    path = tmp_path / "schedule.yaml"
    path.write_text(yaml.safe_dump({"cap_h": cap_h, "jobs": jobs}))
    return path


def test_budget_schedule_dry_run_never_touches_the_gpu_and_applies_the_stop_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WP8 integration fix list item 10: `budget --schedule` prints the plan §4 stop-rule table
    without touching the GPU. A job whose cumulative `spent + 1.5*est` would exceed the cap is
    skipped, and the walk continues to the *next* job in the list (not retried later)."""
    schedule = _write_schedule(
        tmp_path,
        cap_h=1.0,
        jobs=[
            {"id": "A", "job": "a", "est_min": 20},  # 1/3 h; 0 + 1.5*(1/3) = 0.5 <= 1.0 -> fits
            {"id": "B", "job": "b", "est_min": 40},  # 2/3 h; 1/3 + 1.5*(2/3) = 1.33 > 1.0 -> skip
            {"id": "C", "job": "c", "est_min": 6},  # 0.1h; 1/3 + 0.15 = 0.483 <= 1.0 -> fits
            {"id": "R", "job": "reserve", "est_min": None},  # never auto-scheduled
        ],
    )
    rc = main(["budget", "--out", str(tmp_path / "out"), "--schedule", str(schedule)])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["spent_h"] == 0.0
    assert result["cap_h"] == 1.0
    assert result["n_jobs"] == 4
    assert result["n_scheduled"] == 2
    assert result["n_skipped"] == 1
    statuses = {row["job"]: row["status"] for row in result["rows"]}
    assert statuses == {
        "a": "scheduled",
        "b": "skipped",
        "c": "scheduled",
        "reserve": "unestimated",
    }
    # the real schedule.yaml itself parses and every non-reserve job fits under the plan's own cap.
    real_schedule = (
        Path(__file__).resolve().parents[2]
        / "experiments/current/benchmarking_20260930/schedule.yaml"
    )
    if real_schedule.is_file():
        rc2 = main(["budget", "--out", str(tmp_path / "out2"), "--schedule", str(real_schedule)])
        assert rc2 == 0
        real_result = json.loads(capsys.readouterr().out)
        assert real_result["n_skipped"] == 0
        assert real_result["projected_total_h"] < 9.5


def test_budget_schedule_cap_h_override_wins_over_the_schedule_files_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schedule = _write_schedule(tmp_path, cap_h=9.5, jobs=[{"id": "A", "job": "a", "est_min": 60}])
    rc = main(
        [
            "budget",
            "--out",
            str(tmp_path / "out"),
            "--schedule",
            str(schedule),
            "--cap-h",
            "0.1",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cap_h"] == 0.1
    assert result["rows"][0]["status"] == "skipped"


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
        "figures",
    }


def test_build_suite_choices_include_glue_length_sweep() -> None:
    """WP8 integration fix list item 7: `suites.glue.build_length_sweep` matches the builder
    contract but WP4 left it unregistered ("that registry is WP0-owned"); it must be reachable as
    `--suite glue_length_sweep` like every other builder."""
    parser = build_parser()
    build_sub = next(
        a
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction) and "build" in a.choices
    ).choices["build"]
    suite_action = next(a for a in build_sub._actions if a.dest == "suite")
    assert "glue_length_sweep" in suite_action.choices
    assert "glue" in suite_action.choices  # unaffected sibling entry, still present


def test_resolve_instance_instantiates_a_bare_class_but_passes_through_an_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP8 integration fix list item 10: `build`'s `cfg["certifier"]`/`cfg["excluder"]` may name a
    zero-arg-constructible class directly (e.g. `configs/textbook_dev.yaml`'s `certifier:
    code.ndbench.logic.certify:MinlenCertifier`) -- `cmd_build` must instantiate it, not hand the
    bare class object to a builder expecting a `Certifier` *instance*. An already-built instance
    named the same way must pass through unchanged (not be called again)."""
    from code.ndbench.cli import _resolve_instance
    from code.ndbench.logic import certify
    from code.ndbench.logic.certify import MinlenCertifier

    certifier = _resolve_instance("code.ndbench.logic.certify:MinlenCertifier")
    assert isinstance(certifier, MinlenCertifier)
    assert not isinstance(certifier, type)

    already_built = MinlenCertifier(procs=1)
    monkeypatch.setattr(certify, "_test_only_instance", already_built, raising=False)
    assert _resolve_instance("code.ndbench.logic.certify:_test_only_instance") is already_built


def test_builders_registry_resolves_glue_length_sweep() -> None:
    from code.ndbench.suites import BUILDERS
    from code.ndbench.suites.glue import build_length_sweep

    module_name, _, attr = BUILDERS["glue_length_sweep"].partition(":")
    import importlib

    resolved = getattr(importlib.import_module(module_name), attr)
    assert resolved is build_length_sweep
