"""Pipeline construction checks; all job submissions are mocked."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from r1_cot import make_pair


def build(tmp_path, *extra):
    (tmp_path / "plans").mkdir()
    argv = [
        "make_pair.py",
        "--root",
        str(tmp_path),
        "--name",
        "test",
        "--parent",
        "/models/cot/final.pt",
        *extra,
    ]
    with patch.object(sys, "argv", argv), patch.object(make_pair.subprocess, "run") as submit:
        make_pair.main()
        return [
            json.loads(path.read_text()) for path in sorted((tmp_path / "plans").glob("*.json"))
        ], submit.call_count


def stage(spec, name):
    return next(row for row in spec["stages"] if row.get("name") == name)


def option(command, flag):
    return command[command.index(flag) + 1]


def test_default_pair_retains_training_settings(tmp_path):
    specs, submissions = build(tmp_path)
    assert submissions == 2
    assert len(specs) == 2
    for spec in specs:
        assert spec["stages"][0]["name"] == "sft"
        rl = stage(spec, "rl")["command"]
        for flag, value in (
            ("--temperature", "0.8"),
            ("--group-size", "8"),
            ("--prompts-per-step", "8"),
            ("--retention-weight", "0.1"),
            ("--advantage", "loo"),
            ("--loss-normalization", "fixed"),
        ):
            assert option(rl, flag) == value
        assert spec["experiment_metadata"]["sft_performed"] is True


def test_requested_rl_knobs_and_mode_reach_command(tmp_path):
    specs, submissions = build(
        tmp_path,
        "--temperature",
        "1.2",
        "--group-size",
        "32",
        "--prompts-per-step",
        "2",
        "--retention-weight",
        "0",
        "--advantage",
        "standardized",
        "--loss-normalization",
        "sequence",
        "--modes",
        "cot",
    )
    assert submissions == 1
    rl = stage(specs[0], "rl")["command"]
    for flag, value in (
        ("--temperature", "1.2"),
        ("--group-size", "32"),
        ("--prompts-per-step", "2"),
        ("--retention-weight", "0.0"),
        ("--advantage", "standardized"),
        ("--loss-normalization", "sequence"),
        ("--mode", "cot"),
    ):
        assert option(rl, flag) == value


def test_continuation_has_no_fictitious_sft_and_releases_actual_parent(tmp_path):
    specs, _ = build(
        tmp_path,
        "--skip-sft",
        "--cap",
        "8",
        "--inherited-initial-cap",
        "6",
        "--scratch-style",
        "reverse",
        "--modes",
        "cot",
    )
    spec = specs[0]
    assert spec["stages"][0]["name"] == "rl"
    assert not any(row.get("name", "").startswith("sft") for row in spec["stages"])
    assert option(stage(spec, "rl")["command"], "--checkpoint") == "/models/cot/final.pt"
    assert option(stage(spec, "parent_book")["command"], "--checkpoint") == "/models/cot/final.pt"
    release = next(row for row in spec["stages"] if row["kind"] == "release_development")
    assert "/models/cot/final.pt" in release["checkpoints"]
    assert len(release["checkpoints"]) == 2
    assert not any(Path(path).parent.name.endswith("_sft") for path in release["checkpoints"])
    metadata = spec["experiment_metadata"]
    assert metadata["sft_performed"] is False
    assert metadata["inherited_initial_proof_cap"] == 6
    assert metadata["current_supervised_proof_cap"] == 8
    assert metadata["scratch_style"] == "reverse"
