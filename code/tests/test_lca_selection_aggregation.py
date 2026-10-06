"""Theorem weighting, family resampling and ID joins in frozen selection reports."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/current/learned_compute_allocation_20261006/selection/aggregate_selection.py"
)
spec = importlib.util.spec_from_file_location("selection_aggregation_test", SCRIPT)
aggregation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregation)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_stratified_bootstrap_retains_fixed_family_mass():
    result = aggregation.stratified_interval([1, 1, 0], ["forward", "forward", "ore2"], 100, 7)
    assert result["mean"] == pytest.approx(2 / 3)
    assert result["low"] == result["high"] == pytest.approx(2 / 3)


def test_oracle_gap_and_unavailable_pools_are_retained():
    result = aggregation.endpoint([1, 0], [0.125, 0], [1, 0])
    assert result["accuracy"] == 0.5
    assert result["conditional_on_available_accuracy"] == 1
    assert result["oracle_gap_recovery"] == 1
    no_valid = aggregation.endpoint([0, 0], [0, 0], [0, 0])
    assert no_valid["conditional_on_available_accuracy"] is None
    assert no_valid["oracle_gap_recovery"] is None


def test_seed_average_and_private_candidate_id_join(tmp_path, monkeypatch):
    public, private, feature = [
        tmp_path / name for name in ["public.jsonl", "private.jsonl", "feature.jsonl"]
    ]
    write_rows(
        public,
        [
            {"theorem_id": tid, "split": "dev", "family": family}
            for tid, family in [("a", "forward"), ("b", "ore2")]
        ],
    )
    write_rows(
        private,
        [
            {
                "theorem_id": tid,
                "labels": [
                    {"candidate_id": i, "valid": tid == "a" and i == 3, "parse_complete": True}
                    for i in reversed(range(8))
                ],
            }
            for tid in ["a", "b"]
        ],
    )
    write_rows(
        feature,
        [
            {"theorem_id": tid, "accuracy": {"learned_features": int(tid == "a")}}
            for tid in ["a", "b"]
        ],
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("immutable checkpoint identity, not loaded by statistical analysis")
    runs = []
    for group in aggregation.GROUPS:
        for seed, ties in enumerate([[3], [4], [3, 4]]):
            predictions = tmp_path / f"{group}{seed}.json"
            predictions.write_text(
                json.dumps(
                    [
                        {
                            "theorem_id": tid,
                            "argmax_tied_candidate_ids": ties,
                            "argmax_tie_expected_accuracy": sum(tid == "a" and i == 3 for i in ties)
                            / len(ties),
                            "candidate_order": list(range(8)),
                            "overflow": False,
                            "max_forward_tokens": 100,
                        }
                        for tid in ["a", "b"]
                    ]
                )
            )
            runs.append(
                {
                    "group": group,
                    "training_seed": seed,
                    "permutation_seed": 77106,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "predictions": str(predictions),
                }
            )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"runs": runs}))
    out = tmp_path / "out"
    monkeypatch.setattr(aggregation, "figure", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate",
            "--inventory",
            str(inventory),
            "--public",
            str(public),
            "--private",
            str(private),
            "--features",
            str(feature),
            "--split",
            "dev",
            "--out",
            str(out),
            "--bootstrap",
            "20",
        ],
    )
    aggregation.main()
    result = json.loads((out / "metrics.json").read_text())
    for group in aggregation.GROUPS:
        assert result["groups"][group]["primary"]["accuracy"] == 0.25
        assert result["groups"][group]["primary"]["conditional_on_available_accuracy"] == 0.5
        assert [r["accuracy"] for r in result["groups"][group]["seed_values"]] == [0.5, 0, 0.25]


def test_invalid_final_protocol_rejects_before_private_read(tmp_path, monkeypatch):
    public = tmp_path / "public.jsonl"
    write_rows(public, [{"theorem_id": "x", "split": "test"}])
    inventory = tmp_path / "inventory.json"
    inventory.write_text('{"runs": []}')
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"status": "draft"}')
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate",
            "--inventory",
            str(inventory),
            "--public",
            str(public),
            "--private",
            str(tmp_path / "NEVER_READ"),
            "--features",
            str(tmp_path / "NEVER_READ_EITHER"),
            "--split",
            "test",
            "--out",
            str(tmp_path / "out"),
            "--frozen-test-protocol",
            str(protocol),
        ],
    )
    with pytest.raises(ValueError):
        aggregation.main()
