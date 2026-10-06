"""Fixed causal weights and unavailable-state behavior."""

import importlib.util
import sys
from pathlib import Path

import pytest

FOLDER = (
    Path(__file__).resolve().parents[1]
    / "experiments/current/learned_compute_allocation_20261006/context"
)
sys.path.insert(0, str(FOLDER))
SPEC = importlib.util.spec_from_file_location(
    "causal_aggregate_test", FOLDER / "aggregate_causal.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def panel_and_mapping():
    panel = [
        {"theorem_id": f"t{i}", "family": "forward" if i % 2 == 0 else "ore2"} for i in range(64)
    ]
    mapping = {}
    for row in panel:
        for s in MOD.SEEDS:
            for n in (1, 2):
                values = {field: 0.0 for field in MOD.FIELDS}
                values["delta_continuation"] = (s - 101) * 0.2 + n * 0.1
                mapping[s, row["theorem_id"], n, "actual_history"] = values
    return panel, mapping


def test_average_states_then_checkpoints_within_theorem():
    panel, mapping = panel_and_mapping()
    result = MOD.aggregate({condition: mapping for condition in MOD.CONDITIONS}, panel, 20, 81006)
    assert result["primary"]["metrics"]["delta_continuation"]["mean"] == pytest.approx(0.25)
    assert result["conditions"]["one_extra64"]["by_count"]["1"]["metrics"]["delta_continuation"][
        "mean"
    ] == pytest.approx(0.2)


def test_missing_single_model_state_disables_primary_without_reweighting():
    panel, mapping = panel_and_mapping()
    del mapping[102, "t1", 2, "actual_history"]
    result = MOD.aggregate({"one_extra64": mapping}, panel, 20, 81006)
    assert not result["primary"]["available"]
    assert result["primary"]["metrics"] is None
    per_seed = result["conditions"]["one_extra64"]["by_seed"]
    assert per_seed["101"]["fixed_weight"]["available"]
    assert not per_seed["102"]["fixed_weight"]["available"]
    assert per_seed["102"]["eligible_both_counts_descriptive"]["planned_theorems"] == 63
    assert not result["common_first16"]["one_extra64"]["available"]


def test_history_contrast_is_paired_before_aggregation():
    panel, mapping = panel_and_mapping()
    for key, value in list(mapping.items()):
        if int(key[1][1:]) < 16:
            altered = dict(value)
            altered["delta_continuation"] += 0.4
            mapping[key[:3] + ("history_removed",)] = altered
    result = MOD.aggregate({"one_extra64": mapping}, panel, 20, 81006)
    paired = result["conditions"]["one_extra64"]["history_removed_minus_actual"]
    assert paired["metrics"]["delta_continuation"]["mean"] == pytest.approx(0.4)
