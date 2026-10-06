"""Accept FP32 probability roundoff without altering outcomes or other validity guards."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

FOLDER = (
    Path(__file__).resolve().parents[1]
    / "experiments/current/learned_compute_allocation_20261006/allocation"
)


def load_statistics(monkeypatch, filename):
    monkeypatch.syspath_prepend(str(FOLDER))
    spec = importlib.util.spec_from_file_location(
        "test_" + filename.removesuffix(".py"), FOLDER / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome(correct=0.7, attempts=2.0):
    work = dict.fromkeys(
        (
            "generated_tokens",
            "forward_tokens",
            "controller_forward_tokens",
            "forward_calls",
            "controller_forward_calls",
            "attention_score_cells_per_layer",
            "controller_attention_score_cells_per_layer",
            "policy_wall_seconds",
        ),
        0.0,
    )
    return {
        "correct_probability": correct,
        "attempts": attempts,
        "reward": correct - 0.03 * (attempts - 1),
        "forced_termination_probability": 0.1,
        "expected_policy_work": work,
    }


def test_fp32_normalization_roundoff_is_retained_without_clipping(monkeypatch):
    original = load_statistics(monkeypatch, "aggregate_live.py")
    successor = load_statistics(monkeypatch, "aggregate_live_fp32_v2.py")
    # Summing FP32 categorical probabilities in FP64 can expose their rounded sum.
    correct = float(np.float32(0.50000006)) + float(np.float32(0.5))
    row = outcome(correct)
    with pytest.raises(ValueError, match="invalid integrated"):
        original.metrics(row, cost=0.03)
    values = successor.metrics(row, cost=0.03)
    assert values[successor.METRICS.index("correct_probability")] == correct
    assert values[successor.METRICS.index("correct_probability")] > 1
    assert values[successor.METRICS.index("reward_native")] == row["reward"]


@pytest.mark.parametrize("correct", [0.0, 0.2, 0.8, 1.0])
def test_originally_admitted_metrics_remain_bitwise_identical(monkeypatch, correct):
    original = load_statistics(monkeypatch, "aggregate_live.py")
    successor = load_statistics(monkeypatch, "aggregate_live_fp32_v2.py")
    row = outcome(correct)
    assert np.array_equal(original.metrics(row, cost=0.03), successor.metrics(row, cost=0.03))


@pytest.mark.parametrize(
    "invalid",
    ["large_probability", "negative_probability", "nan", "attempts", "forced", "work", "reward"],
)
def test_successor_still_rejects_invalid_integrated_rows(monkeypatch, invalid):
    successor = load_statistics(monkeypatch, "aggregate_live_fp32_v2.py")
    row = outcome()
    if invalid in ("large_probability", "negative_probability", "nan"):
        row = outcome(
            {"large_probability": 1 + 2e-6, "negative_probability": -2e-6, "nan": np.nan}[invalid]
        )
    elif invalid == "attempts":
        row = outcome(attempts=8 + 2e-8)
    elif invalid == "forced":
        row["forced_termination_probability"] = 1 + 2e-8
    elif invalid == "work":
        row["expected_policy_work"]["forward_calls"] = -1
    else:
        row["reward"] = 0.0
    with pytest.raises(ValueError, match="invalid integrated"):
        successor.metrics(row, cost=0.03)
