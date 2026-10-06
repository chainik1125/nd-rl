"""DEV-only reproducer of FP32 probability roundoff versus frozen analysis guard."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"
sys.path.insert(0, str(ALLOC))
spec = importlib.util.spec_from_file_location(
    "original_live_aggregate_review", ALLOC / "aggregate_live.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
registry = json.loads((ART / "final_evaluation/main_protocol.json").read_text())
failures = []
count = 0
max_error = 0.0
examples = []
for model in registry["registered_models"]:
    if model["arm"] not in ("cost", "no_cost"):
        continue
    directory = Path(model["dev_evaluation_path"])
    config = json.loads((directory / "config.json").read_text())
    assert config["evaluation_split"] == "dev"
    records = json.loads((directory / "integrated.json").read_text())
    for method, rows in records.items():
        for row in rows:
            count += 1
            probability = row["correct_probability"]
            max_error = max(max_error, probability - 1, -probability)
            try:
                module.metrics(row, model["native_cost"])
            except ValueError:
                assert 1 + 1e-8 < probability <= 1 + 1e-6
                assert 1 - 1e-8 <= row["attempts"] <= 8 + 1e-8
                assert -1e-8 <= row["forced_termination_probability"] <= 1 + 1e-8
                assert all(
                    np.isfinite(v) and v >= -1e-8 for v in row["expected_policy_work"].values()
                )
                assert np.isclose(
                    row["reward"],
                    probability - model["native_cost"] * (row["attempts"] - 1),
                    rtol=1e-7,
                    atol=1e-8,
                )
                failures.append(
                    (model["arm"], model["rl_seed"], method, row["theorem_id"], probability)
                )
                if len(examples) < 5:
                    examples.append({"dev_path": str(directory), "method": method, "row": row})
result = {
    "original_aggregate_sha256": hashlib.sha256(
        (ALLOC / "aggregate_live.py").read_bytes()
    ).hexdigest(),
    "dev_rows": count,
    "rejected_rows": len(failures),
    "max_probability_excursion": max_error,
    "all_rejections_exclusively_probability_upper_bound": True,
    "proposed_tolerance": 1e-6,
    "failures": failures,
    "minimal_examples": examples,
    "live_test_outcomes_read": False,
}
Path(__file__).with_name("live_numeric_guard_dev_reproducer.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("failures", "minimal_examples")}, indent=2
    )
)
