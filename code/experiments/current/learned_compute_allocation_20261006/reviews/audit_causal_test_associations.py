"""Independent descriptive association audit of authorized causal TEST outputs."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
inventory = json.loads((ART / "freeze_inputs/selection_v2/causal_test_inventory.json").read_text())


def statistics(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    xx = xy = yy = 0.0
    for group in groups.values():
        a = np.array([r["accept_probability"] for r in group])
        d = np.array([r["delta_continuation"] for r in group])
        a = a - a.mean()
        d = d - d.mean()
        xx += float(a @ a)
        xy += float(a @ d)
        yy += float(d @ d)
    return {
        "slope": xy / xx,
        "correlation": xy / np.sqrt(xx * yy),
        "acceptance_residual_sd": np.sqrt(xx / len(rows)),
    }


def draws(families, rng):
    families = np.array(families)
    groups = [np.flatnonzero(families == f) for f in sorted(set(families))]
    return np.concatenate([g[rng.integers(len(g), size=(5000, len(g)))] for g in groups], axis=1)


results = []
max_error = 0.0
for entry in inventory["runs"]:
    directory = Path(entry["analysis"])
    rows = [json.loads(line) for line in (directory / "per_state.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["mode"] == "actual_history"]
    summary = json.loads((directory / "summary.json").read_text())
    by_theorem = defaultdict(list)
    for row in rows:
        by_theorem[row["theorem_id"]].append(row)
    ids = sorted(by_theorem)
    families = [by_theorem[tid][0]["family"] for tid in ids]
    rng = np.random.default_rng(81006)
    for _ in range(12):
        draws(families, rng)
    output = {"condition": entry["condition"], "seed": entry["training_seed"], "associations": {}}
    for name, fields in [
        ("unconditional", []),
        ("within_count", ["count"]),
        ("within_count_and_incumbent_validity", ["count", "oracle_before"]),
    ]:
        calculated = statistics(rows, fields)
        expected = summary["acceptance_value_association"][name]
        for field, value in calculated.items():
            max_error = max(max_error, abs(value - expected[field]))
        if entry["condition"] == "one_extra64":
            samples = []
            for draw in draws(families, rng):
                sample = [r for i in draw for r in by_theorem[ids[i]]]
                value = statistics(sample, fields)
                samples.append([value["slope"], value["correlation"]])
            intervals = np.quantile(samples, [0.025, 0.975], axis=0)
            for j, field in enumerate(["slope", "correlation"]):
                target = expected[field + "_interval"]
                max_error = max(
                    max_error,
                    abs(intervals[0, j] - target["low"]),
                    abs(intervals[1, j] - target["high"]),
                )
        output["associations"][name] = expected
    results.append(output)
assert max_error < 1e-12, max_error
report = {
    "all_eight_point_estimates_checked": True,
    "primary_two_seed_intervals_checked": True,
    "max_error": max_error,
    "results": results,
}
Path(__file__).with_name("causal_test_associations_independent_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
