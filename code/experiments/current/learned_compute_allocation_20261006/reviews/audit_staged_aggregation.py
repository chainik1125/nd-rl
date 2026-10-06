"""Independent reconstruction of staged DEV aggregation and paired intervals."""

import json
from pathlib import Path

import numpy as np

B = Path("artifacts/learned_compute_allocation_20261006/analysis")


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.open()]


inv = read(B / "staged_dev_inventory.json")
saved = read(B / "staged_dev_aggregate_v1/metrics.json")
rs = rows(B / "staged_dev_aggregate_v1/per_theorem.jsonl")
errors = []
counts = {}
sources = {}
for run in inv["runs"]:
    d = Path(run["evaluation"])
    policy = read(Path(run["policy"]))
    src = {(r["theorem_id"], r["policy"]): r for r in rows(d / "per_theorem.jsonl")}
    sources[run["group"], run["training_seed"]] = (policy, src)
for r in rs:
    p, src = sources[r["group"], r["seed"]]
    choice = p["choices"][r["cost"]]
    kind = r["kind"]
    name = (
        choice["threshold"]
        if kind == "adaptive"
        else choice["fixed"]
        if kind == "dev_best_fixed"
        else choice["matched_mixtures"][kind[len("matched_") :]]
        if kind.startswith("matched_")
        else kind
    )
    original = src[r["theorem_id"], name]
    for k in ["accuracy", "attempts", "total_model_input_tokens_lower_bound"]:
        errors.append(abs(r[k] - original[k]))
        assert np.isfinite(r[k])
    errors.append(abs(r["reward"] - (r["accuracy"] - float(r["cost"]) * (r["attempts"] - 1))))
for group, bycost in saved["groups"].items():
    for cost, report in bycost.items():
        subset = [r for r in rs if r["group"] == group and r["cost"] == cost]
        ids = list(dict.fromkeys(r["theorem_id"] for r in subset))
        families = [next(r["family"] for r in subset if r["theorem_id"] == tid) for tid in ids]
        vectors = {}
        for kind, metrics in report["policies"].items():
            rr = [r for r in subset if r["kind"] == kind]
            expected = 3 if group == "bce" else 1
            assert all(sum(r["theorem_id"] == tid for r in rr) == expected for tid in ids)
            vectors[kind] = {
                m: np.array([np.mean([r[m] for r in rr if r["theorem_id"] == tid]) for tid in ids])
                for m in metrics
            }
            for m, v in vectors[kind].items():
                errors.append(abs(v.mean() - metrics[m]))
        for comp, metrics in report["comparisons"].items():
            for m, result in metrics.items():
                diff = vectors["adaptive"][m] - vectors[comp][m]
                rng = np.random.default_rng(61006)
                boot = np.zeros(5000)
                for family in sorted(set(families)):
                    v = diff[np.array(families) == family]
                    draws = rng.integers(len(v), size=(5000, len(v)))
                    boot += v[draws].sum(axis=1) / len(ids)
                low, high = np.quantile(boot, [0.025, 0.975])
                errors.extend(
                    [
                        abs(result["mean"] - diff.mean()),
                        abs(result["low"] - low),
                        abs(result["high"] - high),
                    ]
                )
        counts[group, cost] = len(ids)
out = {
    "rows_checked": len(rs),
    "max_discrepancy": max(errors),
    "theorems_each_group_cost": {"/".join(k): v for k, v in counts.items()},
    "cost03": {
        g: v["0.03"]["comparisons"]["matched_total_model_input_tokens_lower_bound"]
        for g, v in saved["groups"].items()
    },
}
Path(__file__).with_name("staged_aggregation_independent_audit.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out, indent=2))
