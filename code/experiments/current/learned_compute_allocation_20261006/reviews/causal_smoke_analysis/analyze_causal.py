"""Theorem-cluster analysis of saved one-step continuation interventions."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = [
    "accept_value",
    "continue_value",
    "delta_one",
    "old_correct_mass",
    "new_correct_mass",
    "new_attempt_valid",
    "oracle_before",
    "oracle_after",
]


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def aggregate_states(branches):
    groups = defaultdict(list)
    for branch in branches:
        groups[(branch["state_id"], branch["mode"])].append(branch)
    result = []
    for (state_id, mode), rows in sorted(groups.items()):
        if len({row["repeat"] for row in rows}) != len(rows):
            raise ValueError("Duplicate continuation replicate")
        first = rows[0]
        for row in rows:
            if (
                abs(row["continue_value"] - row["old_correct_mass"] - row["new_correct_mass"])
                > 1e-6
            ):
                raise ValueError("Continuation decomposition is inconsistent")
            if abs(row["delta_one"] - row["continue_value"] + row["accept_value"]) > 1e-6:
                raise ValueError("Continuation contrast is inconsistent")
            if row["oracle_before"] and not row["oracle_after"]:
                raise ValueError("Retained-candidate oracle availability cannot decrease")
        item = {key: first[key] for key in ["theorem_id", "family", "count", "accept_probability"]}
        item.update(state_id=state_id, mode=mode, replicates=len(rows))
        item.update({key: float(np.mean([row[key] for row in rows])) for key in FIELDS})
        item["delta_mc_standard_error"] = (
            float(np.std([row["delta_one"] for row in rows], ddof=1) / np.sqrt(len(rows)))
            if len(rows) > 1
            else None
        )
        result.append(item)
    return result


def theorem_summary(rows, field, rng, repetitions):
    groups = defaultdict(list)
    for row in rows:
        groups[row["theorem_id"]].append(row[field])
    values = np.array([np.mean(groups[tid]) for tid in sorted(groups)])
    if not len(values):
        return None
    draws = rng.integers(0, len(values), (repetitions, len(values)))
    means = values[draws].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "low": float(lower),
        "high": float(upper),
        "theorems": len(values),
        "states": len(rows),
    }


def within_stratum_slope(rows):
    """Descriptive association, controlling for count and any-valid incumbent."""
    strata = defaultdict(list)
    for row in rows:
        strata[(row["count"], bool(row["oracle_before"]))].append(row)
    numerator = denominator = 0.0
    for subset in strata.values():
        x = np.array([r["accept_probability"] for r in subset])
        y = np.array([r["delta_one"] for r in subset])
        x -= x.mean()
        y -= y.mean()
        numerator += float(x @ y)
        denominator += float(x @ x)
    return numerator / denominator if denominator > 1e-12 else None


def association_summary(rows, rng, repetitions):
    point = within_stratum_slope(rows)
    if point is None:
        return {"slope": None, "reason": "no within-stratum acceptance-probability variation"}
    groups = defaultdict(list)
    for row in rows:
        groups[row["theorem_id"]].append(row)
    ids = sorted(groups)
    estimates = []
    for draw in rng.integers(0, len(ids), (repetitions, len(ids))):
        sample = [row for index in draw for row in groups[ids[index]]]
        slope = within_stratum_slope(sample)
        if slope is not None:
            estimates.append(slope)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "slope": point,
        "low": float(lower),
        "high": float(upper),
        "usable_bootstrap_draws": len(estimates),
        "interpretation": "delta_one per unit acceptance probability, within count/oracle strata",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=81006)
    args = parser.parse_args()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    if not (run / "completion.json").is_file():
        raise ValueError("Intervention run did not finish")
    branches = load_rows(run / "branches_scored.jsonl")
    states = aggregate_states(branches)
    if not states:
        raise ValueError("No nonforced continuation states were collected")
    actual = [row for row in states if row["mode"] == "actual_history"]
    if any(row["replicates"] != config["replicates"] for row in states):
        raise ValueError("Incomplete continuation branches")
    rng = np.random.default_rng(args.seed)
    summary = {
        "estimand": config["estimand"],
        "uncertainty": "percentile bootstrap of whole theorems; state/branch means remain together",
        "scope": "conditions on checkpoint, sampled intervention panel and development choices",
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "confidence": 0.95},
        "pooled_actual_history": {
            field: theorem_summary(actual, field, rng, args.bootstrap) for field in FIELDS
        },
        "by_attempt_count": {},
        "by_count_and_oracle": {},
        "acceptance_value_association": association_summary(actual, rng, args.bootstrap),
    }
    for count in sorted({row["count"] for row in actual}):
        subset = [row for row in actual if row["count"] == count]
        summary["by_attempt_count"][str(count)] = {
            field: theorem_summary(subset, field, rng, args.bootstrap) for field in FIELDS
        }
        for available in [False, True]:
            group = [r for r in subset if bool(r["oracle_before"]) == available]
            summary["by_count_and_oracle"][f"n{count}_oracle{int(available)}"] = {
                field: theorem_summary(group, field, rng, args.bootstrap)
                for field in ["accept_value", "delta_one", "new_attempt_valid"]
            }
    actual_by_id = {row["state_id"]: row for row in actual}
    paired = []
    for row in states:
        if row["mode"] != "history_removed":
            continue
        before = actual_by_id[row["state_id"]]
        paired.append(
            {
                "theorem_id": row["theorem_id"],
                "state_id": row["state_id"],
                **{
                    field: row[field] - before[field]
                    for field in [
                        "delta_one",
                        "new_attempt_valid",
                        "old_correct_mass",
                        "new_correct_mass",
                    ]
                },
            }
        )
    summary["history_removed_minus_actual"] = {
        field: theorem_summary(paired, field, rng, args.bootstrap)
        for field in ["delta_one", "new_attempt_valid", "old_correct_mass", "new_correct_mass"]
    }
    summary["inputs"] = {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for name in ["config.json", "completion.json", "branches_scored.jsonl"]
    }
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "analyze_causal.py").write_bytes(Path(__file__).read_bytes())
    with (out / "per_state.jsonl").open("w") as stream:
        for row in states:
            stream.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "states": len(actual),
                "delta_one": summary["pooled_actual_history"]["delta_one"],
                "association": summary["acceptance_value_association"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
