"""Theorem-cluster analysis of saved continuation interventions."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "code/shared_code"))
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402

FIELDS = [
    "accept_value",
    "continue_value",
    "delta_continuation",
    "old_correct_mass",
    "new_correct_mass",
    "new_attempt_valid",
    "any_new_valid",
    "new_attempt_count",
    "oracle_before",
    "oracle_after",
    "forced_count_shortfall",
    "terminal_hard_limit",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_signature(intervention, bootstrap=5000, seed=81006):
    return {
        "method": "causal_intervention_analysis",
        "intervention_signature": intervention,
        "analysis_sha256": digest(__file__),
        "validator_sha256": digest(ROOT / "code/shared_code/evaluation_freeze.py"),
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": seed,
        "confidence": 0.95,
        "resampling": (
            "whole theorem within family; future mean within state, state mean within theorem"
        ),
    }


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def aggregate_states(branches):
    groups = defaultdict(list)
    for branch in branches:
        branch = dict(branch)
        if "delta_continuation" not in branch:
            branch["delta_continuation"] = branch["delta_one"]
        branch.setdefault("any_new_valid", branch["new_attempt_valid"])
        branch.setdefault("new_attempt_count", 1)
        branch["forced_count_shortfall"] = float(
            branch["new_attempt_count"] < branch.get("forced_extra_requested", 1)
        )
        branch["terminal_hard_limit"] = float(branch["after"]["forced"])
        groups[(branch["state_id"], branch["mode"])].append(branch)
    result = []
    for (state_id, mode), rows in sorted(groups.items()):
        if len({row["repeat"] for row in rows}) != len(rows):
            raise ValueError("Duplicate continuation replicate")
        first = rows[0]
        for row in rows:
            if any(not math.isfinite(row[field]) for field in FIELDS):
                raise ValueError("Nonfinite causal measurement")
            bounded = [
                key for key in FIELDS if key not in {"delta_continuation", "new_attempt_count"}
            ]
            if any(not -1e-6 <= row[key] <= 1 + 1e-6 for key in bounded):
                raise ValueError("Causal probability outside its support")
            if any(
                row[key] != first[key]
                for key in ["theorem_id", "family", "count", "accept_probability"]
            ):
                raise ValueError("State metadata differs across continuation replicates")
            if (
                abs(row["continue_value"] - row["old_correct_mass"] - row["new_correct_mass"])
                > 1e-6
            ):
                raise ValueError("Continuation decomposition is inconsistent")
            if abs(row["delta_continuation"] - row["continue_value"] + row["accept_value"]) > 1e-6:
                raise ValueError("Continuation contrast is inconsistent")
            if row["oracle_before"] and not row["oracle_after"]:
                raise ValueError("Retained-candidate oracle availability cannot decrease")
        item = {key: first[key] for key in ["theorem_id", "family", "count", "accept_probability"]}
        item.update(state_id=state_id, mode=mode, replicates=len(rows))
        item.update({key: float(np.mean([row[key] for row in rows])) for key in FIELDS})
        item["delta_mc_standard_error"] = (
            float(np.std([row["delta_continuation"] for row in rows], ddof=1) / np.sqrt(len(rows)))
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
    families = {row["theorem_id"]: row["family"] for row in rows}
    draws = stratified_indices([families[tid] for tid in sorted(groups)], rng, repetitions)
    means = values[draws].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "low": float(lower),
        "high": float(upper),
        "theorems": len(values),
        "states": len(rows),
    }


def stratified_indices(families, rng, repetitions):
    families = np.asarray(families)
    groups = []
    for family in sorted(set(families)):
        indices = np.flatnonzero(families == family)
        groups.append(indices[rng.integers(len(indices), size=(repetitions, len(indices)))])
    return np.concatenate(groups, axis=1)


def within_stratum_statistics(rows, conditioning):
    """Descriptive association; no observed future outcome defines its strata."""
    strata = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in conditioning)
        strata[key].append(row)
    numerator = denominator = y_squares = 0.0
    for subset in strata.values():
        x = np.array([r["accept_probability"] for r in subset])
        y = np.array([r["delta_continuation"] for r in subset])
        x -= x.mean()
        y -= y.mean()
        numerator += float(x @ y)
        denominator += float(x @ x)
        y_squares += float(y @ y)
    x_sd = float(np.sqrt(denominator / len(rows)))
    return {
        "slope": numerator / denominator if x_sd > 1e-6 else None,
        "correlation": numerator / np.sqrt(denominator * y_squares)
        if x_sd > 1e-6 and y_squares > 1e-12
        else None,
        "acceptance_residual_sd": x_sd,
    }


def association_summary(rows, rng, repetitions, conditioning):
    point = within_stratum_statistics(rows, conditioning)
    point["conditioning"] = conditioning
    point["acceptance_min"] = min(r["accept_probability"] for r in rows)
    point["acceptance_max"] = max(r["accept_probability"] for r in rows)
    if point["slope"] is None:
        return point | {"reason": "acceptance variation below 1e-6 within the specified strata"}
    groups = defaultdict(list)
    for row in rows:
        groups[row["theorem_id"]].append(row)
    ids = sorted(groups)
    estimates = {"slope": [], "correlation": []}
    families = [groups[tid][0]["family"] for tid in ids]
    for draw in stratified_indices(families, rng, repetitions):
        sample = [row for index in draw for row in groups[ids[index]]]
        statistics = within_stratum_statistics(sample, conditioning)
        for field in estimates:
            if statistics[field] is not None:
                estimates[field].append(statistics[field])
    for field, values in estimates.items():
        if values:
            lower, upper = np.quantile(values, [0.025, 0.975])
            point[f"{field}_interval"] = {
                "low": float(lower),
                "high": float(upper),
                "usable_bootstrap_draws": len(values),
            }
    point["interpretation"] = (
        "Descriptive association between acceptance probability and continuation value"
    )
    return point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=81006)
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    public = load_rows(config["public"])
    splits = {row["split"] for row in public}
    if len(splits) != 1 or next(iter(splits)) not in {"dev", "test"}:
        raise ValueError("Public intervention panel has an invalid split")
    split = next(iter(splits))
    if config.get("split", "dev") != split:
        raise ValueError("Saved split differs from public intervention panel")
    if digest(config["public"]) != config["public_sha256"]:
        raise ValueError("Public intervention panel changed")
    public = public[: config["limit"]]
    public_map = {row["theorem_id"]: row for row in public}
    if len(public_map) != len(public) or [r["theorem_id"] for r in public] != config["theorem_ids"]:
        raise ValueError("Saved theorem panel differs from public input")
    registered = config.get("registered_signature")
    if args.signature_only:
        if not registered:
            raise ValueError("Legacy smoke has no registered intervention signature")
        print(json.dumps(analysis_signature(registered, args.bootstrap, args.seed), indent=2))
        return
    if split == "test":
        if not args.frozen_test_protocol or not registered:
            raise ValueError("Test causal analysis requires frozen registration")
        if registered["analysis_sha256"] != digest(__file__):
            raise ValueError("Analysis differs from registered intervention")
        if config["frozen_test_protocol_sha256"] != digest(args.frozen_test_protocol):
            raise ValueError("Generation and analysis use different test registrations")
        if any(config[key] != value for key, value in registered["protocol"].items()):
            raise ValueError("Generation arguments differ from registered intervention")
        for key in ["base_sha256", "checkpoint_sha256", "public_sha256"]:
            if config[key] != registered[key]:
                raise ValueError("Generation inputs differ from registered intervention")
        if config["upstream_hashes"] != registered["source_hashes"]:
            raise ValueError("Generation upstream sources differ from registration")
        for name, recorded in registered["runner_hashes"].items():
            if digest(run / "runner_snapshot" / name) != recorded:
                raise ValueError("Generation helper snapshot differs from registration")
        if digest(run / "runner_snapshot/causal_probe.py") != registered["implementation_sha256"]:
            raise ValueError("Generation implementation differs from registration")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()),
            config["public"],
            analysis_signature(registered, args.bootstrap, args.seed),
        )
    if not (run / "completion.json").is_file():
        raise ValueError("Intervention run did not finish")
    branches = load_rows(run / "branches_scored.jsonl")
    completion = json.loads((run / "completion.json").read_text())
    if completion["branches"] != len(branches):
        raise ValueError("Saved branch count differs from completion record")
    collected = load_rows(run / "states_public.jsonl")
    if completion["states"] != len(collected) or completion["forced_states"] != sum(
        r["forced"] for r in collected
    ):
        raise ValueError("Collected state counts differ from completion")
    collected_keys = [(row["theorem_id"], row["count"]) for row in collected]
    if len(collected_keys) != len(set(collected_keys)):
        raise ValueError("Duplicate collected intervention state")
    state_map = {}
    for row in collected:
        tid, count = row["theorem_id"], row["count"]
        if tid not in public_map or count not in config["counts"]:
            raise ValueError("Collected state is outside planned theorem/count panel")
        if row["family"] != public_map[tid]["family"] or row["state_id"] != f"{tid}/state/{count}":
            raise ValueError("Collected state identity/family differs from public theorem")
        if not math.isfinite(row["accept_probability"]) or not 0 <= row["accept_probability"] <= 1:
            raise ValueError("Invalid collected acceptance probability")
        state_map[row["state_id"]] = row
    repeats = defaultdict(set)
    for row in branches:
        if row["state_id"] not in state_map:
            raise ValueError("Branch refers to an uncollected state")
        state = state_map[row["state_id"]]
        if state["forced"] or any(
            row[key] != state[key]
            for key in ["theorem_id", "count", "family", "accept_probability"]
        ):
            raise ValueError("Branch metadata differs from its authoritative state")
        count = row.get("new_attempt_count", 1)
        if type(count) is not int or not 1 <= count <= config["max_attempts"] - row["count"]:
            raise ValueError("Invalid number of future attempts")
        if row.get("forced_extra_requested", 1) != config.get("forced_extra", 1):
            raise ValueError("Branch forced-attempt request differs from protocol")
        if row.get("continuation", "one_extra") != config.get("continuation", "one_extra"):
            raise ValueError("Branch continuation mode differs from protocol")
        if "future_ids" in row and len(row["future_ids"]) != count:
            raise ValueError("Future count differs from generated candidate identities")
        if "decision_path" in row and len(row["decision_path"]) != count:
            raise ValueError("Future count differs from controller decision path")
        if count < config.get("forced_extra", 1) and not row["after"]["forced"]:
            raise ValueError("Forced continuation stopped early without a hard limit")
        repeats[row["state_id"], row["mode"]].add(row["repeat"])
    if any(values != set(range(config["replicates"])) for values in repeats.values()):
        raise ValueError("Continuation replicate IDs differ from protocol")
    states = aggregate_states(branches)
    eligible = {(row["theorem_id"], row["count"]) for row in collected if not row["forced"]}
    actual_keys = {
        (row["theorem_id"], row["count"]) for row in states if row["mode"] == "actual_history"
    }
    if eligible != actual_keys:
        raise ValueError("Eligible state missing its actual-history continuation")
    history_ids = set(config["history_removal_theorem_ids"])
    expected_modes = {(tid, count, "actual_history") for tid, count in eligible}
    expected_modes.update(
        (tid, count, "history_removed") for tid, count in eligible if tid in history_ids
    )
    if {(row["theorem_id"], row["count"], row["mode"]) for row in states} != expected_modes:
        raise ValueError("Continuation modes differ from the prespecified state panel")
    eligibility = {}
    for count in config["counts"]:
        reached = [row for row in collected if row["count"] == count]
        eligibility[str(count)] = {
            "planned_theorems": len(config["theorem_ids"]),
            "reached_states": len(reached),
            "forced_at_state": sum(row["forced"] for row in reached),
            "eligible_theorem_ids": [row["theorem_id"] for row in reached if not row["forced"]],
        }
    if not states:
        raise ValueError("No nonforced continuation states were collected")
    actual = [row for row in states if row["mode"] == "actual_history"]
    if any(row["replicates"] != config["replicates"] for row in states):
        raise ValueError("Incomplete continuation branches")
    rng = np.random.default_rng(args.seed)
    summary = {
        "estimand": config["estimand"],
        "uncertainty": (
            "percentile bootstrap of whole theorems within family; "
            "state/branch means remain together"
        ),
        "scope": "conditions on checkpoint, sampled intervention panel and development choices",
        "registered_analysis": analysis_signature(registered, args.bootstrap, args.seed)
        if registered
        else None,
        "state_collection": {
            key: completion[key] for key in ["states", "forced_states", "branches"]
        },
        "eligibility_by_count": eligibility,
        "complete_prespecified_state_panel": all(
            len(item["eligible_theorem_ids"]) == item["planned_theorems"]
            for item in eligibility.values()
        ),
        "missing_state_rule": (
            "Pooled descriptive summaries use available eligible states; the fixed-weight primary "
            "two-checkpoint estimand is unavailable if any planned state is ineligible "
            "or unreachable."
        ),
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "confidence": 0.95},
        "pooled_actual_history": {
            field: theorem_summary(actual, field, rng, args.bootstrap) for field in FIELDS
        },
        "by_attempt_count": {},
        "by_count_and_oracle": {},
        "acceptance_value_association": {
            name: association_summary(actual, rng, args.bootstrap, fields)
            for name, fields in {
                "unconditional": [],
                "within_count": ["count"],
                "within_count_and_incumbent_validity": ["count", "oracle_before"],
            }.items()
        },
        "new_attempt_valid_interpretation": "Validity of the first forced future attempt",
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
                for field in ["accept_value", "delta_continuation", "new_attempt_valid"]
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
                "family": row["family"],
                **{
                    field: row[field] - before[field]
                    for field in [
                        "delta_continuation",
                        "new_attempt_valid",
                        "old_correct_mass",
                        "new_correct_mass",
                    ]
                },
            }
        )
    summary["history_removed_minus_actual"] = {
        field: theorem_summary(paired, field, rng, args.bootstrap)
        for field in [
            "delta_continuation",
            "new_attempt_valid",
            "old_correct_mass",
            "new_correct_mass",
        ]
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
                "delta_continuation": summary["pooled_actual_history"]["delta_continuation"],
                "association": summary["acceptance_value_association"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
