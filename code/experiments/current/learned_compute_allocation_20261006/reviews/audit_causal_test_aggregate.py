"""Independently reconstruct authorized TEST aggregate from saved per-state and branch records."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
CONTEXT = ROOT / "artifacts/learned_compute_allocation_20261006/context"
INVENTORY_PATH = CONTEXT.parent / "freeze_inputs/selection_v2/causal_test_inventory.json"
INVENTORY = json.loads(INVENTORY_PATH.read_text())
METRICS = json.loads((CONTEXT / "causal_main_test_aggregate_v1/metrics.json").read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


panel = read_rows(INVENTORY["public"])
assert {r["split"] for r in panel} == {"test"}
common_ids = {r["theorem_id"] for r in panel[:16]}
mappings = defaultdict(dict)
canonical_states = {}
max_branch_error = 0.0
branch_count = 0
common_comparisons = 0
state_count = 0
raw_probability_error = 0.0
associations = {}
registry_path = CONTEXT.parent / "final_evaluation/main_protocol.json"
registry = json.loads(registry_path.read_text())
assert METRICS["signature"] in [r["signature"] for r in registry["registered_analyses"]]
assert (
    hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest()
    == METRICS["signature"]["inventory_sha256"]
)
for entry in INVENTORY["runs"]:
    directory = Path(entry["run"])
    states = read_rows(directory / "states_public.jsonl")
    attempts = {
        r["attempt_uid"]: r["candidate"] for r in read_rows(directory / "attempts_public.jsonl")
    }
    branches = read_rows(directory / "branches_scored.jsonl")
    state_count += len(states)
    settings = entry["intervention_signature"]["protocol"]
    expected_ids = [r["theorem_id"] for r in panel[: settings["limit"]]]
    assert {(r["theorem_id"], r["count"]) for r in states} == set(
        (tid, count) for tid in expected_ids for count in (1, 2)
    )
    state_lookup = {r["state_id"]: r for r in states}
    validity = json.loads((directory / "validity.json").read_text())
    launch = json.loads(directory.with_suffix(".launch.json").read_text())
    assert launch["status"] == "complete" and launch["returncode"] == 0
    assert launch["registry_sha256"] == hashlib.sha256(registry_path.read_bytes()).hexdigest()
    grouped = defaultdict(list)
    for row in branches:
        branch_count += 1
        incumbent = state_lookup[row["state_id"]]
        old_ids = incumbent["incumbent_ids"]
        new_ids = row["future_ids"]
        before = sum(
            p * validity[uid]
            for p, uid in zip(incumbent["selection_probabilities"], old_ids, strict=True)
        )
        probabilities = row["after"]["selection_probabilities"]
        old_mass = sum(
            p * validity[uid] for p, uid in zip(probabilities[: len(old_ids)], old_ids, strict=True)
        )
        new_mass = sum(
            p * validity[uid] for p, uid in zip(probabilities[len(old_ids) :], new_ids, strict=True)
        )
        for actual, expected in [
            (row["accept_value"], before),
            (row["old_correct_mass"], old_mass),
            (row["new_correct_mass"], new_mass),
        ]:
            raw_probability_error = max(raw_probability_error, abs(actual - expected))
        assert row["new_attempt_count"] == len(new_ids) == len(row["decision_path"])
        assert row["count"] + len(new_ids) <= 8
        assert row["new_attempt_valid"] == validity[new_ids[0]]
        assert row["any_new_valid"] == any(validity[uid] for uid in new_ids)
        assert row["oracle_before"] == any(validity[uid] for uid in old_ids)
        assert row["oracle_after"] == any(validity[uid] for uid in old_ids + new_ids)
        assert not row["oracle_before"] or row["oracle_after"]
        if len(new_ids) < row["forced_extra_requested"]:
            assert row["after"]["forced"]
        grouped[row["state_id"], row["mode"]].append(row)
        assert (
            abs(row["continue_value"] - row["old_correct_mass"] - row["new_correct_mass"]) < 1e-12
        )
        assert abs(row["delta_continuation"] - row["continue_value"] + row["accept_value"]) < 1e-12
    for state in states:
        assert not state["forced"]
        if state["theorem_id"] not in common_ids:
            continue
        key = entry["training_seed"], state["state_id"]
        fields = [
            "candidate_id",
            "canonical_name_base",
            "raw_actions",
            "effective_actions",
            "transition_tokens",
            "generation_action_seeds",
        ]
        canonical = (
            state,
            [{k: attempts[uid][k] for k in fields} for uid in state["incumbent_ids"]],
        )
        if key in canonical_states:
            assert canonical_states[key] == canonical
            common_comparisons += 1
        canonical_states[key] = canonical
    for records in grouped.values():
        assert sorted(r["repeat"] for r in records) == list(range(settings["replicates"]))
    for row in read_rows(Path(entry["analysis"]) / "per_state.jsonl"):
        records = grouped[row["state_id"], row["mode"]]
        for field in (
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
        ):
            error = abs(np.mean([b[field] for b in records]) - row[field])
            max_branch_error = max(max_branch_error, error)
        row["old_mass_change"] = row["old_correct_mass"] - row["accept_value"]
        mappings[entry["condition"]][
            entry["training_seed"], row["theorem_id"], row["count"], row["mode"]
        ] = row

max_mean_error = 0.0
max_ci_error = 0.0
checks = 0


def verify(item, condition, mode="actual_history", contrast=False, seeds=(101, 102), counts=(1, 2)):
    global max_mean_error, max_ci_error, checks
    assert item["available"]
    table = item["per_theorem"]
    fields = list(item["metrics"])
    values = []
    for theorem in table:
        row = []
        for field in fields:
            samples = []
            for seed in seeds:
                for count in counts:
                    key = seed, theorem["theorem_id"], count, mode
                    value = mappings[condition][key][field]
                    if contrast:
                        value -= mappings[condition][
                            seed, theorem["theorem_id"], count, "actual_history"
                        ][field]
                    samples.append(value)
            value = float(np.mean(samples))
            assert abs(value - theorem[field]) < 1e-12
            row.append(value)
        values.append(row)
    values = np.asarray(values)
    families = np.asarray([r["family"] for r in table])
    groups = [np.flatnonzero(families == f) for f in sorted(set(families))]
    rng = np.random.default_rng(81006)
    resamples = []
    for _ in range(5000):
        chosen = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        resamples.append(values[chosen].mean(axis=0))
    intervals = np.quantile(resamples, [0.025, 0.975], axis=0)
    for j, field in enumerate(fields):
        expected = item["metrics"][field]
        max_mean_error = max(max_mean_error, abs(values[:, j].mean() - expected["mean"]))
        max_ci_error = max(max_ci_error, float(np.max(np.abs(intervals[:, j] - expected["ci95"]))))
        checks += 1


for condition, item in METRICS["conditions"].items():
    verify(item["fixed_weight"], condition)
    verify(METRICS["common_first16"][condition], condition)
    for count, item_count in item["by_count"].items():
        verify(item_count, condition, counts=(int(count),))
    for seed, item_seed in item["by_seed"].items():
        verify(item_seed["fixed_weight"], condition, seeds=(int(seed),))
    if "history_removed_minus_actual" in item:
        verify(item["history_removed_minus_actual"], condition, "history_removed", True)
assert max(max_branch_error, max_mean_error, max_ci_error, raw_probability_error) < 1e-12
result = {
    "runs": len(INVENTORY["runs"]),
    "branches_checked": branch_count,
    "states_checked": state_count,
    "raw_probability_error": raw_probability_error,
    "common_state_exact_comparisons": common_comparisons,
    "aggregate_metric_checks": checks,
    "max_branch_to_state_error": max_branch_error,
    "max_mean_error": max_mean_error,
    "max_ci_error": max_ci_error,
    "primary": METRICS["primary"]["metrics"],
    "common16_delta": {
        k: v["metrics"]["delta_continuation"] for k, v in METRICS["common_first16"].items()
    },
}
result["conditions"] = {
    condition: {
        "fixed": item["fixed_weight"]["metrics"],
        "by_count": {k: v["metrics"] for k, v in item["by_count"].items()},
        "by_seed": {k: v["fixed_weight"]["metrics"] for k, v in item["by_seed"].items()},
        "history_removed_minus_actual": item.get("history_removed_minus_actual", {}).get("metrics"),
    }
    for condition, item in METRICS["conditions"].items()
}
output = Path(__file__).with_name("causal_test_aggregate_independent_audit.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("primary", "conditions")}, indent=2))
