"""Separate descriptive readout of the reused72-theorem historical benchmark."""

import argparse
import json
from collections import Counter
from pathlib import Path

import aggregate_live as primary
import numpy as np

HISTORICAL_SEED = 92006
BOOTSTRAP_SEED = 102006


def validate_registration(freeze, main_freeze_file, roles_file, exposure_file):
    requested = {
        "script_sha256": primary.sha(__file__),
        "aggregate_live_dependency_sha256": primary.sha(primary.__file__),
        "bootstrap_samples": 10000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
        "family_data_sha256": primary.sha(roles_file),
    }
    if any(freeze.get("statistical_analysis", {}).get(k) != v for k, v in requested.items()):
        raise ValueError("Historical statistical code/settings/roles are not registered")
    if (freeze.get("test_seed"), freeze.get("test_repeats"), freeze.get("test_theorems")) != (
        HISTORICAL_SEED,
        2,
        72,
    ):
        raise ValueError("Historical draw protocol differs from the prespecified72x2")
    if freeze.get("main_frozen_protocol_sha256") != primary.sha(main_freeze_file):
        raise ValueError("Historical registry does not bind the original MAIN registry")
    if freeze.get("historical_exposure_audit_sha256") != primary.sha(exposure_file):
        raise ValueError("Historical exposure audit is not registered")
    main = json.loads(Path(main_freeze_file).read_text())
    if main.get("status") != "frozen" or main.get("selection_split") != "dev":
        raise ValueError("MAIN model choices were not frozen on development")
    fields = (
        "arm",
        "rl_seed",
        "base_sha256",
        "checkpoint_sha256",
        "inference_sha256",
        "counter_hazards_sha256",
        "budget_baselines_sha256",
    )
    expected = {tuple(entry.get(k) for k in fields) for entry in main["registered_models"]}
    observed = {tuple(entry.get(k) for k in fields) for entry in freeze["registered_models"]}
    if expected != observed or len(observed) != 9:
        raise ValueError("Historical models or DEV-fitted policies differ from MAIN choices")
    return requested


def summarize(tables, availability, theorem_ids, mask, roles):
    selected_ids = [value for value, selected in zip(theorem_ids, mask, strict=True) if selected]
    selected_roles = [role for role, selected in zip(roles, mask, strict=True) if selected]
    weights = primary.bootstrap_weights(selected_roles, samples=10000, seed=BOOTSTRAP_SEED)
    subset = {
        key: {method: values[mask] for method, values in methods.items()}
        for key, methods in tables.items()
    }
    arms = {
        arm: primary.arm_mean(
            [methods for (condition, _), methods in subset.items() if condition == arm]
        )
        for arm in sorted({key[0] for key in subset})
    }
    contrasts = {}
    for arm in primary.PRIMARY_ARMS:
        for baseline in primary.BASELINES:
            if baseline in arms[arm]:
                contrasts[f"{arm}:learned-minus-{baseline}"] = primary.estimate(
                    arms[arm]["learned"] - arms[arm][baseline], weights
                )
    cross = primary.estimate(arms["cost"]["learned"] - arms["no_cost"]["learned"], weights)
    cross.pop("reward_native")
    contrasts["cost-minus-no_cost"] = cross
    for arm in primary.PRIMARY_ARMS:
        contrasts[f"secondary:{arm}:fixed1-minus-format_only-fixed1"] = primary.estimate(
            arms[arm]["fixed1"] - arms["format_only"]["fixed1"], weights
        )
        for control in ("fixed4", "fixed4_control"):
            if control in arms:
                value = primary.estimate(arms[arm]["learned"] - arms[control]["fixed4"], weights)
                value.pop("reward_native")
                contrasts[f"secondary:{arm}:learned-minus-{control}-fixed4"] = value
    return {
        "theorems": len(selected_ids),
        "theorem_ids": selected_ids,
        "per_seed": {
            f"{arm}:{seed}": {
                method: dict(zip(primary.METRICS, values.mean(axis=0).tolist(), strict=True))
                for method, values in methods.items()
            }
            for (arm, seed), methods in subset.items()
        },
        "arms": {
            arm: {method: primary.estimate(values, weights) for method, values in methods.items()}
            for arm, methods in arms.items()
        },
        "contrasts": contrasts,
        "finite_path_oracle_availability": {
            arm: primary.estimate(
                np.mean(
                    [
                        values[mask]
                        for (condition, _), values in availability.items()
                        if condition == arm
                    ],
                    axis=0,
                ),
                weights,
                names=("after1", "after2", "after4", "after8"),
            )
            for arm in arms
        },
    }


def main():
    parser = argparse.ArgumentParser()
    for argument in (
        "runs",
        "roles",
        "frozen-protocol",
        "main-frozen-protocol",
        "exposure-audit",
        "out",
    ):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    freeze = json.loads(Path(args.frozen_protocol).read_text())
    settings = validate_registration(
        freeze, args.main_frozen_protocol, args.roles, args.exposure_audit
    )
    metadata = [json.loads(line) for line in Path(args.roles).read_text().splitlines()]
    roles = {row["theorem_id"]: row["source_role"] for row in metadata}
    if (
        len(metadata) != 72
        or len(roles) != 72
        or Counter(roles.values()) != {"textbook_dev": 58, "rl_train": 14}
        or any(row["family"] != row["source_role"] or "reference_lines" in row for row in metadata)
    ):
        raise ValueError(
            "Historical metadata does not preserve the72 source roles without reference lengths"
        )
    theorem_ids = sorted(roles)
    runs = json.loads(Path(args.runs).read_text())["runs"]
    keys = [(run["arm"], run["seed"]) for run in runs]
    if len(keys) != 9 or len(set(keys)) != 9:
        raise ValueError("Historical report requires exactly nine distinct models")
    for arm in primary.PRIMARY_ARMS:
        if {seed for condition, seed in keys if condition == arm} != {101, 102, 103}:
            raise ValueError("Historical report requires all three primary seeds")
    if (
        ("format_only", None) not in keys
        or not any((arm, 101) in keys for arm in ("fixed1", "fixed1_control"))
        or not any((arm, 101) in keys for arm in ("fixed4", "fixed4_control"))
    ):
        raise ValueError("Historical report is missing a prespecified secondary control")
    tables, availability, provenance = {}, {}, {}
    for run in runs:
        key = run["arm"], run["seed"]
        tables[key], availability[key], provenance[f"{key[0]}:{key[1]}"] = primary.load_run(
            run, freeze, primary.sha(args.frozen_protocol), theorem_ids
        )
    role_order = np.array([roles[value] for value in theorem_ids])
    result = {
        "status": "descriptive_historical_retention_and_transfer",
        "scope": (
            "Reused historical benchmark, at least one known canonical pretraining exposure; "
            "not fresh generalization. Source rl_train role does not establish "
            "actual r16 training exposure. "
            "Same bounded MAIN caps, not upstream k256/max_action512/max_steps96."
        ),
        "uncertainty_scope": (
            "Paired source-role-stratified theorem bootstrap conditional on fitted models; "
            "descriptive benchmark variation, not new-training-seed uncertainty."
        ),
        "provenance": {
            "frozen_protocol_sha256": primary.sha(args.frozen_protocol),
            "main_frozen_protocol_sha256": primary.sha(args.main_frozen_protocol),
            "runs_manifest_sha256": primary.sha(args.runs),
            "exposure_audit_sha256": primary.sha(args.exposure_audit),
            "statistics": settings,
            "inputs": provenance,
        },
        "combined": summarize(
            tables, availability, theorem_ids, np.ones(72, dtype=bool), role_order
        ),
        "by_source_role": {
            role: summarize(tables, availability, theorem_ids, role_order == role, role_order)
            for role in ("textbook_dev", "rl_train")
        },
    }
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(result, stream)
    with out.with_suffix(".analysis.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
