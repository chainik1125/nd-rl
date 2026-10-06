"""Registered, paired theorem-level analysis of completed held-out live paths."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analysis_addendum import registered_statistics
from evaluation_protocol import (
    fingerprint,
    signature,
    validate_saved_implementation,
    validate_test_freeze,
)

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 72006
CONFIDENCE = 0.95
COSTS = (0.0, 0.01, 0.03, 0.1)
PRIMARY_ARMS = ("no_cost", "cost")
BASELINES = (
    "fixed1",
    "fixed2",
    "fixed4",
    "fixed8",
    "counter",
    "context_threshold",
    "attempt_matched_mixture",
    "forward_token_matched_mixture",
    "joint_matched_mixture",
)
METRICS = (
    "correct_probability",
    "attempts",
    "reward_native",
    "forced_termination_probability",
    "generated_tokens",
    "generator_forward_tokens",
    "controller_forward_tokens",
    "total_forward_tokens",
    "generator_forward_calls",
    "controller_forward_calls",
    "total_forward_calls",
    "generator_attention_cells_per_layer",
    "controller_attention_cells_per_layer",
    "total_attention_cells_per_layer",
    "model_section_wall_seconds",
    *[f"utility_at_cost_{cost}" for cost in COSTS],
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(row, cost=0.0, max_attempts=8):
    work = row["expected_policy_work"]
    correct, attempts = row["correct_probability"], row["attempts"]
    values = {
        "correct_probability": correct,
        "attempts": attempts,
        "reward_native": row["reward"],
        "forced_termination_probability": row["forced_termination_probability"],
        "generated_tokens": work["generated_tokens"],
        "generator_forward_tokens": work["forward_tokens"],
        "controller_forward_tokens": work["controller_forward_tokens"],
        "total_forward_tokens": work["forward_tokens"] + work["controller_forward_tokens"],
        "generator_forward_calls": work["forward_calls"],
        "controller_forward_calls": work["controller_forward_calls"],
        "total_forward_calls": work["forward_calls"] + work["controller_forward_calls"],
        "generator_attention_cells_per_layer": work["attention_score_cells_per_layer"],
        "controller_attention_cells_per_layer": work["controller_attention_score_cells_per_layer"],
        "total_attention_cells_per_layer": work["attention_score_cells_per_layer"]
        + work["controller_attention_score_cells_per_layer"],
        "model_section_wall_seconds": work["policy_wall_seconds"],
        **{f"utility_at_cost_{cost}": correct - cost * (attempts - 1) for cost in COSTS},
    }
    result = np.array([values[key] for key in METRICS], dtype=np.float64)
    if (
        not np.isfinite(result).all()
        or not -1e-6 <= correct <= 1 + 1e-6
        or not 1 - 1e-8 <= attempts <= max_attempts + 1e-8
        or not -1e-8 <= row["forced_termination_probability"] <= 1 + 1e-8
        or any(value < -1e-8 for value in work.values())
        or not np.isclose(row["reward"], correct - cost * (attempts - 1), rtol=1e-7, atol=1e-8)
    ):
        raise ValueError("invalid integrated outcome or model work")
    return result


def draw_means(rows, theorem_ids, repeats, cost=0.0, max_attempts=8):
    lookup = {}
    for row in rows:
        key = row["theorem_id"], row["repeat"]
        if key in lookup:
            raise ValueError("duplicate theorem/draw result")
        lookup[key] = metrics(row, cost, max_attempts)
    expected = {(theorem_id, repeat) for theorem_id in theorem_ids for repeat in range(repeats)}
    if lookup.keys() != expected:
        raise ValueError("missing, unexpected, or mismatched theorem/draw results")
    return np.stack(
        [
            np.mean([lookup[theorem_id, repeat] for repeat in range(repeats)], axis=0)
            for theorem_id in theorem_ids
        ]
    )


def arm_mean(seed_tables):
    common = set.intersection(*(set(table) for table in seed_tables))
    return {
        method: np.mean([table[method] for table in seed_tables], axis=0)
        for method in sorted(common)
    }


def bootstrap_weights(families, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    families = np.asarray(families)
    rng = np.random.default_rng(seed)
    weights = np.zeros((samples, len(families)), dtype=np.float64)
    for family in sorted(set(families)):
        positions = np.flatnonzero(families == family)
        counts = rng.multinomial(
            len(positions), np.full(len(positions), 1 / len(positions)), size=samples
        )
        weights[:, positions] = counts / len(families)
    return weights


def estimate(values, weights, confidence=CONFIDENCE, names=METRICS):
    draws = weights @ values
    low, high = np.quantile(draws, ((1 - confidence) / 2, 1 - (1 - confidence) / 2), axis=0)
    mean = values.mean(axis=0)
    return {
        key: {"mean": float(mean[i]), "ci_low": float(low[i]), "ci_high": float(high[i])}
        for i, key in enumerate(names)
    }


def validate_statistics(freeze, families_file, addendum_file, registry_file):
    requested = {
        "script_sha256": sha(__file__),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "family_data_sha256": sha(families_file),
    }
    if any(
        registered_statistics(addendum_file, "main", registry_file, __file__).get(key) != value
        for key, value in requested.items()
    ):
        raise ValueError("statistical code, settings, or family data are not frozen")
    if freeze.get("test_repeats") != 2 or freeze.get("test_seed") != 62006:
        raise ValueError("primary final analysis requires the prespecified two test draws")
    return requested


def load_run(run, freeze, freeze_hash, theorem_ids):
    folder = Path(run["paths"])
    config = json.loads((folder / "config.json").read_text())
    validate_saved_implementation(config)
    if (
        config.get("evaluation_split") != "test"
        or config.get("frozen_protocol_sha256") != freeze_hash
    ):
        raise ValueError("analysis requires test paths from the identical frozen registry")
    current_signature = signature(config)
    validate_test_freeze(
        freeze,
        current_signature,
        config["data_sha256"],
        config["seed"],
        config["repeats"],
        config["evaluated_theorems"],
    )
    entries = [
        entry
        for entry in freeze.get("registered_models", [])
        if entry.get("checkpoint_sha256") == config["checkpoint_sha256"]
        and entry.get("base_sha256") == config["base_sha256"]
        and entry.get("inference_sha256") == fingerprint(current_signature)
    ]
    if not any(
        entry.get("arm") == run["arm"] and entry.get("rl_seed") == run["seed"] for entry in entries
    ):
        raise ValueError(
            "registered model arm or training seed does not match the analysis manifest"
        )
    if run["arm"] in PRIMARY_ARMS:
        expected_cost = 0.0 if run["arm"] == "no_cost" else 0.03
        if config["cost"] != expected_cost:
            raise ValueError("primary arm uses an unexpected native reward coefficient")
    # Guards above run before integrated private outcomes are opened.
    integrated = json.loads((folder / "integrated.json").read_text())
    provenance = {
        "config_sha256": sha(folder / "config.json"),
        "integrated_sha256": sha(folder / "integrated.json"),
    }
    feasible_mixtures = set()
    for field, artifact_key, code_name in (
        ("counter", "counter_hazards_sha256", "reanalyze_counter.py"),
        ("budgets", "budget_baselines_sha256", "reanalyze_budget_baselines.py"),
    ):
        if field not in run:
            if run["arm"] in PRIMARY_ARMS:
                raise ValueError("primary model is missing a prespecified stopping baseline")
            continue
        derived = json.loads(Path(run[field]).read_text())
        prov = derived["provenance"]
        artifact_hash = prov["hazards_sha256"] if field == "counter" else prov["fit_sha256"]
        if not any(entry.get(artifact_key) == artifact_hash for entry in entries):
            raise ValueError("derived DEV-fitted baseline artifact was not registered")
        if prov["implementation_sha256"] != freeze.get("postprocessing_hashes", {}).get(code_name):
            raise ValueError("derived baseline uses unregistered postprocessing code")
        if prov["paths_sha256"] != sha(folder / "public_paths.jsonl") or prov[
            "labels_sha256"
        ] != sha(folder / "private_labels.jsonl"):
            raise ValueError("derived baseline does not belong to this exact evaluated path set")
        if prov["evaluation_sha256"] != config["runner_hashes"]["evaluate_live_path.py"]:
            raise ValueError("derived baseline uses a different integration implementation")
        if any(
            integrated[name] != rows
            for name, rows in derived["integrated"].items()
            if name in integrated
        ):
            raise ValueError("duplicate readout differs between primary and derived results")
        integrated.update(derived["integrated"])
        if field == "budgets":
            feasible_mixtures = {
                name
                for name in BASELINES
                if name.endswith("_mixture") and derived["fit"][name]["feasible"]
            }
        provenance[field + "_sha256"] = sha(run[field])
    if run["arm"] in PRIMARY_ARMS:
        required = {
            "learned",
            "fixed1",
            "fixed2",
            "fixed4",
            "fixed8",
            "counter",
            "context_threshold",
            "learned_greedy_index",
            "fixed1_greedy_index",
            "fixed2_greedy_index",
            "fixed4_greedy_index",
            "fixed8_greedy_index",
            "counter_greedy_index",
            "greedy_stop_greedy_index",
        }
        if not required | feasible_mixtures <= integrated.keys():
            raise ValueError("primary seed is missing a prespecified feasible comparison")
    tables = {
        method: draw_means(rows, theorem_ids, 2, config["cost"], config["max_attempts"])
        for method, rows in integrated.items()
    }
    labels = [
        json.loads(line) for line in (folder / "private_labels.jsonl").read_text().splitlines()
    ]
    availability = availability_means(labels, theorem_ids, 2)
    provenance["private_labels_sha256"] = sha(folder / "private_labels.jsonl")
    return tables, availability, provenance


def availability_means(labels, theorem_ids, repeats):
    lookup = {}
    for row in labels:
        key = row["theorem_id"], row["repeat"]
        validities = row["validities"]
        if (
            key in lookup
            or not validities
            or any(not isinstance(value, bool) for value in validities)
        ):
            raise ValueError("invalid or duplicate private path labels")
        lookup[key] = np.array([float(any(validities[:k])) for k in (1, 2, 4, 8)])
    expected = {(theorem_id, repeat) for theorem_id in theorem_ids for repeat in range(repeats)}
    if lookup.keys() != expected:
        raise ValueError("private labels do not match the registered theorem/draw set")
    return np.stack(
        [
            np.mean([lookup[theorem_id, repeat] for repeat in range(repeats)], axis=0)
            for theorem_id in theorem_ids
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", required=True, help="JSON manifest listing arm,seed,paths,counter,budgets"
    )
    parser.add_argument("--families", required=True)
    parser.add_argument("--frozen-protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis-addendum", required=True)
    args = parser.parse_args()
    freeze_file = Path(args.frozen_protocol)
    freeze = json.loads(freeze_file.read_text())
    settings = validate_statistics(freeze, args.families, args.analysis_addendum, freeze_file)
    families_rows = [json.loads(line) for line in Path(args.families).read_text().splitlines()]
    families = {row["theorem_id"]: row["family"] for row in families_rows}
    if len(families) != len(families_rows) or len(families) != freeze["test_theorems"]:
        raise ValueError("family metadata does not match the full registered theorem set")
    theorem_ids = sorted(families)
    manifest = json.loads(Path(args.runs).read_text())["runs"]
    keys = [(run["arm"], run["seed"]) for run in manifest]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate arm/training seed")
    for arm in PRIMARY_ARMS:
        if {seed for condition, seed in keys if condition == arm} != {101, 102, 103}:
            raise ValueError("primary analysis requires all three prespecified training seeds")
    tables, availability, input_provenance = {}, {}, {}
    for run in manifest:
        key = run["arm"], run["seed"]
        tables[key], availability[key], input_provenance[f"{key[0]}:{key[1]}"] = load_run(
            run, freeze, sha(freeze_file), theorem_ids
        )
    weights = bootstrap_weights([families[theorem_id] for theorem_id in theorem_ids])
    per_seed = {
        f"{arm}:{seed}": {
            method: dict(zip(METRICS, values.mean(axis=0).tolist(), strict=True))
            for method, values in methods.items()
        }
        for (arm, seed), methods in tables.items()
    }
    arms = {
        arm: arm_mean([methods for (condition, _), methods in tables.items() if condition == arm])
        for arm in sorted({key[0] for key in keys})
    }
    arm_results = {
        arm: {method: estimate(values, weights) for method, values in methods.items()}
        for arm, methods in arms.items()
    }
    contrasts = {}
    for arm in PRIMARY_ARMS:
        for baseline in BASELINES:
            if baseline in arms[arm]:
                contrasts[f"{arm}:learned-minus-{baseline}"] = estimate(
                    arms[arm]["learned"] - arms[arm][baseline], weights
                )
        for method, baseline in (
            ("learned_greedy_index", "fixed1_greedy_index"),
            ("learned_greedy_index", "counter_greedy_index"),
            ("greedy_stop_greedy_index", "fixed1_greedy_index"),
        ):
            if method in arms[arm] and baseline in arms[arm]:
                contrasts[f"secondary:{arm}:{method}-minus-{baseline}"] = estimate(
                    arms[arm][method] - arms[arm][baseline], weights
                )
    cross = estimate(arms["cost"]["learned"] - arms["no_cost"]["learned"], weights)
    cross.pop("reward_native")  # Native coefficients differ; use common-lambda utilities.
    contrasts["cost-minus-no_cost"] = cross
    for method in ("learned_greedy_index", "greedy_stop_greedy_index"):
        if method in arms["cost"] and method in arms["no_cost"]:
            value = estimate(arms["cost"][method] - arms["no_cost"][method], weights)
            value.pop("reward_native")
            contrasts[f"secondary:cost-minus-no_cost:{method}"] = value
    if "format_only" in arms:
        for arm in PRIMARY_ARMS:
            contrasts[f"secondary:{arm}:fixed1-minus-format_only-fixed1"] = estimate(
                arms[arm]["fixed1"] - arms["format_only"]["fixed1"], weights
            )
    for control in ("fixed1_control", "fixed1"):
        if control in arms:
            for arm in PRIMARY_ARMS:
                contrasts[f"secondary:{arm}:fixed1-minus-{control}-fixed1"] = estimate(
                    arms[arm]["fixed1"] - arms[control]["fixed1"], weights
                )
    for control in ("fixed4_control", "fixed4"):
        if control in arms:
            for arm in PRIMARY_ARMS:
                value = estimate(arms[arm]["learned"] - arms[control]["fixed4"], weights)
                value.pop("reward_native")
                contrasts[f"secondary:{arm}:learned-minus-{control}-fixed4"] = value
    result = {
        "provenance": {
            "frozen_protocol_sha256": sha(freeze_file),
            "runs_manifest_sha256": sha(args.runs),
            "analysis_addendum_sha256": sha(args.analysis_addendum),
            "analysis_change": (
                "correct_probability-domain tolerance only: 1e-8 to 1e-6; estimates unchanged"
            ),
            "statistics": settings,
            "inputs": input_provenance,
        },
        "aggregation_rule": "mean two draws within theorem, then mean seeds within arm",
        "uncertainty_scope": (
            "paired family-stratified theorem bootstrap conditional on the fitted models; "
            "does not estimate uncertainty over new training seeds"
        ),
        "model_wall_scope": (
            "sampled generator and controller sections; excludes parser/prompt/checker/setup "
            "and diagnostics"
        ),
        "per_seed": per_seed,
        "training_seed_standard_deviation": {
            arm: {
                method: dict(
                    zip(
                        METRICS,
                        np.std(
                            [
                                methods[method].mean(axis=0)
                                for (condition, _), methods in tables.items()
                                if condition == arm
                            ],
                            axis=0,
                            ddof=1,
                        ).tolist(),
                        strict=True,
                    )
                )
                for method in arms[arm]
            }
            for arm in PRIMARY_ARMS
        },
        "arms": arm_results,
        "contrasts": contrasts,
        "theorem_ids": theorem_ids,
        "per_theorem": {
            arm: {method: values.tolist() for method, values in methods.items()}
            for arm, methods in arms.items()
        },
        "metric_order": METRICS,
        "finite_path_oracle_availability": {
            arm: estimate(
                np.mean(
                    [values for (condition, _), values in availability.items() if condition == arm],
                    axis=0,
                ),
                weights,
                names=("after1", "after2", "after4", "after8"),
            )
            for arm in arms
        },
        "oracle_scope": (
            "retained correct candidate on this finite forced path; "
            "not a deployable selector or a guarantee about other paths"
        ),
        "control_primary_readout": {
            "format_only": "fixed1",
            "fixed1_control": "fixed1",
            "fixed4_control": "fixed4",
            "fixed1": "fixed1",
            "fixed4": "fixed4",
        },
        "untrained_head_diagnostics": ["format_only"],
        "single_seed_control_scope": (
            "control intervals describe theorem variation conditional on one trained control; "
            "they do not establish stability across control training seeds"
        ),
        "missing_shared_methods": {
            arm: sorted(
                set.union(
                    *(
                        set(methods)
                        for (condition, _), methods in tables.items()
                        if condition == arm
                    )
                )
                - set(arms[arm])
            )
            for arm in arms
        },
    }
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(result, stream)
    with out.with_suffix(".analysis.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
