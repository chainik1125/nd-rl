"""Fixed-panel, two-checkpoint aggregation of prespecified causal interventions."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "code/shared_code"))
import analyze_causal as helper  # noqa: E402
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402

CONDITIONS = {
    "one_extra64": (64, 8, "one_extra", 1, 16),
    "policy_forced1_32": (32, 4, "policy", 1, 16),
    "policy_forced2_16": (16, 4, "policy", 2, 0),
    "policy_forced4_16": (16, 4, "policy", 4, 0),
}
SEEDS = (101, 102)
FIELDS = helper.FIELDS + ["old_mass_change"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def signature(inventory_path, bootstrap=5000, seed=81006):
    return {
        "method": "fixed_panel_two_checkpoint_causal_aggregate",
        "inventory_sha256": sha(inventory_path),
        "public_sha256": sha(json.loads(Path(inventory_path).read_text())["public"]),
        "aggregate_sha256": sha(__file__),
        "analyzer_sha256": sha(Path(helper.__file__)),
        "validator_sha256": sha(ROOT / "code/shared_code/evaluation_freeze.py"),
        "conditions": CONDITIONS,
        "training_seeds": list(SEEDS),
        "bootstrap": bootstrap,
        "seed": seed,
        "resampling": "paired whole theorem within family; fixed checkpoints and n1/n2",
        "missing_rule": "fixed-weight estimate unavailable if any planned state missing",
    }


def interval(values, families, bootstrap=5000, seed=81006):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    sample = []
    groups = [np.flatnonzero(np.asarray(families) == family) for family in sorted(set(families))]
    for _ in range(bootstrap):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        sample.append(float(values[indices].mean()))
    return {"mean": float(values.mean()), "ci95": np.quantile(sample, [0.025, 0.975]).tolist()}


def describe(
    mapping,
    ids,
    families,
    counts=(1, 2),
    seeds=SEEDS,
    mode="actual_history",
    bootstrap=5000,
    seed=81006,
    contrast=False,
):
    missing = [
        (s, tid, n, mode)
        for tid in ids
        for s in seeds
        for n in counts
        if (s, tid, n, mode) not in mapping
    ]
    if contrast:
        missing += [
            (s, tid, n, "actual_history")
            for tid in ids
            for s in seeds
            for n in counts
            if (s, tid, n, "actual_history") not in mapping
        ]
    if missing or not ids:
        return {
            "available": False,
            "planned_theorems": len(ids),
            "missing_states": [list(k) for k in missing],
            "metrics": None,
        }
    metrics, per_theorem = {}, []
    for tid in ids:
        result = {"theorem_id": tid, "family": families[tid]}
        for field in FIELDS:
            values = []
            for s in seeds:
                for n in counts:
                    value = mapping[s, tid, n, mode][field]
                    if contrast:
                        value -= mapping[s, tid, n, "actual_history"][field]
                    values.append(value)
            result[field] = float(np.mean(values))
        per_theorem.append(result)
    for field in FIELDS:
        metrics[field] = interval(
            [r[field] for r in per_theorem], [r["family"] for r in per_theorem], bootstrap, seed
        )
    return {
        "available": True,
        "planned_theorems": len(ids),
        "metrics": metrics,
        "per_theorem": per_theorem,
    }


def aggregate(mapping_by_condition, panel, bootstrap=5000, seed=81006):
    families = {r["theorem_id"]: r["family"] for r in panel}
    result = {"conditions": {}, "common_first16": {}}
    for condition, (limit, _, _, _, history_limit) in CONDITIONS.items():
        ids = [r["theorem_id"] for r in panel[:limit]]
        mapping = mapping_by_condition.get(condition, {})
        item = {
            "fixed_weight": describe(mapping, ids, families, bootstrap=bootstrap, seed=seed),
            "by_seed": {},
            "by_count": {},
        }
        for s in SEEDS:
            eligible = [
                tid for tid in ids if all((s, tid, n, "actual_history") in mapping for n in (1, 2))
            ]
            available_state_ids = sorted(
                {k[1] for k in mapping if k[0] == s and k[3] == "actual_history"}
            )
            item["by_seed"][str(s)] = {
                "fixed_weight": describe(
                    mapping, ids, families, seeds=(s,), bootstrap=bootstrap, seed=seed
                ),
                "eligible_both_counts_descriptive": describe(
                    mapping, eligible, families, seeds=(s,), bootstrap=bootstrap, seed=seed
                ),
                "reached_eligible_any_count_theorems": len(available_state_ids),
                "count_specific_descriptive": {},
            }
            for n in (1, 2):
                eligible_n = [tid for tid in ids if (s, tid, n, "actual_history") in mapping]
                item["by_seed"][str(s)]["count_specific_descriptive"][str(n)] = describe(
                    mapping,
                    eligible_n,
                    families,
                    counts=(n,),
                    seeds=(s,),
                    bootstrap=bootstrap,
                    seed=seed,
                )
        for n in (1, 2):
            item["by_count"][str(n)] = describe(
                mapping, ids, families, counts=(n,), bootstrap=bootstrap, seed=seed
            )
        if history_limit:
            item["history_removed_minus_actual"] = describe(
                mapping,
                ids[:history_limit],
                families,
                mode="history_removed",
                contrast=True,
                bootstrap=bootstrap,
                seed=seed,
            )
        result["conditions"][condition] = item
        common_ids = [r["theorem_id"] for r in panel[:16]]
        result["common_first16"][condition] = describe(
            mapping, common_ids, families, bootstrap=bootstrap, seed=seed
        )
    result["primary"] = result["conditions"]["one_extra64"]["fixed_weight"]
    return result


def validate_config(entry, config, panel, generation_seed):
    training = config["training_config"]
    if training["seed"] != entry["training_seed"] or training["cost"] != 0:
        raise ValueError("Seed alias or zero-cost training arm mismatch")
    if training.get("fixed_count") is not None:
        raise ValueError("Expected the learned zero-cost arm")
    if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    saved_training = json.loads((Path(training["out"]) / "config.json").read_text())
    if saved_training != training:
        raise ValueError("Training metadata differs from original saved configuration")
    condition = entry["condition"]
    limit, replicates, continuation, forced_extra, history_limit = CONDITIONS[condition]
    expected = {
        "limit": limit,
        "replicates": replicates,
        "continuation": continuation,
        "forced_extra": forced_extra,
        "history_removal_limit": history_limit,
        "counts": [1, 2],
        "seed": generation_seed,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Run {condition}: mismatched {key}")
    if config["theorem_ids"] != [r["theorem_id"] for r in panel[:limit]]:
        raise ValueError("Run is not the fixed nested theorem panel")
    if config["history_removal_theorem_ids"] != [r["theorem_id"] for r in panel[:history_limit]]:
        raise ValueError("History-removal panel differs")
    registered = entry["intervention_signature"]
    if config["registered_signature"] != registered:
        raise ValueError("Run differs from planned intervention signature")
    for key, value in registered["protocol"].items():
        if config[key] != value:
            raise ValueError("Protocol fields changed")
    for key in ("base_sha256", "checkpoint_sha256", "public_sha256"):
        if config[key] != registered[key]:
            raise ValueError("Run input hash changed")
    if sha(Path(helper.__file__)) != registered["analysis_sha256"]:
        raise ValueError("Analyzer differs from intervention registration")
    if config["upstream_hashes"] != registered["source_hashes"]:
        raise ValueError("Upstream source mismatch")
    run = Path(entry["run"])
    for name, value in registered["runner_hashes"].items():
        if sha(run / "runner_snapshot" / name) != value:
            raise ValueError("Runner snapshot mismatch")
    if sha(run / "runner_snapshot/causal_probe.py") != registered["implementation_sha256"]:
        raise ValueError("Intervention snapshot mismatch")
    if sha(config["public"]) != config["public_sha256"]:
        raise ValueError("Run public panel changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=81006)
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    registered = signature(args.inventory, args.bootstrap, args.seed)
    if args.signature_only:
        print(json.dumps(registered, indent=2))
        return
    inventory = json.loads(Path(args.inventory).read_text())
    panel = rows(inventory["public"])
    if len(panel) != 64 or len({r["theorem_id"] for r in panel}) != 64:
        raise ValueError("Expected 64 unique planned theorems")
    split = {r["split"] for r in panel}
    if split not in ({"dev"}, {"test"}):
        raise ValueError("Invalid split")
    entries = inventory["runs"]
    keys = {(e["condition"], e["training_seed"]) for e in entries}
    if len(entries) != 8 or keys != {(c, s) for c in CONDITIONS for s in SEEDS}:
        raise ValueError("Inventory must retain all eight planned runs")
    checkpoints = inventory["checkpoint_sha256_by_seed"]
    if set(checkpoints) != {str(s) for s in SEEDS} or len(set(checkpoints.values())) != 2:
        raise ValueError("Require two distinct frozen checkpoint identities")
    for entry in entries:
        if (
            entry["intervention_signature"]["checkpoint_sha256"]
            != checkpoints[str(entry["training_seed"])]
        ):
            raise ValueError("Checkpoint identity differs across conditions or seed alias")
    configs = {}
    for entry in entries:
        run = Path(entry["run"])
        if not (run / "config.json").is_file():
            continue  # Planned missing run remains null, never silently reweighted.
        config = json.loads((run / "config.json").read_text())
        validate_config(entry, config, panel, inventory["generation_seed"])
        if config["public_sha256"] != sha(inventory["public"]):
            raise ValueError("Run and aggregate public panel bytes differ")
        configs[entry["condition"], entry["training_seed"]] = config
    if split == {"test"}:
        if not args.frozen_test_protocol:
            raise ValueError("TEST aggregate requires a frozen registry")
        protocol = json.loads(Path(args.frozen_test_protocol).read_text())
        validate_frozen_evaluation(protocol, inventory["public"], registered)
        for entry in entries:
            config = configs.get((entry["condition"], entry["training_seed"]))
            if config is not None:
                original = Path(entry["generation_registry"])
                if sha(original) != config["frozen_test_protocol_sha256"]:
                    raise ValueError("Original generation registry changed")
                old_protocol = json.loads(original.read_text())
                validate_frozen_evaluation(
                    old_protocol,
                    config["public"],
                    helper.analysis_signature(
                        entry["intervention_signature"], args.bootstrap, args.seed
                    ),
                )
    # No per-state outcomes or analyzed summaries are read before all guards above.
    mappings, provenance, starting_states = {}, {}, {}
    for entry in entries:
        key = entry["condition"], entry["training_seed"]
        if key not in configs:
            provenance[str(key)] = {"status": "planned_run_missing"}
            continue
        analysis = Path(entry["analysis"])
        if not (analysis / "summary.json").is_file():
            provenance[str(key)] = {"status": "planned_analysis_missing"}
            continue
        summary = json.loads((analysis / "summary.json").read_text())
        if summary["analysis_sha256"] != sha(Path(helper.__file__)):
            raise ValueError("Analyzed outcomes use a different analyzer")
        if sha(analysis / "analyze_causal.py") != summary["analysis_sha256"]:
            raise ValueError("Analyzer snapshot mismatch")
        expected_signature = helper.analysis_signature(
            entry["intervention_signature"], args.bootstrap, args.seed
        )
        if summary["registered_analysis"] != expected_signature:
            raise ValueError("Analysis settings or intervention differ")
        for name, value in summary["inputs"].items():
            if sha(Path(entry["run"]) / name) != value:
                raise ValueError("Analyzed run input changed")
        per_state = rows(analysis / "per_state.jsonl")
        with tempfile.TemporaryDirectory(prefix="lca_causal_authoritative_") as temporary:
            review = Path(temporary) / "analysis"
            command = [
                sys.executable,
                str(Path(helper.__file__)),
                "--run",
                entry["run"],
                "--out",
                str(review),
                "--bootstrap",
                str(args.bootstrap),
                "--seed",
                str(args.seed),
            ]
            if split == {"test"}:
                command += ["--frozen-test-protocol", entry["generation_registry"]]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise ValueError(
                    "Unchanged authoritative analyzer rejected run: " + completed.stderr
                )
            if rows(review / "per_state.jsonl") != per_state:
                raise ValueError("Per-state rows differ from authoritative analyzer rerun")
            if json.loads((review / "summary.json").read_text()) != summary:
                raise ValueError("Summary differs from authoritative analyzer rerun")
        run = Path(entry["run"])
        attempts = {r["attempt_uid"]: r for r in rows(run / "attempts_public.jsonl")}
        canonical_fields = [
            "candidate_id",
            "canonical_name_base",
            "raw_actions",
            "effective_actions",
            "transition_tokens",
            "generation_action_seeds",
        ]
        for state in rows(run / "states_public.jsonl"):
            if state["theorem_id"] not in {r["theorem_id"] for r in panel[:16]}:
                continue
            key_state = entry["training_seed"], state["theorem_id"], state["count"]
            histories = [
                {field: attempts[uid]["candidate"][field] for field in canonical_fields}
                for uid in state["incumbent_ids"]
            ]
            canonical = {
                "incumbent_ids": state["incumbent_ids"],
                "histories": histories,
                "family": state["family"],
                "forced": state["forced"],
                "controller_tokens": state["controller_tokens"],
            }
            probabilities = [state["accept_probability"], *state["selection_probabilities"]]
            if key_state in starting_states:
                old_canonical, old_probabilities = starting_states[key_state]
                if old_canonical != canonical or not np.allclose(
                    old_probabilities, probabilities, atol=1e-7, rtol=0
                ):
                    raise ValueError("Common-panel conditions have different starting states")
            else:
                starting_states[key_state] = canonical, probabilities
        mapping = mappings.setdefault(entry["condition"], {})
        config = configs[key]
        expected_ids = set(config["theorem_ids"])
        for row in per_state:
            if row["theorem_id"] not in expected_ids or row["count"] not in (1, 2):
                raise ValueError("Unplanned per-state row")
            row = dict(row)
            row["old_mass_change"] = row["old_correct_mass"] - row["accept_value"]
            k = entry["training_seed"], row["theorem_id"], row["count"], row["mode"]
            if k in mapping:
                raise ValueError("Duplicate paired state")
            mapping[k] = row
        provenance[str(key)] = {
            "status": "analyzed",
            "eligibility_by_count": summary["eligibility_by_count"],
            "per_state_sha256": sha(analysis / "per_state.jsonl"),
            "summary_sha256": sha(analysis / "summary.json"),
            "config_sha256": sha(Path(entry["run"]) / "config.json"),
            "consumed_inputs": {
                name: sha(Path(entry["run"]) / name)
                for name in [
                    "states_public.jsonl",
                    "attempts_public.jsonl",
                    "branches_scored.jsonl",
                    "completion.json",
                    "config.json",
                ]
            },
            "authoritative_analyzer_rerun_matches": True,
        }
    result = aggregate(mappings, panel, args.bootstrap, args.seed)
    result["signature"] = registered
    result["provenance"] = provenance
    result["common_starting_state_identity_verified"] = True
    result["starting_state_identity_scope"] = (
        "all present paired states on common first16; "
        "missing states retain unavailable fixed endpoints"
    )
    result["scope"] = "fixed two checkpoints; eligible-only summaries are descriptive"
    result["cross_condition_inference"] = (
        "common-first16 condition effects are descriptive; "
        "no paired horizon-difference test is reported"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "aggregate_causal.py").write_bytes(Path(__file__).read_bytes())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.7))
    labels, values, errors = [], [], []
    for condition, item in result["common_first16"].items():
        if item["available"]:
            metric = item["metrics"]["delta_continuation"]
            labels.append(condition.replace("_", " "))
            values.append(metric["mean"])
            errors.append([metric["mean"] - metric["ci95"][0], metric["ci95"][1] - metric["mean"]])
    if labels:
        ax.errorbar(values, range(len(labels)), xerr=np.asarray(errors).T, fmt="o")
        ax.set_yticks(range(len(labels)), labels)
    else:
        ax.text(0.5, 0.5, "Fixed-panel aggregates unavailable", ha="center", transform=ax.transAxes)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Continuation correctness change, common first 16 theorems")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"continuation_effects.{suffix}", dpi=160)
    plt.close(fig)
    print(json.dumps({"primary_available": result["primary"]["available"], "out": str(out)}))


if __name__ == "__main__":
    main()
