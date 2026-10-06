"""Prespecified paired summaries for frozen-pool adaptive policies."""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
sys.path.insert(0, str(ROOT / "code/shared_code"))
sys.path.insert(0, str(WORK / "selection"))
from aggregate_selection import stratified_interval  # noqa: E402
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402

METRICS = ["accuracy", "attempts", "total_model_input_tokens_lower_bound"]
COSTS = ["0.0", "0.01", "0.03", "0.1"]
MATCHES = ["attempts", "total_model_input_tokens_lower_bound"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(args):
    return {
        "method": "staged_policy_aggregation",
        "code_sha256": digest(__file__),
        "bootstrap_code_sha256": digest(WORK / "selection/aggregate_selection.py"),
        "validator_sha256": digest(ROOT / "code/shared_code/evaluation_freeze.py"),
        "inventory_sha256": digest(args.inventory),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "confidence": 0.95,
        "resampling": "theorem within family; critic-seed mean within theorem",
        "costs": COSTS,
        "primary_cost": "0.03",
        "matched_metrics": MATCHES,
    }


def load_run(run, public_rows, protocol_hash):
    if digest(run["policy"]) != run["policy_sha256"]:
        raise ValueError("Frozen policy hash differs")
    policy = json.loads(Path(run["policy"]).read_text())
    folder = Path(run["evaluation"])
    report = json.loads((folder / "summary.json").read_text())
    source_public = report["inputs"]["public"]
    if source_public["sha256"] != digest(source_public["path"]):
        raise ValueError("Staged output public pool hash differs")
    source_rows = [
        json.loads(line) for line in Path(source_public["path"]).read_text().splitlines()
    ]
    original = {row["theorem_id"]: row for row in public_rows}
    if len(source_rows) != len(original) or {r["theorem_id"] for r in source_rows} != set(original):
        raise ValueError("Staged output has a different number of public theorems")
    for row in source_rows:
        base = original[row["theorem_id"]]
        if any(row[key] != value for key, value in base.items() if key != "candidates"):
            raise ValueError("Staged public theorem differs from aggregation pool")
        candidates = {item["candidate_id"]: item for item in row["candidates"]}
        if len(candidates) != len(base["candidates"]) or len(candidates) != len(row["candidates"]):
            raise ValueError("Staged candidate count differs")
        for candidate in base["candidates"]:
            if any(
                candidates[candidate["candidate_id"]][key] != value
                for key, value in candidate.items()
            ):
                raise ValueError("Staged public candidate differs from aggregation pool")
    if report["selected_on_development"] != policy["choices"]:
        raise ValueError("Staged choices differ from frozen development choices")
    if protocol_hash and report.get("frozen_test_protocol_sha256") != protocol_hash:
        raise ValueError("Staged output used a different test registration")
    if (
        "critic_signature" in policy
        and report.get("critic_signature") != policy["critic_signature"]
    ):
        raise ValueError("Critic signature differs from policy")
    if "critic_signature" in policy:
        config_input = report["inputs"]["prediction_config"]
        if digest(config_input["path"]) != config_input["sha256"]:
            raise ValueError("Critic evaluation provenance changed")
        config = json.loads(Path(config_input["path"]).read_text())
        if config["registered_signature"] != policy["critic_signature"]:
            raise ValueError("Critic provenance signature differs")
        if config["checkpoint_metadata"]["selector_config"]["args"]["seed"] != run["training_seed"]:
            raise ValueError("Critic training seed differs from inventory")
    elif report["inputs"]["feature_model"]["sha256"] != policy["feature_model_sha256"]:
        raise ValueError("Feature model differs from frozen policy")
    sources = {
        "metrics_sha256": "selection_metrics.py",
        "evaluator_sha256": "critic_thresholds.py"
        if "critic_signature" in policy
        else "threshold_baselines.py",
    }
    if "critic_signature" in policy:
        sources["policy_evaluator_sha256"] = "threshold_baselines.py"
    for key, name in sources.items():
        if digest(folder / "source_snapshot" / name) != policy[key]:
            raise ValueError("Staged evaluator snapshot differs from frozen policy")
    rows = [json.loads(line) for line in (folder / "per_theorem.jsonl").read_text().splitlines()]
    index = {(row["theorem_id"], row["policy"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate policy/theorem observations")
    for row in rows:
        if any(not math.isfinite(row[key]) for key in METRICS):
            raise ValueError("Nonfinite policy metric")
        if (
            not -1e-12 <= row["accuracy"] <= 1 + 1e-12
            or not 1 - 1e-12 <= row["attempts"] <= 8 + 1e-12
        ):
            raise ValueError("Policy metric outside its support")
        if row["total_model_input_tokens_lower_bound"] < 0:
            raise ValueError("Negative model work")
    for name, metrics in report["metrics"].items():
        selected = [row for row in rows if row["policy"] == name]
        if len(selected) != len(public_rows):
            raise ValueError("Missing policy/theorem rows")
        if any(
            not math.isclose(
                np.mean([r[key] for r in selected]), metrics[key], abs_tol=1e-10, rel_tol=1e-12
            )
            for key in METRICS
        ):
            raise ValueError("Policy summary differs from theorem records")
    return policy, index


def aggregate(args):
    public = [json.loads(line) for line in Path(args.public).read_text().splitlines()]
    ids = [row["theorem_id"] for row in public]
    if len(ids) != len(set(ids)) or any(row["split"] != args.split for row in public):
        raise ValueError("Public pool has duplicate IDs or wrong split")
    protocol_hash = None
    if args.split == "test":
        if not args.frozen_test_protocol:
            raise ValueError("Test aggregation requires frozen registration")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()), args.public, signature(args)
        )
        protocol_hash = digest(args.frozen_test_protocol)
    inventory = json.loads(Path(args.inventory).read_text())
    identities = [(run["group"], run["training_seed"]) for run in inventory["runs"]]
    if len(identities) != 4 or set(identities) != {
        ("feature", None),
        ("bce", 0),
        ("bce", 1),
        ("bce", 2),
    }:
        raise ValueError("Require one feature model and exactly BCE seeds0,1,2")
    families = [row["family"] for row in public]
    groups = {}
    all_rows = []
    seen = set()
    for run in inventory["runs"]:
        identity = (run["group"], run["training_seed"])
        if identity in seen:
            raise ValueError("Duplicate training seed within group")
        seen.add(identity)
        policy, index = load_run(run, public, protocol_hash)
        if {key[0] for key in index} != set(ids):
            raise ValueError("Staged theorem IDs differ from public pool")
        for cost in COSTS:
            choice = policy["choices"][cost]
            names = {"adaptive": choice["threshold"], "dev_best_fixed": choice["fixed"]}
            names.update({f"fixed{k}": f"fixed{k}" for k in [1, 2, 4, 8]})
            names.update(
                {f"matched_{metric}": choice["matched_mixtures"][metric] for metric in MATCHES}
            )
            for kind, name in names.items():
                if name is None:
                    continue
                for theorem_id in ids:
                    source = index[theorem_id, name]
                    if source["family"] != public[ids.index(theorem_id)]["family"]:
                        raise ValueError("Family differs from public record")
                    row = {key: source[key] for key in METRICS}
                    row["reward"] = row["accuracy"] - float(cost) * (row["attempts"] - 1)
                    row.update(
                        group=run["group"],
                        seed=run["training_seed"],
                        cost=cost,
                        kind=kind,
                        theorem_id=theorem_id,
                        family=source["family"],
                    )
                    all_rows.append(row)
    for group in sorted({run["group"] for run in inventory["runs"]}):
        groups[group] = {}
        for cost in COSTS:
            subset = [row for row in all_rows if row["group"] == group and row["cost"] == cost]
            vectors = {}
            for kind in sorted({row["kind"] for row in subset}):
                selected = [row for row in subset if row["kind"] == kind]
                expected_seeds = {seed for name, seed in identities if name == group}
                if {row["seed"] for row in selected} != expected_seeds:
                    continue
                vectors[kind] = {
                    metric: np.array(
                        [
                            np.mean([r[metric] for r in selected if r["theorem_id"] == tid])
                            for tid in ids
                        ]
                    )
                    for metric in METRICS + ["reward"]
                }
            result = {
                "policies": {
                    kind: {m: float(v.mean()) for m, v in values.items()}
                    for kind, values in vectors.items()
                },
                "comparisons": {},
                "training_seed_metrics": {
                    str(seed): {
                        kind: {
                            metric: float(
                                np.mean(
                                    [
                                        r[metric]
                                        for r in subset
                                        if r["seed"] == seed and r["kind"] == kind
                                    ]
                                )
                            )
                            for metric in METRICS + ["reward"]
                        }
                        for kind in sorted({r["kind"] for r in subset if r["seed"] == seed})
                    }
                    for name, seed in identities
                    if name == group
                },
            }
            for comparator in ["dev_best_fixed", *[f"matched_{m}" for m in MATCHES]]:
                if comparator not in vectors:
                    continue
                result["comparisons"][comparator] = {
                    metric: stratified_interval(
                        vectors["adaptive"][metric] - vectors[comparator][metric],
                        families,
                        args.bootstrap,
                        args.seed,
                    )
                    for metric in METRICS + ["reward"]
                }
            groups[group][cost] = result
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "split": args.split,
        "signature": signature(args),
        "groups": groups,
        "consumed_inputs": {
            str(Path(run["evaluation"]) / name): digest(Path(run["evaluation"]) / name)
            for run in inventory["runs"]
            for name in ["summary.json", "per_theorem.jsonl"]
        },
        "caveats": [
            "DEV choices are optimized and exploratory; TEST freezes them.",
            "Intervals condition on saved candidate pools, checkpoints and development choices.",
            "Fixed-mixture compute equality is fitted on DEV; TEST mismatch is reported.",
            "Logical input-token lower bounds are not actual FLOPs or GPU latency.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "per_theorem.jsonl").write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    shutil.copy2(__file__, out / "source.py")
    shutil.copy2(WORK / "selection/aggregate_selection.py", out / "bootstrap_source.py")
    plot(groups, out, args.split)
    print(json.dumps({group: data["0.03"] for group, data in groups.items()}, indent=2))


def plot(groups, out, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for group, color in [("feature", "#ba7740"), ("bce", "#447dad")]:
        data = groups[group]["0.03"]["policies"]
        fixed = [data[f"fixed{k}"] for k in [1, 2, 4, 8]]
        adaptive = data["adaptive"]
        for ax, metric, scale in [
            (axes[0], "attempts", 1),
            (axes[1], "total_model_input_tokens_lower_bound", 1000),
        ]:
            ax.plot(
                [r[metric] / scale for r in fixed],
                [100 * r["accuracy"] for r in fixed],
                "o--",
                color=color,
                alpha=0.75,
                label=f"{group}: fixed budgets",
            )
            ax.scatter(
                adaptive[metric] / scale,
                100 * adaptive["accuracy"],
                marker="*",
                s=180,
                color=color,
                label=f"{group}: adaptive, cost 0.03",
                zorder=5,
            )
            ax.grid(alpha=0.2)
            ax.set_ylabel("Selected proof correct (%)")
    axes[0].set_xlabel("Mean attempts")
    axes[1].set_xlabel("Logical model input tokens, lower bound (thousands)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{split.upper()}: independent attempts with frozen scoring and stopping")
    for extension in ["png", "svg"]:
        fig.savefig(out / f"staged_policies.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=61006)
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    if args.signature_only:
        print(json.dumps(signature(args), indent=2))
    else:
        aggregate(args)
