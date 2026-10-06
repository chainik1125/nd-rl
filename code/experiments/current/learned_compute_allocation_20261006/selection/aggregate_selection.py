"""Frozen selection endpoints, stratified paired uncertainty and standalone figures."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SHARED = Path(__file__).resolve().parents[5] / "code/shared_code"
sys.path.insert(0, str(SHARED))
from evaluation_freeze import validate_frozen_evaluation  # noqa: E402

PRIMARY_PERMUTATION = 77106
PERMUTATIONS = [77106, 77107, 77108]
GROUPS = ["pooled", "factorized", "bce"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def analysis_signature(args):
    return {
        "method": "frozen_selection_aggregation",
        "code_sha256": digest(__file__),
        "freeze_validator_sha256": digest(SHARED / "evaluation_freeze.py"),
        "inventory_sha256": digest(args.inventory),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "confidence": 0.95,
        "resampling": "theorem within family; training-seed mean within theorem",
        "primary_permutation": PRIMARY_PERMUTATION,
        "supplementary_permutations": PERMUTATIONS,
    }


def stratified_interval(values, families, samples, seed):
    """Percentile bootstrap preserves the observed number of each family."""
    values = np.asarray(values, dtype=float)
    families = np.asarray(families)
    if len(values) != len(families) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need one finite paired value and family per theorem")
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples)
    for family in sorted(set(families)):
        group = values[families == family]
        indices = rng.integers(len(group), size=(samples, len(group)))
        draws += group[indices].sum(axis=1) / len(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "bootstrap_samples": samples,
        "seed": seed,
        "theorems": len(values),
    }


def endpoint(scores, random, oracle):
    score = float(np.mean(scores))
    base = float(np.mean(random))
    available = float(np.mean(oracle))
    return {
        "theorems": len(scores),
        "accuracy": score,
        "random_accuracy": base,
        "oracle_accuracy": available,
        "conditional_on_available_accuracy": score / available if available else None,
        "oracle_gap_recovery": (score - base) / (available - base) if available > base else None,
    }


def figure(metrics, output, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Pooled\nprimary", "Per-proof\ncontrol", "Binary critic\nsecondary"]
    colors = ["#5575a1", "#d18445", "#719b75"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for index, (group, _name, color) in enumerate(zip(GROUPS, names, colors, strict=True)):
        row = metrics["groups"][group]
        axes[0].bar(index, row["primary"]["accuracy"], color=color, width=0.6)
        for seed in row["seed_values"]:
            axes[0].plot(index, seed["accuracy"], "o", color="#222222", markersize=4)
        pair = row["paired_vs_feature"]
        axes[1].errorbar(
            index,
            pair["mean"],
            yerr=[[pair["mean"] - pair["low"]], [pair["high"] - pair["mean"]]],
            fmt="o",
            color=color,
            capsize=4,
        )
    for key, label, style in [
        ("random", "Random", ":"),
        ("feature", "Public features", "--"),
        ("oracle", "Oracle", "-"),
    ]:
        axes[0].axhline(
            metrics["baselines"][key], color="#444444", linestyle=style, linewidth=1, label=label
        )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Selected proof validity")
    axes[0].plot([], [], "o", color="#222222", markersize=4, label="Training seeds")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_ylabel("Paired gain over public features")
    for ax in axes:
        ax.set_xticks(range(3), names)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{split.upper()}: frozen three-seed selection; order 77106")
    fig.savefig(output / "selection.png", dpi=220)
    fig.savefig(output / "selection.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=61006)
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    signature = analysis_signature(args)
    if args.signature_only:
        print(json.dumps(signature, sort_keys=True))
        return
    inventory = json.loads(Path(args.inventory).read_text())
    public = {r["theorem_id"]: r for r in read_rows(args.public)}
    assert all(r["split"] == args.split for r in public.values())
    if args.split == "test":
        if not args.frozen_test_protocol:
            raise ValueError(
                "Final aggregation needs frozen analysis registration before private read"
            )
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()), args.public, signature
        )
    comparator = inventory.get("feature_comparator")
    if args.split == "test" and not comparator:
        raise ValueError("Final feature comparison requires frozen model/source provenance")
    if comparator:
        assert digest(comparator["model_path"]) == comparator["model_sha256"]
        assert digest(comparator["source_path"]) == comparator["source_sha256"]
    private = {r["theorem_id"]: r for r in read_rows(args.private)}
    feature = {r["theorem_id"]: r for r in read_rows(args.features)}
    assert set(public) == set(private) == set(feature)
    ids = sorted(public)
    labels = []
    valid_by_id = {}
    complete_by_id = {}
    complete = []
    for tid in ids:
        row = sorted(private[tid]["labels"], key=lambda c: c["candidate_id"])
        assert [c["candidate_id"] for c in row] == list(range(8))
        labels.append([bool(c["valid"]) for c in row])
        valid_by_id[tid] = {c["candidate_id"]: bool(c["valid"]) for c in row}
        complete_by_id[tid] = {c["candidate_id"]: bool(c["parse_complete"]) for c in row}
        assert all(not c["valid"] or c["parse_complete"] for c in row)
        complete.append(sum(bool(c["parse_complete"]) for c in row))
    labels = np.asarray(labels)
    n_valid = labels.sum(axis=1)
    families = np.asarray([public[t]["family"] for t in ids])
    assert sorted(set(families)) == ["forward", "ore2"]
    assert sum(families == "forward") == sum(families == "ore2")
    random = n_valid / 8
    oracle = (n_valid > 0).astype(float)
    feature_scores = np.asarray([feature[t]["accuracy"]["learned_features"] for t in ids])
    runs = {}
    file_hashes = {p: digest(p) for p in [args.inventory, args.public, args.private, args.features]}
    for row in inventory["runs"]:
        key = (row["group"], row["training_seed"], row["permutation_seed"])
        assert key not in runs and row["group"] in GROUPS
        assert digest(row["checkpoint"]) == row["checkpoint_sha256"]
        path = row["predictions"]
        config_path = row.get("prediction_config")
        if args.split == "test" and not config_path:
            raise ValueError("Final predictions require matching frozen evaluator config")
        if config_path:
            config = json.loads(Path(config_path).read_text())
            assert config["args"]["split"] == args.split
            assert config["args"]["evaluation_seed"] == row["permutation_seed"]
            assert config["registered_signature"]["architecture"] == (
                "pooled" if row["group"] == "pooled" else "factorized"
            )
            assert config["registered_signature"]["objective"] == (
                "binary" if row["group"] == "bce" else "marginal"
            )
            assert (
                config["checkpoint_metadata"]["selector_config"]["args"]["seed"]
                == row["training_seed"]
            )
            assert config["registered_signature"]["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert Path(config["args"]["public"]).resolve() == Path(args.public).resolve()
            assert config["source_hashes"][config["args"]["public"]] == digest(args.public)
            file_hashes[config_path] = digest(config_path)
        file_hashes[path] = digest(path)
        if row.get("predictions_sha256"):
            assert file_hashes[path] == row["predictions_sha256"]
        predictions = {r["theorem_id"]: r for r in json.loads(Path(path).read_text())}
        assert set(predictions) == set(public)
        values, overflows, positions = [], [], []
        selected_complete, context_lengths, position_validities = [], [], []
        for tid in ids:
            pred = predictions[tid]
            ties = pred["argmax_tied_candidate_ids"]
            assert ties and len(ties) == len(set(ties)) and all(0 <= i < 8 for i in ties)
            value = sum(valid_by_id[tid][i] for i in ties) / len(ties)
            assert abs(value - pred["argmax_tie_expected_accuracy"]) < 1e-7
            assert sorted(pred["candidate_order"]) == list(range(8))
            values.append(value)
            overflows.append(bool(pred["overflow"]))
            positions.append([pred["candidate_order"].index(i) for i in ties])
            position_validities.append([valid_by_id[tid][i] for i in ties])
            selected_complete.append(sum(complete_by_id[tid][i] for i in ties) / len(ties))
            context_lengths.append(pred["max_forward_tokens"])
        runs[key] = {
            "values": np.asarray(values),
            "overflow": np.asarray(overflows),
            "positions": positions,
            "position_validities": position_validities,
            "selected_complete": np.asarray(selected_complete),
            "context_lengths": np.asarray(context_lengths),
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
    result = {
        "split": args.split,
        "baselines": {
            "random": float(random.mean()),
            "feature": float(feature_scores.mean()),
            "oracle": float(oracle.mean()),
        },
        "groups": {},
        "interpretation": "Intervals condition on saved pools and these three training seeds; "
        "they do not estimate uncertainty over unseen training runs.",
    }
    per_theorem = []
    for group in GROUPS:
        primary_keys = [(group, seed, PRIMARY_PERMUTATION) for seed in [0, 1, 2]]
        assert all(key in runs for key in primary_keys)
        averaged = np.mean([runs[key]["values"] for key in primary_keys], axis=0)
        overflow = np.mean([runs[key]["overflow"] for key in primary_keys], axis=0)
        selected_complete = np.mean(
            [runs[key]["selected_complete"] for key in primary_keys], axis=0
        )
        context_lengths = runs[primary_keys[0]]["context_lengths"]
        assert all(
            np.array_equal(runs[key]["context_lengths"], context_lengths) for key in primary_keys
        )
        position_mass = np.zeros(8)
        position_correct_mass = np.zeros(8)
        for key in primary_keys:
            for positions, validities in zip(
                runs[key]["positions"], runs[key]["position_validities"], strict=True
            ):
                for position, valid in zip(positions, validities, strict=True):
                    position_mass[position] += 1 / (3 * len(positions))
                    position_correct_mass[position] += valid / (3 * len(positions))
        row = {
            "primary": endpoint(averaged, random, oracle),
            "seed_values": [
                {
                    "training_seed": k[1],
                    "accuracy": float(runs[k]["values"].mean()),
                    "checkpoint_sha256": runs[k]["checkpoint_sha256"],
                    "overflow_count": int(runs[k]["overflow"].sum()),
                }
                for k in primary_keys
            ],
            "paired_vs_random": stratified_interval(
                averaged - random, families, args.bootstrap, args.seed
            ),
            "paired_vs_feature": stratified_interval(
                averaged - feature_scores, families, args.bootstrap, args.seed
            ),
            "overflow_theorem_mass": float(overflow.sum()),
            "selected_complete_theorem_mass": float(selected_complete.sum()),
            "selected_incomplete_theorem_mass": float(len(ids) - selected_complete.sum()),
            "selected_complete_accuracy": float(averaged.sum() / selected_complete.sum())
            if selected_complete.sum()
            else None,
            "selected_position": [
                {
                    "position": position,
                    "theorem_mass": float(position_mass[position]),
                    "accuracy": float(position_correct_mass[position] / position_mass[position])
                    if position_mass[position]
                    else None,
                }
                for position in range(8)
            ],
            "strata": [],
            "permutations": [],
        }
        masks = [("family", family, families == family) for family in sorted(set(families))]
        masks += [("n_valid", n, n_valid == n) for n in range(9)]
        masks += [("n_complete", n, np.asarray(complete) == n) for n in range(9)]
        masks += [
            (
                "all_complete_mixed",
                True,
                (np.asarray(complete) == 8) & (n_valid > 0) & (n_valid < 8),
            )
        ]
        for low, high, name in [
            (0, 1024, "<=1024"),
            (1025, 2048, "1025-2048"),
            (2049, 4096, "2049-4096"),
            (4097, 8192, "4097-8192"),
            (8193, float("inf"), ">8192"),
        ]:
            masks.append(
                (
                    "single_forward_context_length",
                    name,
                    (context_lengths >= low) & (context_lengths <= high),
                )
            )
        for axis, value, mask in masks:
            metrics = endpoint(averaged[mask], random[mask], oracle[mask]) if mask.any() else None
            row["strata"].append(
                {
                    "axis": axis,
                    "value": value,
                    "metrics": metrics,
                    "feature_accuracy": float(feature_scores[mask].mean()) if mask.any() else None,
                }
            )
        for permutation in PERMUTATIONS:
            keys = [(group, seed, permutation) for seed in [0, 1, 2]]
            if all(key in runs for key in keys):
                mean = np.mean([runs[key]["values"] for key in keys], axis=0)
                row["permutations"].append(
                    {
                        "permutation_seed": permutation,
                        "accuracy": float(mean.mean()),
                        "seed_values": [float(runs[key]["values"].mean()) for key in keys],
                    }
                )
            elif args.split == "test":
                raise ValueError("Final report requires every declared seed and permutation")
        permutation_scores = [r["accuracy"] for r in row["permutations"]]
        row["permutation_summary"] = {
            "available": len(permutation_scores),
            "required": 3,
            "mean": float(np.mean(permutation_scores)),
            "range": [min(permutation_scores), max(permutation_scores)],
        }
        result["groups"][group] = row
        for index, tid in enumerate(ids):
            per_theorem.append(
                {
                    "group": group,
                    "theorem_id": tid,
                    "family": families[index],
                    "n_valid": int(n_valid[index]),
                    "n_complete": complete[index],
                    "seed_mean_accuracy": float(averaged[index]),
                    "random_accuracy": float(random[index]),
                    "feature_accuracy": float(feature_scores[index]),
                    "oracle": float(oracle[index]),
                    "overflow_seed_fraction": float(overflow[index]),
                    "selected_complete_seed_fraction": float(selected_complete[index]),
                    "max_forward_tokens": int(context_lengths[index]),
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "per_theorem.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_theorem))
    (out / "config.json").write_text(
        json.dumps({"args": vars(args), "signature": signature, "inputs": file_hashes}, indent=2)
        + "\n"
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    figure(result, out, args.split)
    print(json.dumps({g: result["groups"][g]["primary"] for g in GROUPS}))


if __name__ == "__main__":
    main()
