"""Descriptive transfer readouts with paired base-problem cluster intervals."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OLD = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def cluster_weights(rows, samples, seed):
    groups = sorted({r["base_cluster"] for r in rows})
    membership = np.array(
        [[r["base_cluster"] == group for r in rows] for group in groups], dtype=float
    )
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=samples)
    weights = sampled @ membership
    return weights / weights.sum(axis=1, keepdims=True)


def estimates(table, weights):
    draws = weights @ table
    return {
        name: {
            "mean": float(table[:, i].mean()),
            "ci_low": float(np.quantile(draws[:, i], 0.025)),
            "ci_high": float(np.quantile(draws[:, i], 0.975)),
        }
        for i, name in enumerate(METRICS)
    }


METRICS = (
    "correct_probability",
    "attempts",
    "forced_termination_probability",
    "model_input_tokens",
    "model_forward_calls",
)
METHODS = (
    "learned",
    "fixed1",
    "fixed2",
    "fixed4",
    "fixed8",
    "learned_greedy_index",
    "fixed1_greedy_index",
    "fixed4_greedy_index",
    "fixed8_greedy_index",
    "greedy_stop_greedy_index",
    "oracle4",
    "oracle8",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text())
    if registry["status"] != "frozen" or registry["aggregation_sha256"] != sha(__file__):
        raise ValueError("Transfer statistics not sealed")
    if sha(registry["data_path"]) != registry["test_data_sha256"]:
        raise ValueError("Registered dataset changed")
    for path, expected in registry["implementation_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Immutable implementation changed")
    # Check every launch/config/model binding before reading any integrated verdicts.
    inventories = []
    for entry in registry["registered_models"]:
        folder = Path(registry["results_root"]) / entry["model_key"]
        ledger = json.loads(folder.with_suffix(".launch.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if (
            ledger["status"] != "complete"
            or ledger["returncode"] != 0
            or ledger["registry_sha256"] != sha(registry_path)
            or ledger["launcher_sha256"] != registry["launcher_sha256"]
            or config["frozen_protocol_sha256"] != sha(registry_path)
            or config["data_sha256"] != registry["test_data_sha256"]
            or config["inference_sha256"] != entry["inference_sha256"]
            or config["checkpoint_sha256"] != entry["checkpoint_sha256"]
        ):
            raise ValueError("Incomplete/unbound transfer run")
        inventories.append((entry, folder, config))
    sys.path.insert(0, str(OLD))
    from aggregate_live_fp32_v2 import metrics as checked_metrics
    from evaluation_protocol import fingerprint, signature, validate_saved_implementation

    rows = read(registry["data_path"])
    ids = [r["theorem_id"] for r in rows]
    if len(set(ids)) != len(ids) or len(ids) != registry["test_theorems"]:
        raise ValueError("Duplicate/missing metadata identity")
    tables, provenance = {}, []
    for entry, folder, config in inventories:
        validate_saved_implementation(config)
        if fingerprint(signature(config)) != entry["inference_sha256"]:
            raise ValueError("Inference signature mismatch")
        integrated = json.loads((folder / "integrated.json").read_text())
        private = read(folder / "private_labels.jsonl")
        labels = {(r["theorem_id"], r["repeat"]): r["validities"] for r in private}
        expected = {
            (theorem_id, repeat) for theorem_id in ids for repeat in range(registry["test_repeats"])
        }
        if len(labels) != len(private) or set(labels) != expected:
            raise ValueError("Private theorem/draw joins incomplete")
        model = {}
        for method in METHODS:
            if method.startswith("oracle"):
                count = int(method[6:])
                model[method] = np.array(
                    [
                        [
                            np.mean(
                                [
                                    any(labels[theorem_id, repeat][:count])
                                    for repeat in range(registry["test_repeats"])
                                ]
                            ),
                            count,
                            0.0,
                            0.0,
                            0.0,
                        ]
                        for theorem_id in ids
                    ]
                )
                continue
            lookup = {(r["theorem_id"], r["repeat"]): r for r in integrated[method]}
            if len(lookup) != len(integrated[method]) or set(lookup) != expected:
                raise ValueError("Integrated theorem/draw joins incomplete")
            values = []
            for theorem_id in ids:
                draw_values = []
                for repeat in range(registry["test_repeats"]):
                    row = lookup[theorem_id, repeat]
                    checked_metrics(row, entry["native_cost"], 8)
                    work = row["expected_policy_work"]
                    draw_values.append(
                        [
                            row["correct_probability"],
                            row["attempts"],
                            row["forced_termination_probability"],
                            work["forward_tokens"] + work["controller_forward_tokens"],
                            work["forward_calls"] + work["controller_forward_calls"],
                        ]
                    )
                values.append(np.mean(draw_values, axis=0))
            model[method] = np.array(values)
        tables[entry["model_key"]] = model
        provenance.append(
            {
                "model_key": entry["model_key"],
                "config_sha256": sha(folder / "config.json"),
                "integrated_sha256": sha(folder / "integrated.json"),
                "labels_sha256": sha(folder / "private_labels.jsonl"),
            }
        )
    scopes = {
        "ordinary187": [i for i, r in enumerate(rows) if r["component"] == "ordinary187"],
        "derived225projection": [i for i, r in enumerate(rows) if r["component"] == "derived225"],
    }
    for family in sorted({r["family"] for r in rows}):
        scopes["component:" + family] = [i for i, r in enumerate(rows) if r["family"] == family]
    result = {
        "scope": registry["scope"],
        "provenance": provenance,
        "registry_sha256": sha(registry_path),
        "aggregation_sha256": sha(__file__),
        "uncertainty": registry["uncertainty"],
        "scopes": {},
    }
    for scope, positions in scopes.items():
        scoped_rows = [rows[i] for i in positions]
        weights = cluster_weights(
            scoped_rows, registry["bootstrap_samples"], registry["bootstrap_seed"]
        )
        per_model = {
            key: {method: table[positions] for method, table in model.items()}
            for key, model in tables.items()
        }
        arms = {}
        for arm in sorted({e["arm"] for e in registry["registered_models"]}):
            keys = [e["model_key"] for e in registry["registered_models"] if e["arm"] == arm]
            arms[arm] = {
                method: np.mean([per_model[key][method] for key in keys], axis=0)
                for method in METHODS
            }
        contrasts = {
            arm + ":native-minus-ownfirst": estimates(
                methods["learned"] - methods["fixed1"], weights
            )
            for arm, methods in arms.items()
            if arm in ("cost", "no_cost")
        }
        result["scopes"][scope] = {
            "theorems": len(positions),
            "base_clusters": len({r["base_cluster"] for r in scoped_rows}),
            "arms": {
                arm: {
                    method: (
                        {"correct_probability": estimates(table, weights)["correct_probability"]}
                        if method.startswith("oracle")
                        else estimates(table, weights)
                    )
                    for method, table in methods.items()
                }
                for arm, methods in arms.items()
            },
            "per_model": {
                key: {
                    method: {
                        name: float(table[:, i].mean())
                        for i, name in enumerate(METRICS)
                        if not method.startswith("oracle") or name == "correct_probability"
                    }
                    for method, table in methods.items()
                }
                for key, methods in per_model.items()
            },
            "contrasts": contrasts,
        }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
