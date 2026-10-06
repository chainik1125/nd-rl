"""Freeze length-only stopping and fixed-budget mixtures using DEV paths only."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from evaluate_live_path import integrate
from evaluation_protocol import fingerprint, signature, validate_saved_implementation

CONTEXT_GRID = (512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384)
FIXED_COUNTS = (1, 2, 4, 8)


def averages(values):
    return {
        "correct_probability": np.mean([value["correct_probability"] for value in values]).item(),
        "attempts": np.mean([value["attempts"] for value in values]).item(),
        "reward": np.mean([value["reward"] for value in values]).item(),
        "total_forward_tokens": np.mean(
            [
                value["expected_policy_work"]["forward_tokens"]
                + value["expected_policy_work"]["controller_forward_tokens"]
                for value in values
            ]
        ).item(),
    }


def matched_mixture(fixed, target, metric):
    candidates = []
    for a, b in itertools.combinations_with_replacement(FIXED_COUNTS, 2):
        low, high = fixed[str(a)][metric], fixed[str(b)][metric]
        if abs(high - low) < 1e-12:
            if abs(target - low) > 1e-8:
                continue
            weight = 1.0
        else:
            weight = (high - target) / (high - low)
            if not -1e-8 <= weight <= 1 + 1e-8:
                continue
            weight = min(1.0, max(0.0, weight))
        weights = dict.fromkeys(map(str, FIXED_COUNTS), 0.0)
        weights[str(a)] += weight
        weights[str(b)] += 1 - weight
        reward = sum(weights[k] * row["reward"] for k, row in fixed.items())
        candidates.append((reward, weights))
    if not candidates:
        return {"feasible": False, "target": target, "metric": metric}
    _, weights = max(candidates, key=lambda item: item[0])
    return {"feasible": True, "metric": metric, "target": target, "weights": weights}


def joint_mixture(fixed, learned):
    candidates = []
    scale = max(learned["total_forward_tokens"], 1.0)
    target = np.array([1, learned["attempts"], learned["total_forward_tokens"] / scale])
    subsets = itertools.chain.from_iterable(
        itertools.combinations(FIXED_COUNTS, size) for size in (1, 2, 3)
    )
    for counts in subsets:
        matrix = np.array(
            [
                [1.0] * len(counts),
                [fixed[str(k)]["attempts"] for k in counts],
                [fixed[str(k)]["total_forward_tokens"] / scale for k in counts],
            ]
        )
        values = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if not np.allclose(matrix @ values, target, rtol=1e-8, atol=1e-8):
            continue
        if np.min(values) < -1e-8:
            continue
        values = np.maximum(values, 0)
        values /= values.sum()
        weights = dict.fromkeys(map(str, FIXED_COUNTS), 0.0)
        weights.update(zip(map(str, counts), values.tolist(), strict=True))
        reward = sum(weights[k] * row["reward"] for k, row in fixed.items())
        candidates.append((reward, weights))
    if not candidates:
        return {"feasible": False, "reason": "learned DEV budget outside fixed-mixture convex hull"}
    _, weights = max(candidates, key=lambda item: item[0])
    return {"feasible": True, "weights": weights}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths_file, labels_file, data_file = map(Path, (args.paths, args.labels, args.data))
    config = json.loads(paths_file.with_name("config.json").read_text())
    validate_saved_implementation(config)
    data = [json.loads(line) for line in data_file.read_text().splitlines()]
    if any(row.get("split") != "dev" for row in data):
        raise ValueError("baseline fitting requires development only")
    if config["data_sha256"] != hashlib.sha256(data_file.read_bytes()).hexdigest():
        raise ValueError("fitting dataset differs from evaluated development data")
    labels = {}
    for line in labels_file.read_text().splitlines():
        row = json.loads(line)
        key = row["theorem_id"], row["repeat"]
        if key in labels:
            raise ValueError("duplicate private path key")
        labels[key] = row["validities"]
    paths = [json.loads(line) for line in paths_file.read_text().splitlines()]
    keys = [(path["theorem_id"], path["repeat"]) for path in paths]
    if len(set(keys)) != len(keys) or set(keys) != labels.keys():
        raise ValueError("public/private path keys disagree")
    if not {key[0] for key in keys} <= {row["theorem_id"] for row in data}:
        raise ValueError("paths include unregistered theorem IDs")
    summaries = {}
    for method in [
        "learned",
        *[f"fixed{k}" for k in FIXED_COUNTS],
        *[f"context{k}" for k in CONTEXT_GRID],
    ]:
        summaries[method] = averages(
            [
                integrate(path, labels[key], method=method, cost=config["cost"])
                for path, key in zip(paths, keys, strict=True)
            ]
        )
    threshold = max(
        CONTEXT_GRID,
        key=lambda k: (
            summaries[f"context{k}"]["reward"],
            -summaries[f"context{k}"]["attempts"],
            -k,
        ),
    )
    fixed = {str(k): summaries[f"fixed{k}"] for k in FIXED_COUNTS}
    learned = summaries["learned"]
    result = {
        "fit_split": "dev",
        "objective_cost": config["cost"],
        "fit_config_sha256": hashlib.sha256(
            paths_file.with_name("config.json").read_bytes()
        ).hexdigest(),
        "base_sha256": config["base_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "inference_sha256": fingerprint(signature(config)),
        "fit_data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "fit_public_paths_sha256": hashlib.sha256(paths_file.read_bytes()).hexdigest(),
        "fit_private_labels_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest(),
        "fitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "context_threshold": threshold,
        "context_grid": CONTEXT_GRID,
        "selection_rule": (
            "maximize native DEV expected reward; tie fewer attempts then smaller threshold"
        ),
        "mixture_rule": (
            "among exact DEV budget matches maximize native DEV reward; "
            "retain frozen weights on TEST"
        ),
        "attempt_matched_mixture": matched_mixture(fixed, learned["attempts"], "attempts"),
        "forward_token_matched_mixture": matched_mixture(
            fixed, learned["total_forward_tokens"], "total_forward_tokens"
        ),
        "joint_matched_mixture": joint_mixture(fixed, learned),
        "dev_summaries": summaries,
    }
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
    with out.with_suffix(".fitter.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
