"""Independent saved-path arithmetic; final outcomes require explicit barrier release."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def reconstruct(path, valid, method, cost, hazards=None, threshold=None):
    decisions = path["decisions"]
    count = len(decisions)
    assert count == len(valid) == len(path["candidates"])
    greedy = method.endswith("_greedy_index")
    base = method.removesuffix("_greedy_index")
    h = []
    forced = []
    quality = []
    selected_age = []
    for i, d in enumerate(decisions):
        force = bool(d["forced"] or i == count - 1)
        forced.append(force)
        if base.startswith("fixed"):
            stop = float(i + 1 >= int(base[5:]))
        elif base == "context_threshold":
            stop = float(len(d["ids"]) >= threshold)
        elif base == "counter":
            stop = hazards[i]
        else:
            assert base in ("learned", "greedy_stop")
            stop = d["accept_probability"]
            if base == "greedy_stop":
                stop = float(stop >= 0.5)
        h.append(1.0 if force else stop)
        p = np.asarray(d["selection_probabilities"], dtype=float)
        assert len(p) == i + 1 and abs(p.sum() - 1) <= 1e-5
        if greedy:
            p = (p == p.max()).astype(float)
            p /= p.sum()
        quality.append(float(p @ np.asarray(valid[: i + 1])))
        selected_age.append(float(p @ np.arange(i + 1, 0, -1)))
    reach = np.r_[1.0, np.cumprod(1 - np.asarray(h))[:-1]]
    mass = reach * h
    assert abs(mass.sum() - 1) < 1e-12
    # Fixed/count/length rules invoke the selector only at a terminal pool of size >=2.
    if base in ("learned", "greedy_stop"):
        cm = reach.copy()
        if forced[0]:
            cm[0] = 0
    else:
        cm = mass.copy()
        cm[0] = 0
    work = {}
    for key in (
        "generated_tokens",
        "forward_calls",
        "forward_tokens",
        "attention_score_cells_per_layer",
    ):
        work[key] = float(reach @ np.array([c[key] for c in path["candidates"]]))
    lengths = np.array([len(d["ids"]) for d in decisions])
    work["controller_forward_tokens"] = float(cm @ lengths)
    work["controller_forward_calls"] = float(cm.sum())
    work["controller_attention_score_cells_per_layer"] = float(cm @ (8 * lengths**2))
    work["policy_wall_seconds"] = float(
        reach @ np.array([c["sampling_wall_seconds"] for c in path["candidates"]])
        + cm @ np.array([d["controller_wall_seconds"] for d in decisions])
    )
    accuracy = float(mass @ quality)
    attempts = float(reach.sum())
    return {
        "correct_probability": accuracy,
        "attempts": attempts,
        "reward": accuracy - cost * (attempts - 1),
        "forced_termination_probability": float(mass @ forced),
        "expected_policy_work": work,
        "selected_age": float(mass @ selected_age),
    }


def audit(directory, released, counter=None, budgets=None, hazards_file=None):
    directory = Path(directory)
    config = read(directory / "config.json")
    if config["evaluation_split"] != "dev" and not released:
        raise ValueError("Final outcomes remain blocked until explicit root all-nine release")
    public = {(r["theorem_id"], r["repeat"]): r for r in rows(directory / "public_paths.jsonl")}
    private_rows = rows(directory / "private_labels.jsonl")
    private = {(r["theorem_id"], r["repeat"]): r["validities"] for r in private_rows}
    assert len(private) == len(private_rows) and public.keys() == private.keys()
    integrated = read(directory / "integrated.json")
    hazard = read(hazards_file)["hazards"] if hazards_file else None
    fit = None
    if counter:
        derived = read(counter)
        integrated.update(derived["integrated"])
        assert (
            hashlib.sha256(Path(hazards_file).read_bytes()).hexdigest()
            == derived["provenance"]["hazards_sha256"]
        )
    if budgets:
        derived = read(budgets)
        integrated.update(derived["integrated"])
        fit = derived["fit"]
    cache = {}
    checked = 0
    max_relative = 0.0
    ages = {}
    for method, outcomes in integrated.items():
        if method.startswith("counter") and hazard is None:
            raise ValueError("Counter output needs explicit hazard vector in audit input")
        for row in outcomes:
            key = row["theorem_id"], row["repeat"]
            if method.endswith("_mixture"):
                weights = fit[method]["weights"]
                assert fit[method]["feasible"] and abs(sum(weights.values()) - 1) < 1e-8
                assert min(weights.values()) >= -1e-8
                values = [
                    reconstruct(public[key], private[key], "fixed" + k, config["cost"])
                    for k in weights
                ]
                expected = {
                    k: sum(weights[n] * v[k] for n, v in zip(weights, values, strict=True))
                    for k in (
                        "correct_probability",
                        "attempts",
                        "reward",
                        "forced_termination_probability",
                        "selected_age",
                    )
                }
                expected["expected_policy_work"] = {
                    k: sum(
                        weights[n] * v["expected_policy_work"][k]
                        for n, v in zip(weights, values, strict=True)
                    )
                    for k in values[0]["expected_policy_work"]
                }
            else:
                expected = reconstruct(
                    public[key],
                    private[key],
                    method,
                    config["cost"],
                    hazard,
                    fit["context_threshold"] if fit else None,
                )
            for field in (
                "correct_probability",
                "attempts",
                "reward",
                "forced_termination_probability",
            ):
                assert np.isclose(expected[field], row[field], rtol=1e-12, atol=1e-12)
            for field, value in expected["expected_policy_work"].items():
                err = abs(value - row["expected_policy_work"][field]) / max(1.0, abs(value))
                max_relative = max(max_relative, err)
                assert err < 1e-12
            checked += 1
            cache[method, key] = expected
        ages[method] = float(
            np.mean(
                [cache[method, (r["theorem_id"], r["repeat"])]["selected_age"] for r in outcomes]
            )
        )
    return {
        "directory": str(directory),
        "split": config["evaluation_split"],
        "policy_rows": checked,
        "maximum_relative_work_error": max_relative,
        "selected_age_means": ages,
        "config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--root-barrier-release", action="store_true")
    parser.add_argument("--counter")
    parser.add_argument("--budgets")
    parser.add_argument("--hazards-file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(
        args.paths, args.root_barrier_release, args.counter, args.budgets, args.hazards_file
    )
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
