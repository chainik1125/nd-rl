"""Apply frozen DEV length thresholds and mixtures to immutable evaluated paths."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_live_path import integrate
from evaluation_protocol import signature, validate_counter, validate_reanalysis


def mix(values, weights):
    first = next(iter(values.values()))
    return {
        **{
            key: sum(weights[k] * value[key] for k, value in values.items())
            for key in (
                "correct_probability",
                "attempts",
                "reward",
                "forced_termination_probability",
            )
        },
        "expected_policy_work": {
            key: sum(weights[k] * value["expected_policy_work"][key] for k, value in values.items())
            for key in first["expected_policy_work"]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-protocol")
    args = parser.parse_args()
    paths_file, labels_file, fit_file = map(Path, (args.paths, args.labels, args.fit))
    config = json.loads(paths_file.with_name("config.json").read_text())
    validate_reanalysis(
        config,
        args.frozen_protocol,
        fit_file,
        "budget_baselines_sha256",
        ("reanalyze_budget_baselines.py", "fit_budget_baselines.py"),
    )
    fit = json.loads(fit_file.read_text())
    if fit["objective_cost"] != config["cost"]:
        raise ValueError("baseline fitting objective differs from the evaluation objective")
    validate_counter(fit, signature(config))
    labels = {}
    for line in labels_file.read_text().splitlines():
        row = json.loads(line)
        key = row["theorem_id"], row["repeat"]
        if key in labels:
            raise ValueError("duplicate private path key")
        labels[key] = row["validities"]
    results = {"context_threshold": []}
    mixture_names = (
        "attempt_matched_mixture",
        "forward_token_matched_mixture",
        "joint_matched_mixture",
    )
    for name in mixture_names:
        if fit[name]["feasible"]:
            results[name] = []
    seen = set()
    for line in paths_file.read_text().splitlines():
        path = json.loads(line)
        key = path["theorem_id"], path["repeat"]
        if key in seen:
            raise ValueError("duplicate public path key")
        seen.add(key)
        values = {
            str(k): integrate(path, labels[key], f"fixed{k}", cost=config["cost"])
            for k in (1, 2, 4, 8)
        }
        context = integrate(
            path, labels[key], f"context{fit['context_threshold']}", cost=config["cost"]
        )
        context.update(theorem_id=key[0], repeat=key[1])
        results["context_threshold"].append(context)
        for name in results:
            if name == "context_threshold":
                continue
            value = mix(values, fit[name]["weights"])
            value.update(theorem_id=key[0], repeat=key[1])
            results[name].append(value)
    if seen != labels.keys():
        raise ValueError("public and private path keys disagree")
    summaries = {}
    for name, rows in results.items():
        summaries[name] = {
            key: sum(row[key] for row in rows) / len(rows)
            for key in (
                "correct_probability",
                "attempts",
                "reward",
                "forced_termination_probability",
            )
        }
        summaries[name]["expected_policy_work"] = {
            key: sum(row["expected_policy_work"][key] for row in rows) / len(rows)
            for key in rows[0]["expected_policy_work"]
        }
    result = {
        "provenance": {
            "paths_sha256": hashlib.sha256(paths_file.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest(),
            "fit_sha256": hashlib.sha256(fit_file.read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_sha256": hashlib.sha256(
                Path(__file__).with_name("evaluate_live_path.py").read_bytes()
            ).hexdigest(),
        },
        "fit": fit,
        "integrated": results,
        "summary": summaries,
        "scope": (
            "DEV-fitted weights retained; TEST compute differences are descriptive, never refitted"
        ),
    }
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(result, stream)
    with out.with_suffix(".implementation.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
