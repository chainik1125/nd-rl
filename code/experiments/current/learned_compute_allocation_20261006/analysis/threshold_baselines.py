"""Frozen-pool adaptive baselines, integrating over all candidate permutations.

Selection and stopping use a train-only validity predictor's public features.
Thresholds are chosen on development rewards and frozen before test evaluation.
Additive generation-token totals are proxies, not measured batched GPU compute.
"""

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

import torch
from analyze_candidates import FEATURE_NAMES, ROOT, feature_scores, load_rows, prepare, sha256

from code.shared_code.evaluation_freeze import validate_frozen_evaluation
from code.shared_code.selection_metrics import (
    fixed_count_mixture,
    fixed_pool_policy,
    oracle_pass_at_k,
)

THRESHOLDS = [i / 20 for i in range(20)] + [0.97, 0.98, 0.99, 0.995, 0.999, 1.01]
COSTS = [0.0, 0.01, 0.03, 0.1]


def probability(logit):
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))


def policy_behavior(scores, policy):
    if "mixture" not in policy:
        return fixed_pool_policy(scores, **policy)
    mixture = policy["mixture"]
    left = fixed_pool_policy(scores, count=mixture["left_count"])
    right = fixed_pool_policy(scores, count=mixture["right_count"])
    weight = mixture["right_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("Mixture weight must be a probability")
    result = {
        key: [(1 - weight) * a + weight * b for a, b in zip(left[key], right[key], strict=True)]
        for key in ["selected_probabilities", "included_probabilities"]
    }
    result["expected_attempts"] = (1 - weight) * left["expected_attempts"] + weight * right[
        "expected_attempts"
    ]
    return result


def evaluate(prepared, model, policies):
    results = []
    for pool in prepared:
        scores = (
            [probability(x) for x in feature_scores(pool["features"], model)]
            if model is not None
            else pool["scores"]
        )
        row = pool["row"]
        for name, policy in policies.items():
            behavior = policy_behavior(scores, policy)
            chosen = behavior["selected_probabilities"]
            included = behavior["included_probabilities"]
            # Fixed1 returns its only candidate without scoring. Other fixed
            # counts score their generated subset; thresholds score every arrival.
            score_inclusion = included if policy.get("count") != 1 else [0.0] * len(scores)
            if "mixture" in policy:
                mix = policy["mixture"]
                score_fraction = (
                    (1 - mix["right_weight"]) * mix["left_count"] * (mix["left_count"] != 1)
                    + mix["right_weight"] * mix["right_count"] * (mix["right_count"] != 1)
                ) / len(scores)
                score_inclusion = [score_fraction] * len(scores)
            accuracy = sum(p * y for p, y in zip(chosen, pool["valid"], strict=True))
            item = {
                "theorem_id": row["theorem_id"],
                "family": row.get("family", "unknown"),
                "policy": name,
                "accuracy": accuracy,
                "attempts": behavior["expected_attempts"],
                "generated_tokens_proxy": sum(
                    p * c["generated_tokens"]
                    for p, c in zip(included, row["candidates"], strict=True)
                ),
                "generator_prefill_tokens_proxy": sum(
                    p * c["prompt_tokens"] for p, c in zip(included, row["candidates"], strict=True)
                ),
                "generator_input_tokens_lower_bound": sum(
                    p * (c["prompt_tokens"] + c["generated_tokens"] - len(c["raw_actions"]))
                    for p, c in zip(included, row["candidates"], strict=True)
                ),
                "required_feature_replay_tokens_unpadded": sum(
                    p * (c["prompt_tokens"] + c["generated_tokens"])
                    for p, c in zip(score_inclusion, row["candidates"], strict=True)
                )
                if model is not None
                else 0.0,
            }
            if "critic_input_tokens" in pool:
                item["critic_prefill_tokens_proxy"] = sum(
                    p * tokens
                    for p, tokens in zip(score_inclusion, pool["critic_input_tokens"], strict=True)
                )
                item["critic_dense_attention_positions_proxy"] = sum(
                    p * tokens**2
                    for p, tokens in zip(score_inclusion, pool["critic_input_tokens"], strict=True)
                )
            item["total_model_input_tokens_lower_bound"] = (
                item["generator_input_tokens_lower_bound"]
                + item["required_feature_replay_tokens_unpadded"]
                + item.get("critic_prefill_tokens_proxy", 0.0)
            )
            if "count" in policy:
                item["oracle_accuracy"] = oracle_pass_at_k(pool["valid"], policy["count"])
            results.append(item)
    summary = {}
    for name in policies:
        selected = [r for r in results if r["policy"] == name]
        fields = [
            "accuracy",
            "attempts",
            "generated_tokens_proxy",
            "generator_prefill_tokens_proxy",
            "generator_input_tokens_lower_bound",
            "required_feature_replay_tokens_unpadded",
            "total_model_input_tokens_lower_bound",
        ]
        fields.extend(key for key in selected[0] if key.startswith("critic_"))
        summary[name] = {key: statistics.fmean(r[key] for r in selected) for key in fields}
        summary[name]["rewards"] = {
            str(cost): summary[name]["accuracy"] - cost * (summary[name]["attempts"] - 1)
            for cost in COSTS
        }
        if "count" in policies[name]:
            summary[name]["oracle_accuracy"] = statistics.fmean(
                r["oracle_accuracy"] for r in selected
            )
    return summary, results


def calibrate(prepared, model):
    policies = {f"fixed{k}": {"count": k} for k in [1, 2, 4, 8]}
    policies.update({f"threshold{t:g}": {"threshold": t} for t in THRESHOLDS})
    summary, _ = evaluate(prepared, model, policies)
    choices = {}
    fixed_names = [f"fixed{k}" for k in [1, 2, 4, 8]]
    for cost in COSTS:
        choice = {}
        for kind in ["fixed", "threshold"]:
            candidates = [name for name in policies if name.startswith(kind)]
            choice[kind] = max(
                candidates,
                key=lambda name: (
                    summary[name]["rewards"][str(cost)],
                    -summary[name]["attempts"],
                    name,
                ),
            )
        choice["matched_mixtures"] = {}
        for compute in [
            "attempts",
            "generated_tokens_proxy",
            "generator_prefill_tokens_proxy",
            "total_model_input_tokens_lower_bound",
        ]:
            target = summary[choice["threshold"]][compute]
            envelope = fixed_count_mixture(
                [(summary[name][compute], summary[name]["accuracy"]) for name in fixed_names],
                target,
            )
            if envelope is None:
                choice["matched_mixtures"][compute] = None
                continue
            name = f"mixture_cost{cost:g}_{compute}"
            policies[name] = {
                "mixture": {
                    "left_count": policies[fixed_names[envelope["left_index"]]]["count"],
                    "right_count": policies[fixed_names[envelope["right_index"]]]["count"],
                    "right_weight": envelope["right_weight"],
                    "development_matched_metric": compute,
                    "development_target": target,
                }
            }
            choice["matched_mixtures"][compute] = name
        choices[str(cost)] = choice
    summary, rows = evaluate(prepared, model, policies)
    return policies, choices, summary, rows


def analysis_signature(feature_model_path, policy_path):
    return {
        "method": "public_feature_thresholds",
        "feature_model_sha256": sha256(feature_model_path),
        "policy_sha256": sha256(policy_path),
        "evaluator_sha256": sha256(__file__),
        "extractor_sha256": sha256(Path(__file__).with_name("analyze_candidates.py")),
        "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--feature-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibrate-dev", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--frozen-test-protocol")
    args = parser.parse_args()
    if args.calibrate_dev == bool(args.policy):
        raise ValueError("Either calibrate on development or evaluate a saved policy")
    public = load_rows(args.public)
    if args.calibrate_dev and any(row["split"] != "dev" for row in public):
        raise ValueError("Threshold calibration is development-only")
    if any(row["split"] == "test" for row in public):
        if not args.frozen_test_protocol or not args.policy:
            raise ValueError("Test evaluation requires frozen protocol and policy")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()),
            args.public,
            analysis_signature(args.feature_model, args.policy),
        )
    model = json.loads(Path(args.feature_model).read_text())
    if model["features"] != FEATURE_NAMES:
        raise ValueError("Saved feature model names/order differ from the evaluator")
    if set(model["training_theorem_ids"]) & {r["theorem_id"] for r in public}:
        raise ValueError("Feature predictor was trained on an evaluation theorem")
    if args.calibrate_dev:
        policies = None
        frozen = None
    else:
        frozen = json.loads(Path(args.policy).read_text())
        if frozen["feature_model_sha256"] != sha256(args.feature_model):
            raise ValueError("Feature checkpoint differs from the calibrated policy")
        if frozen["evaluator_sha256"] != sha256(__file__) or frozen["metrics_sha256"] != sha256(
            ROOT / "code/shared_code/selection_metrics.py"
        ):
            raise ValueError("Evaluator differs from the development policy snapshot")
        policies = frozen["policies"]
    source = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
    sys.path.insert(0, str(source))
    from lean_tok import LeanTokenizer
    from state_env import Env

    torch.set_num_threads(4)
    prepared = prepare(args.public, args.private, LeanTokenizer("lean_staten"), Env)
    if args.calibrate_dev:
        policies, choices, summary, rows = calibrate(prepared, model)
    else:
        summary, rows = evaluate(prepared, model, policies)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in [
        Path(__file__),
        Path(__file__).with_name("analyze_candidates.py"),
        ROOT / "code/shared_code/selection_metrics.py",
    ]:
        shutil.copy2(path, snapshot / path.name)
    inputs = {
        key: {"path": str(getattr(args, key)), "sha256": sha256(getattr(args, key))}
        for key in ["public", "private", "feature_model"]
    }
    if args.calibrate_dev:
        frozen = {
            "policies": policies,
            "choices": choices,
            "costs": COSTS,
            "feature_model_sha256": sha256(args.feature_model),
            "development_inputs": inputs,
            "tie_breaking": "expected uniform selection among tied scores",
            "threshold_selection": "maximum dev reward; then fewer attempts; then policy name",
            "scope": "fixed independent candidate pools; exact expectation over uniform order",
            "evaluator_sha256": sha256(__file__),
            "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
        }
        (out / "frozen_policy.json").write_text(json.dumps(frozen, indent=2) + "\n")
    report = {
        "theorems": len(prepared),
        "metrics": summary,
        "inputs": inputs,
        "selected_on_development": frozen["choices"],
        "compute_caveat": (
            "Exact feature predictor requires full-forward LP/entropy replay, charged separately. "
            "Generation input counts are ideal cached lower bounds, excluding padding, "
            "delayed EOS checks and unused decode steps. No actual wall-time equivalence."
        ),
    }
    if args.frozen_test_protocol:
        report["frozen_test_protocol_sha256"] = sha256(args.frozen_test_protocol)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "per_theorem.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"theorems": len(prepared), "choices": frozen["choices"]}, indent=2))


if __name__ == "__main__":
    main()
