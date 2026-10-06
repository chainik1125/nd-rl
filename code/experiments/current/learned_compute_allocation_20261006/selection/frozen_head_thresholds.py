"""Separate frozen-head staging, using the unchanged root threshold protocol."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
ANALYSIS = FOLDER.parent / "analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS))
from threshold_baselines import COSTS, THRESHOLDS, calibrate, evaluate  # noqa: E402

from code.shared_code.evaluation_freeze import validate_frozen_evaluation  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "frozen_head_policy_eval", FOLDER / "evaluate_frozen_head.py"
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
sha = evaluator.common.digest


def analysis_signature(critic_signature, policy_path):
    return {
        "method": "secondary_frozen_original_scalar_head_thresholds",
        "critic_signature": critic_signature,
        "policy_sha256": sha(policy_path),
        "analysis_sha256": sha(__file__),
        "threshold_helper_sha256": sha(ANALYSIS / "threshold_baselines.py"),
        "metrics_sha256": sha(ROOT / "code/shared_code/selection_metrics.py"),
        "thresholds": THRESHOLDS,
        "costs": COSTS,
        "cost_rule": "text-scoring input tokens; no confidence replay; fixed1 scoring exemption",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--calibrate-dev", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    if args.calibrate_dev == bool(args.policy):
        raise ValueError("Choose DEV calibration or frozen-policy evaluation")
    config = json.loads(Path(args.prediction_config).read_text())
    critic = config["registered_signature"]
    expected = evaluator.evaluation_signature(
        config["args"]["head"], config["args"]["checkpoint"], 8192, 8
    )
    if critic != expected:
        raise ValueError("Critic inference source/model differs from registered signature")
    if args.signature_only:
        print(json.dumps(analysis_signature(critic, args.policy), sort_keys=True))
        return
    if config["source_hashes"][config["args"]["public"]] != sha(args.public):
        raise ValueError("Different public input from prediction config")
    if sha(args.predictions) != config["predictions_sha256"]:
        raise ValueError("Prediction bytes changed")
    public = evaluator.extraction.encoding.read(args.public)
    splits = {r["split"] for r in public}
    if args.calibrate_dev and splits != {"dev"}:
        raise ValueError("Calibration is DEV-only")
    frozen = None
    if not args.calibrate_dev:
        frozen = json.loads(Path(args.policy).read_text())
        if frozen["analysis_sha256"] != sha(__file__):
            raise ValueError("Policy wrapper changed after calibration")
        if frozen["critic_signature"] != critic:
            raise ValueError("Frozen policy uses different critic")
        if frozen["threshold_helper_sha256"] != sha(ANALYSIS / "threshold_baselines.py"):
            raise ValueError("Root threshold helper changed since DEV calibration")
        if frozen["metrics_sha256"] != sha(ROOT / "code/shared_code/selection_metrics.py"):
            raise ValueError("Fixed-pool policy behavior changed")
    if splits == {"test"}:
        if not args.frozen_test_protocol or args.calibrate_dev:
            raise ValueError("TEST requires registry and frozen policy")
        protocol = json.loads(Path(args.frozen_test_protocol).read_text())
        validate_frozen_evaluation(protocol, args.public, analysis_signature(critic, args.policy))
        validate_frozen_evaluation(protocol, args.public, critic)
    elif splits != {"dev"}:
        raise ValueError("Invalid split")
    if len(public) != len({r["theorem_id"] for r in public}):
        raise ValueError("Duplicate public theorem IDs")
    prediction_rows = json.loads(Path(args.predictions).read_text())
    if len(prediction_rows) != len({r["theorem_id"] for r in prediction_rows}):
        raise ValueError("Duplicate prediction theorem IDs")
    predictions = {r["theorem_id"]: r for r in prediction_rows}
    private = evaluator.extraction.encoding.read(args.private)  # Guard precedes label access.
    labels = {r["theorem_id"]: r for r in private}
    if (
        len(labels) != len(private)
        or set(labels) != set(predictions)
        or set(labels) != {r["theorem_id"] for r in public}
    ):
        raise ValueError("Theorem identity mismatch")
    prepared = []
    for row in public:
        tid = row["theorem_id"]
        row["candidates"].sort(key=lambda c: c["candidate_id"])
        valid = sorted(labels[tid]["labels"], key=lambda c: c["candidate_id"])
        if [c["candidate_id"] for c in row["candidates"]] != list(range(8)) or [
            c["candidate_id"] for c in valid
        ] != list(range(8)):
            raise ValueError("Candidate IDs differ from canonical0..7")
        pred = predictions[tid]
        probabilities = pred["candidate_validity_probabilities"]
        lengths = pred["candidate_input_tokens"]
        if (
            len(probabilities) != 8
            or len(lengths) != 8
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities)
        ):
            raise ValueError("Malformed critic probabilities")
        if sum(lengths) != pred["input_tokens"] or any(x < 0 or x > 8192 for x in lengths):
            raise ValueError("Malformed per-candidate critic cost")
        flags = pred["candidate_overflow"]
        if len(flags) != 8 or any(
            (length == 0) != flag for length, flag in zip(lengths, flags, strict=True)
        ):
            raise ValueError("Neutral overflow must have zero model-forward charge")
        if any(
            flag and probability != 0.5
            for flag, probability in zip(flags, probabilities, strict=True)
        ):
            raise ValueError("Overflow probability differs from neutral fallback")
        prepared.append(
            {
                "row": row,
                "valid": [int(c["valid"]) for c in valid],
                "scores": probabilities,
                "critic_input_tokens": lengths,
            }
        )
    if args.calibrate_dev:
        policies, choices, summary, results = calibrate(prepared, None)
        frozen = {
            "critic_signature": critic,
            "policies": policies,
            "choices": choices,
            "thresholds": THRESHOLDS,
            "costs": COSTS,
            "threshold_helper_sha256": sha(ANALYSIS / "threshold_baselines.py"),
            "metrics_sha256": sha(ROOT / "code/shared_code/selection_metrics.py"),
            "analysis_sha256": sha(__file__),
            "development_public_sha256": sha(args.public),
            "development_predictions_sha256": sha(args.predictions),
            "selection_rule": "maximum DEV reward; fewer attempts; policy name",
            "tie_rule": "exact uniform ties and all candidate-order expectation",
            "scope": "secondary frozen-head staged control, separate from three-seed BCE",
        }
    else:
        policies = frozen["policies"]
        summary, results = evaluate(prepared, None, policies)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "policy.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "per_theorem.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "inputs": {
                    p: sha(p)
                    for p in [args.public, args.private, args.predictions, args.prediction_config]
                },
                "signature": analysis_signature(critic, out / "policy.json"),
            },
            indent=2,
        )
        + "\n"
    )
    for source in [
        Path(__file__),
        ANALYSIS / "threshold_baselines.py",
        ROOT / "code/shared_code/selection_metrics.py",
    ]:
        (out / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"choices": frozen["choices"], "out": str(out)}))


if __name__ == "__main__":
    main()
