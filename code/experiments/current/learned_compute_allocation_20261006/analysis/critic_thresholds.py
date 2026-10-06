"""Development-calibrated stopping from a frozen factorized validity critic.

Each score uses only the current proof and original theorem. Saved future scores
are used to integrate over candidate orders, never as stopping-policy inputs.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from analyze_candidates import ROOT, load_rows, sha256
from threshold_baselines import COSTS, calibrate, evaluate

from code.shared_code.evaluation_freeze import validate_frozen_evaluation


def analysis_signature(critic_signature, policy_path):
    return {
        "method": "factorized_binary_critic_thresholds",
        "critic_signature": critic_signature,
        "policy_sha256": sha256(policy_path),
        "evaluator_sha256": sha256(__file__),
        "policy_evaluator_sha256": sha256(Path(__file__).with_name("threshold_baselines.py")),
        "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibrate-dev", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--frozen-test-protocol")
    args = parser.parse_args()
    if args.calibrate_dev == bool(args.policy):
        raise ValueError("Choose development calibration or evaluation of a frozen policy")
    public = load_rows(args.public)
    config = json.loads(Path(args.prediction_config).read_text())
    critic_signature = config["registered_signature"]
    if (
        critic_signature["architecture"] != "factorized"
        or critic_signature["objective"] != "binary"
    ):
        raise ValueError("Threshold staging requires an independent per-candidate binary critic")
    if critic_signature["view"] != "transitions":
        raise ValueError("This analysis is frozen to the full public transition view")
    if config["source_hashes"][config["args"]["public"]] != sha256(args.public):
        raise ValueError("Prediction config describes a different public pool")
    if Path(args.predictions).parent.resolve() != Path(args.prediction_config).parent.resolve():
        raise ValueError("Predictions and provenance must come from the same evaluation directory")
    if args.calibrate_dev and any(row["split"] != "dev" for row in public):
        raise ValueError("Threshold calibration is development-only")
    if any(row["split"] == "test" for row in public):
        if not args.frozen_test_protocol or not args.policy:
            raise ValueError("Test analysis requires registered models and frozen policies")
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()),
            args.public,
            analysis_signature(critic_signature, args.policy),
        )
    frozen = None
    if args.policy:
        frozen = json.loads(Path(args.policy).read_text())
        if frozen["critic_signature"] != critic_signature:
            raise ValueError("Critic differs from development calibration")
        expected_hashes = {
            "evaluator_sha256": sha256(__file__),
            "policy_evaluator_sha256": sha256(Path(__file__).with_name("threshold_baselines.py")),
            "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
        }
        if any(frozen[key] != value for key, value in expected_hashes.items()):
            raise ValueError("Policy implementation differs from its development snapshot")
    # Outcome-bearing files are opened only after the test-registration gate.
    predictions = json.loads(Path(args.predictions).read_text())
    predicted = {row["theorem_id"]: row for row in predictions}
    private = {row["theorem_id"]: row["labels"] for row in load_rows(args.private)}
    public_ids = {row["theorem_id"] for row in public}
    if set(predicted) != public_ids or set(private) != public_ids:
        raise ValueError("Prediction, public and private theorem sets differ")
    if len(predicted) != len(predictions) or len(public_ids) != len(public):
        raise ValueError("Duplicate theorem records")
    sys.path.insert(
        0, str(ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan")
    )
    from lean_tok import LeanTokenizer

    tok = LeanTokenizer("lean_staten")
    prepared = []
    for row in public:
        candidates = sorted(row["candidates"], key=lambda candidate: candidate["candidate_id"])
        if [c["candidate_id"] for c in candidates] != list(range(8)):
            raise ValueError("Expected exactly the eight canonical candidate IDs")
        row = dict(row, candidates=candidates)
        prediction = predicted[row["theorem_id"]]
        probabilities = prediction["candidate_validity_probabilities"]
        if len(probabilities) != 8 or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
        ):
            raise ValueError("Critic scores must be finite per-candidate probabilities")
        lengths = [
            len(tok.encode_prompt(row["prompt"])) + len(c["transition_tokens"]) + 5
            for c in candidates
        ]
        if sum(lengths) != prediction["input_tokens"]:
            raise ValueError("Critic token accounting differs from actual evaluator")
        # Oversized candidates receive the independent neutral score without a forward.
        charged_lengths = [
            length if length <= critic_signature["context"] else 0 for length in lengths
        ]
        labels = {label["candidate_id"]: label["valid"] for label in private[row["theorem_id"]]}
        prepared.append(
            {
                "row": row,
                "scores": probabilities,
                "valid": [labels[i] for i in range(8)],
                "critic_input_tokens": charged_lengths,
            }
        )
    if args.calibrate_dev:
        policies, choices, summary, rows = calibrate(prepared, None)
    else:
        policies, choices = frozen["policies"], frozen["choices"]
        summary, rows = evaluate(prepared, None, policies)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {
        key: {"path": getattr(args, key), "sha256": sha256(getattr(args, key))}
        for key in ["public", "private", "predictions", "prediction_config"]
    }
    if args.calibrate_dev:
        frozen = {
            "policies": policies,
            "choices": choices,
            "costs": COSTS,
            "critic_signature": critic_signature,
            "development_inputs": inputs,
            "evaluator_sha256": sha256(__file__),
            "policy_evaluator_sha256": sha256(Path(__file__).with_name("threshold_baselines.py")),
            "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
            "selection": (
                "max dev reward; fewer attempts; policy name; fixed mixtures fitted on dev"
            ),
            "scope": (
                "separate frozen generator and BCE-trained critic; "
                "exact candidate-order expectation"
            ),
        }
        (out / "frozen_policy.json").write_text(json.dumps(frozen, indent=2) + "\n")
    report = {
        "theorems": len(prepared),
        "inputs": inputs,
        "metrics": summary,
        "selected_on_development": choices,
        "critic_signature": critic_signature,
        "compute_caveat": (
            "Text-only critic omits generation-confidence replay. Sequential unpadded critic "
            "token proxies are charged; generation inputs are ideal cached lower bounds, "
            "excluding padding/delayed EOS checks. No wall-time equivalence."
        ),
    }
    if args.frozen_test_protocol:
        report["frozen_test_protocol_sha256"] = sha256(args.frozen_test_protocol)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "per_theorem.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for source in [
        Path(__file__),
        Path(__file__).with_name("threshold_baselines.py"),
        ROOT / "code/shared_code/selection_metrics.py",
    ]:
        shutil.copy2(source, snapshot / source.name)
    print(json.dumps({"theorems": len(prepared), "choices": choices}, indent=2))


if __name__ == "__main__":
    main()
