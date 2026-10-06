"""Prespecified secondary head versus matched BCE seed0 and labeled seed mean."""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("head_compare_" + name, FOLDER / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


statistics = module("aggregate_selection")
head_eval = module("evaluate_frozen_head")


def analysis_signature(args):
    head_config = json.loads(Path(args.head_config).read_text())
    bce_configs = [json.loads(Path(p).read_text()) for p in args.bce_configs]
    return {
        "method": "secondary_frozen_head_paired_selection_comparison",
        "head_signature": head_config["registered_signature"],
        "bce_signatures_seed0_1_2": [c["registered_signature"] for c in bce_configs],
        "feature_model_sha256": statistics.digest(args.feature_model),
        "training_extractor_snapshot_sha256": statistics.digest(
            Path(args.feature_model).parent / "source_snapshot/analyze_candidates.py"
        ),
        "feature_metrics_sha256": statistics.digest(
            head_eval.extraction.ROOT / "code/shared_code/selection_metrics.py"
        ),
        "feature_source_sha256": statistics.digest(
            FOLDER.parent / "analysis/analyze_candidates.py"
        ),
        "analysis_sha256": statistics.digest(__file__),
        "statistics_helper_sha256": statistics.digest(FOLDER / "aggregate_selection.py"),
        "validator_sha256": statistics.digest(statistics.SHARED / "evaluation_freeze.py"),
        "bootstrap": 5000,
        "seed": 61006,
        "matched_comparator": "BCE seed0, primary permutation77106",
        "secondary_comparator": "mean of BCE seeds0/1/2 within theorem, permutation77106",
        "resampling": "paired theorem within family; fixed fitted head/checkpoints",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--head-predictions", required=True)
    parser.add_argument("--head-config", required=True)
    parser.add_argument("--bce-predictions", nargs=3, required=True)
    parser.add_argument("--bce-configs", nargs=3, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-model", required=True)
    parser.add_argument("--out")
    parser.add_argument("--frozen-test-protocol")
    parser.add_argument("--signature-only", action="store_true")
    args = parser.parse_args()
    signature = analysis_signature(args)
    if args.signature_only:
        print(json.dumps(signature, sort_keys=True))
        return
    public = statistics.read_rows(args.public)
    if len(public) != len({r["theorem_id"] for r in public}):
        raise ValueError("Duplicate public theorem IDs")
    splits = {r["split"] for r in public}
    if splits == {"test"}:
        if not args.frozen_test_protocol:
            raise ValueError("TEST requires frozen comparison registration")
        protocol = json.loads(Path(args.frozen_test_protocol).read_text())
        statistics.validate_frozen_evaluation(protocol, args.public, signature)
        for inference in [signature["head_signature"], *signature["bce_signatures_seed0_1_2"]]:
            statistics.validate_frozen_evaluation(protocol, args.public, inference)
    elif splits != {"dev"}:
        raise ValueError("Invalid split")
    configs = [json.loads(Path(args.head_config).read_text())]
    configs += [json.loads(Path(p).read_text()) for p in args.bce_configs]
    paths = [args.head_predictions, *args.bce_predictions]
    for i, (config, path) in enumerate(zip(configs, paths, strict=True)):
        if config["source_hashes"][config["args"]["public"]] != statistics.digest(args.public):
            raise ValueError("Prediction input mismatch")
        if i == 0:
            if statistics.digest(path) != config["predictions_sha256"]:
                raise ValueError("Head predictions changed")
            expected = head_eval.evaluation_signature(
                config["args"]["head"], config["args"]["checkpoint"]
            )
            if config["registered_signature"] != expected:
                raise ValueError("Head inference signature differs")
        else:
            if Path(path).parent.resolve() != Path(args.bce_configs[i - 1]).parent.resolve():
                raise ValueError("BCE predictions and config are from different directories")
            directory = Path(path).parent
            for name, key in [
                ("evaluate.py", "evaluator_sha256"),
                ("trainer.py", "trainer_sha256"),
                ("bce_trainer.py", "bce_trainer_sha256"),
            ]:
                if statistics.digest(directory / name) != config["registered_signature"][key]:
                    raise ValueError("BCE evaluator/trainer snapshot mismatch")
            if config["checkpoint_metadata"]["selector_config"]["args"]["seed"] != i - 1:
                raise ValueError("BCE model seed aliases differ")
            inference = config["registered_signature"]
            if (
                inference["evaluation_permutation_seed"] != 77106
                or inference["objective"] != "binary"
            ):
                raise ValueError("BCE comparison differs from frozen primary permutation")
            if statistics.digest(config["args"]["checkpoint"]) != inference["checkpoint_sha256"]:
                raise ValueError("BCE checkpoint bytes changed")
    outcome_paths = [args.private, args.features, *paths]
    for path in outcome_paths:
        items = (
            statistics.read_rows(path)
            if path in [args.private, args.features]
            else json.loads(Path(path).read_text())
        )
        if len(items) != len({r["theorem_id"] for r in items}):
            raise ValueError("Duplicate outcome theorem IDs")
    private = {r["theorem_id"]: r for r in statistics.read_rows(args.private)}
    features = {r["theorem_id"]: r for r in statistics.read_rows(args.features)}
    models = [{r["theorem_id"]: r for r in json.loads(Path(p).read_text())} for p in paths]
    sys.path.insert(0, str(FOLDER.parent / "analysis"))
    import analyze_candidates as feature_helper

    feature_model = json.loads(Path(args.feature_model).read_text())
    snapshot = Path(args.feature_model).parent / "source_snapshot/analyze_candidates.py"
    if feature_model["feature_extractor_sha256"] != statistics.digest(snapshot):
        raise ValueError("Frozen feature model training snapshot changed")
    old_tree = ast.parse(snapshot.read_text())
    new_tree = ast.parse((FOLDER.parent / "analysis/analyze_candidates.py").read_text())
    for function in ["public_features", "prepare", "feature_scores"]:
        old = next(
            n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == function
        )
        new = next(
            n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == function
        )
        if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
            raise ValueError("Public feature behavior changed from its frozen training snapshot")
    if feature_model["features"] != feature_helper.FEATURE_NAMES:
        raise ValueError("Feature names/order changed")
    if feature_model["public_environment_sha256"] != statistics.digest(
        head_eval.extraction.SOURCE / "state_env.py"
    ):
        raise ValueError("Frozen feature model uses another public parser")
    if set(feature_model["training_theorem_ids"]) & {r["theorem_id"] for r in public}:
        raise ValueError("Feature model training overlaps evaluation")
    from lean_tok import LeanTokenizer
    from state_env import Env

    prepared = feature_helper.prepare(args.public, args.private, LeanTokenizer("lean_staten"), Env)
    authoritative_feature = {}
    for pool in prepared:
        value = feature_helper.uniform_tie_accuracy(
            pool["valid"], feature_helper.feature_scores(pool["features"], feature_model)
        )
        tid = pool["row"]["theorem_id"]
        if abs(value - features[tid]["accuracy"]["learned_features"]) > 1e-7:
            raise ValueError("Feature report differs from authoritative frozen-model replay")
        authoritative_feature[tid] = value
    ids = sorted(r["theorem_id"] for r in public)
    if len(set(ids)) != len(ids) or any(set(m) != set(ids) for m in [private, features, *models]):
        raise ValueError("Theorem identities differ")
    families = np.asarray([{r["theorem_id"]: r["family"] for r in public}[tid] for tid in ids])
    scores = {
        name: []
        for name in [
            "frozen_head",
            "matched_BCE_seed0",
            "BCE_seed1",
            "BCE_seed2",
            "public_features",
            "random",
        ]
    }
    n_valid, n_complete = [], []
    for tid in ids:
        labels = {c["candidate_id"]: c for c in private[tid]["labels"]}
        if sorted(labels) != list(range(8)):
            raise ValueError("Candidate labels differ from0..7")
        n_valid.append(sum(c["valid"] for c in labels.values()))
        n_complete.append(sum(c["parse_complete"] for c in labels.values()))
        for name, predictions in zip(list(scores)[:4], models, strict=True):
            logits = predictions[tid]["candidate_validity_logits"]
            if len(logits) != 8 or not np.isfinite(logits).all():
                raise ValueError("Invalid canonical critic logits")
            tied = np.flatnonzero(np.asarray(logits) == max(logits)).tolist()
            scores[name].append(sum(labels[i]["valid"] for i in tied) / len(tied))
        scores["public_features"].append(authoritative_feature[tid])
        scores["random"].append(n_valid[-1] / 8)
    scores = {k: np.asarray(v) for k, v in scores.items()}
    scores["secondary_BCE_three_seed_mean"] = np.mean(
        [scores[k] for k in ["matched_BCE_seed0", "BCE_seed1", "BCE_seed2"]], axis=0
    )
    oracle = (np.asarray(n_valid) > 0).astype(float)
    masks = [("all", "all", np.ones(len(ids), dtype=bool))]
    masks += [("family", f, families == f) for f in sorted(set(families))]
    masks += [("n_valid", n, np.asarray(n_valid) == n) for n in range(9)]
    masks += [("n_complete", n, np.asarray(n_complete) == n) for n in range(9)]
    masks += [
        (
            "all_complete_mixed",
            True,
            (np.asarray(n_complete) == 8) & (np.asarray(n_valid) > 0) & (np.asarray(n_valid) < 8),
        )
    ]
    result = {
        "signature": signature,
        "strata": [
            {
                "axis": axis,
                "value": value,
                "theorems": int(mask.sum()),
                "metrics": {
                    name: statistics.endpoint(values[mask], scores["random"][mask], oracle[mask])
                    if mask.any()
                    else None
                    for name, values in scores.items()
                },
            }
            for axis, value, mask in masks
        ],
        "paired_head_minus_comparator": {
            name: statistics.stratified_interval(
                scores["frozen_head"] - value, families, 5000, 61006
            )
            for name, value in scores.items()
            if name != "frozen_head"
        },
        "inputs": {
            p: statistics.digest(p)
            for p in [
                args.public,
                args.private,
                args.features,
                *paths,
                args.head_config,
                *args.bce_configs,
            ]
        },
        "scope": (
            "secondary fixed head; matched seed0 and labeled BCE three-seed mean, "
            "no winner selection"
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result["paired_head_minus_comparator"]))


if __name__ == "__main__":
    main()
