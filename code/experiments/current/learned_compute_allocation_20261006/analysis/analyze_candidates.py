"""Independent candidate audit and non-oracle baselines on fixed theorem pools.

All prediction features come from public generation records. The syntax baseline
replays visible actions in the unmodified syntactic environment; it never calls
Lean. Private outcomes are joined only for scoring or training the feature model.
"""

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from code.shared_code.evaluation_freeze import validate_frozen_evaluation  # noqa: E402
from code.shared_code.selection_metrics import (  # noqa: E402
    paired_theorem_interval,
    selection_summary,
    uniform_tie_accuracy,
)


def load_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_signature(feature_model_path=None):
    return {
        "method": "public_feature_baselines",
        "feature_model_sha256": sha256(feature_model_path) if feature_model_path else None,
        "evaluator_sha256": sha256(__file__),
        "metrics_sha256": sha256(ROOT / "code/shared_code/selection_metrics.py"),
        "public_environment_sha256": sha256(
            ROOT
            / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan/state_env.py"
        ),
        "tie_breaking": "exact expected uniform argmax ties",
    }


def public_features(row, candidate, tok, env_class):
    env = env_class(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
    accepted = 0
    actions = candidate["effective_actions"]
    for action in actions:
        tokens, ended = tok.decode_action(action)
        if not ended:
            break
        ok, _ = env.apply(tokens)
        if not ok:
            break
        accepted += 1
        if env.done:
            break
    done = env.done and accepted == len(actions)
    text = candidate["rendered_text"]
    return [
        candidate["mean_base_logprob"],
        candidate["mean_logprob"],
        candidate["mean_entropy"],
        math.log1p(candidate["generated_tokens"]),
        math.log1p(len(actions)),
        math.log1p(candidate["prompt_tokens"]),
        float(done),
        accepted / max(1, len(actions)),
        text.split().count("exact") / max(1, len(actions)),
        text.split().count("Or.elim") / max(1, len(actions)),
    ]


FEATURE_NAMES = [
    "mean_base_logprob",
    "mean_sampling_logprob",
    "mean_entropy",
    "log1p_generated_tokens",
    "log1p_action_count",
    "log1p_prefill_tokens",
    "public_syntax_complete",
    "public_accepted_action_fraction",
    "exact_action_fraction",
    "or_elim_action_fraction",
]


def prepare(public_path, private_path, tok, env_class):
    rows = load_rows(public_path)
    private = {row["theorem_id"]: row["labels"] for row in load_rows(private_path)}
    if len({row["theorem_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate theorem rows need explicit grouped analysis")
    result = []
    for row in rows:
        labels = {label["candidate_id"]: label for label in private[row["theorem_id"]]}
        candidates = row["candidates"]
        if len({c["candidate_id"] for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate IDs")
        result.append(
            {
                "row": row,
                "features": [public_features(row, c, tok, env_class) for c in candidates],
                "labels": [labels[c["candidate_id"]] for c in candidates],
                "valid": [labels[c["candidate_id"]]["valid"] for c in candidates],
            }
        )
    return result


def fit_feature_model(prepared, output):
    if any(pool["row"]["split"] != "train" for pool in prepared):
        raise ValueError("Feature model training is restricted to the training split")
    x = torch.tensor([v for pool in prepared for v in pool["features"]], dtype=torch.float64)
    y = torch.tensor([v for pool in prepared for v in pool["valid"]], dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    weights = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, intercept], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ weights + intercept, y)
        loss = loss + 0.001 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    model = {
        "features": FEATURE_NAMES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.detach().tolist(),
        "intercept": intercept.item(),
        "training_theorems": len(prepared),
        "training_theorem_ids": sorted(pool["row"]["theorem_id"] for pool in prepared),
        "training_candidates": len(y),
        "l2": 0.001,
        "training_loss": float(closure().detach()),
    }
    output.write_text(json.dumps(model, indent=2) + "\n")
    return model


def feature_scores(features, model):
    return [
        model["intercept"]
        + sum(
            (value - mean) / scale * weight
            for value, mean, scale, weight in zip(
                row, model["mean"], model["scale"], model["weights"], strict=True
            )
        )
        for row in features
    ]


def evaluate(prepared, tok, feature_model=None):
    theorem_rows = []
    contingency = {
        "complete_valid": 0,
        "complete_invalid": 0,
        "incomplete_valid": 0,
        "incomplete_invalid": 0,
    }
    public_completion_mismatches = 0
    for pool in prepared:
        row, features, valid = pool["row"], pool["features"], pool["valid"]
        scores = {
            "likelihood_t1": [f[0] for f in features],
            "likelihood_sampling": [f[1] for f in features],
            "negative_entropy": [-f[2] for f in features],
            "shortest": [-f[3] for f in features],
            "public_syntax_complete": [f[6] for f in features],
        }
        if feature_model:
            scores["learned_features"] = feature_scores(features, feature_model)
        accuracy = {name: uniform_tie_accuracy(valid, value) for name, value in scores.items()}
        accuracy["random"] = statistics.fmean(valid)
        accuracy["first_candidate"] = float(valid[0])
        accuracy["oracle"] = float(any(valid))
        # Lexicographic syntax/likelihood uses only public replay plus generator logits.
        lex = [(f[6], f[0]) for f in features]
        maximum = max(lex)
        accuracy["syntax_then_likelihood"] = statistics.fmean(
            float(y) for y, value in zip(valid, lex, strict=True) if value == maximum
        )
        for feature, label in zip(features, pool["labels"], strict=True):
            key = ("complete" if label["parse_complete"] else "incomplete") + "_"
            key += "valid" if label["valid"] else "invalid"
            contingency[key] += 1
            public_completion_mismatches += bool(feature[6]) != label["parse_complete"]
        theorem_rows.append(
            {
                "theorem_id": row["theorem_id"],
                "family": row.get("family", "unknown"),
                "n_valid": sum(valid),
                "n_candidates": len(valid),
                "n_complete": sum(bool(f[6]) for f in features),
                "pool_tokens": len(tok.encode_prompt(row["prompt"]))
                + sum(len(c["effective_tokens"]) + 3 for c in row["candidates"])
                + 1,
                "accuracy": accuracy,
            }
        )
    names = list(theorem_rows[0]["accuracy"])
    metrics = {}
    for name in names:
        values = [r["accuracy"][name] for r in theorem_rows]
        metrics[name] = selection_summary([p["valid"] for p in prepared], values)
        if name != "oracle":
            metrics[name]["paired_vs_random"] = paired_theorem_interval(
                {
                    r["theorem_id"]: [r["accuracy"][name] - r["accuracy"]["random"]]
                    for r in theorem_rows
                },
                bootstrap_samples=2000,
            ).to_dict()
    sizes = sorted(r["pool_tokens"] for r in theorem_rows)
    summary = {
        "theorems": len(prepared),
        "candidates": sum(len(p["valid"]) for p in prepared),
        "completion_validity": contingency,
        "public_completion_mismatches": public_completion_mismatches,
        "pool_tokens": {
            "min": sizes[0],
            "median": statistics.median(sizes),
            "max": sizes[-1],
            "above1024": sum(x > 1024 for x in sizes),
            "above2048": sum(x > 2048 for x in sizes),
            "above4096": sum(x > 4096 for x in sizes),
        },
        "metrics": metrics,
        "tie_breaking": "expected accuracy under uniform ties",
        "sampling_unit": "theorem; bootstrap conditions on fixed saved candidate pools",
        "features": FEATURE_NAMES,
    }
    return summary, theorem_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-public")
    parser.add_argument("--train-private")
    parser.add_argument("--feature-model")
    parser.add_argument("--frozen-test-protocol")
    args = parser.parse_args()
    source = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
    sys.path.insert(0, str(source))
    from lean_tok import LeanTokenizer
    from state_env import Env

    torch.set_num_threads(4)
    tok = LeanTokenizer("lean_staten")
    # Check split and protocol before opening outcomes.
    public = load_rows(args.public)
    if any(row["split"] == "test" for row in public):
        if not args.frozen_test_protocol or args.train_public:
            raise ValueError(
                "Test evaluation requires a frozen protocol and no simultaneous training"
            )
        validate_frozen_evaluation(
            json.loads(Path(args.frozen_test_protocol).read_text()),
            args.public,
            analysis_signature(args.feature_model),
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), ROOT / "code/shared_code/selection_metrics.py"]:
        shutil.copy2(path, snapshot / path.name)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    model = None
    if args.train_public:
        if not args.train_private:
            raise ValueError("Training requires separate labels")
        train = prepare(args.train_public, args.train_private, tok, Env)
        if {r["theorem_id"] for r in public} & {p["row"]["theorem_id"] for p in train}:
            raise ValueError("Training and evaluation theorem sets overlap")
        model = fit_feature_model(train, out / "feature_model.json")
        model["training_inputs"] = {
            "public": str(args.train_public),
            "private": str(args.train_private),
            "public_sha256": sha256(args.train_public),
            "private_sha256": sha256(args.train_private),
        }
        model["feature_extractor_sha256"] = sha256(__file__)
        model["public_environment_sha256"] = sha256(source / "state_env.py")
        (out / "feature_model.json").write_text(json.dumps(model, indent=2) + "\n")
    elif args.feature_model:
        model = json.loads(Path(args.feature_model).read_text())
        if model["features"] != FEATURE_NAMES:
            raise ValueError("Saved feature model names/order differ from the evaluator")
        if {row["theorem_id"] for row in public} & set(model["training_theorem_ids"]):
            raise ValueError("Saved feature model training overlaps evaluation")
    prepared = prepare(args.public, args.private, tok, Env)
    summary, rows = evaluate(prepared, tok, model)
    summary["inputs"] = {
        "public": str(args.public),
        "private": str(args.private),
        "public_sha256": sha256(args.public),
        "private_sha256": sha256(args.private),
    }
    summary["analysis_signature"] = analysis_signature(args.feature_model)
    if args.frozen_test_protocol:
        summary["frozen_test_protocol_sha256"] = sha256(args.frozen_test_protocol)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "per_theorem.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
