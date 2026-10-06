"""Authorized frozen-head MAIN provenance, cached linear readout and policy audit."""

import ast
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Reuse only the two independently written combinatorial functions, with no module execution.
helper = Path(__file__).with_name("audit_main_feature_staging.py")
tree = ast.parse(helper.read_text())
functions = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in {"fixed_probabilities", "behavior"}
]
namespace = {"math": math, "itertools": itertools}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper), "exec"), namespace)
behavior = namespace["behavior"]
registry = read(ART / "final_evaluation/main_protocol.json")
registered = [entry["signature"] for entry in registry["registered_analyses"]]
pred_dir = ART / "selectors/final_main/frozen_head"
config = read(pred_dir / "config.json")
assert config["registered_signature"] in registered
assert sha(config["args"]["head"]) == config["head_sha256"]
assert sha(config["args"]["checkpoint"]) == config["checkpoint_sha256"]
assert (
    config["parameter_sha256_before"]
    == config["parameter_sha256_after"]
    == "594173e7d67a64f5c4ffa9ac5176e4390770e44ce7d307590474bed3957c8eb5"
)
assert sha(pred_dir / "predictions.json") == config["predictions_sha256"]
for path, expected in config["registered_signature"]["extractor_sources"].items():
    assert sha(path) == expected
for path, expected in config["source_hashes"].items():
    assert sha(path) == expected
paired_dir = ART / "analysis/final_main/frozen_head_paired"
paired = read(paired_dir / "metrics.json")
assert paired["signature"] in registered
assert sha(paired_dir / "source.py") == paired["signature"]["analysis_sha256"]
for path, expected in paired["inputs"].items():
    assert sha(path) == expected
stage_dir = ART / "analysis/final_main/frozen_head_thresholds"
stage_config = read(stage_dir / "config.json")
assert stage_config["signature"] in registered
policy = read(stage_config["args"]["policy"])
assert policy == read(stage_dir / "policy.json")
assert sha(stage_config["args"]["policy"]) == stage_config["signature"]["policy_sha256"]
for path, expected in stage_config["inputs"].items():
    assert sha(path) == expected
predictions = {r["theorem_id"]: r for r in read(pred_dir / "predictions.json")}
public = {r["theorem_id"]: r for r in rows(config["args"]["public"])}
private = {
    r["theorem_id"]: sorted(r["labels"], key=lambda c: c["candidate_id"])
    for r in rows(config["args"]["private"])
}
ids = sorted(public)
valid = {tid: np.array([c["valid"] for c in private[tid]]) for tid in ids}
families = np.array([public[tid]["family"] for tid in ids])
head = torch.load(config["args"]["head"], map_location="cpu", weights_only=True)
assert head["weight"].numel() + head["bias"].numel() == 385
cache_path = pred_dir / "test_features.npz"
assert sha(cache_path) == config["feature_metadata"]["cache_sha256"]
cache = np.load(cache_path, allow_pickle=False)
logits = cache["features"].astype(np.float64) @ head["weight"].numpy() + head["bias"].item()
linear_error = max(
    abs(value - predictions[str(tid)]["candidate_validity_logits"][int(cid)])
    for value, tid, cid in zip(logits, cache["theorem_ids"], cache["candidate_ids"], strict=True)
)
assert linear_error < 1e-12
scores = {}
for name, path in [("frozen_head", pred_dir / "predictions.json")] + [
    (
        "matched_BCE_seed0" if seed == 0 else f"BCE_seed{seed}",
        ART
        / f"selectors/final_main/main_factorized_bce_transitions_s{seed}_perm77106"
        / "predictions.json",
    )
    for seed in range(3)
]:
    predictions_by_id = {r["theorem_id"]: r for r in read(path)}
    values = []
    for tid in ids:
        raw = np.asarray(predictions_by_id[tid]["candidate_validity_logits"])
        ties = np.flatnonzero(raw == max(raw))
        values.append(valid[tid][ties].mean())
    scores[name] = np.array(values)
features = {
    r["theorem_id"]: r for r in rows(ART / "analysis/main_test_features_v1/per_theorem.jsonl")
}
scores["public_features"] = np.array([features[t]["accuracy"]["learned_features"] for t in ids])
scores["random"] = np.array([valid[t].mean() for t in ids])
scores["secondary_BCE_three_seed_mean"] = np.mean(
    [scores[k] for k in ["matched_BCE_seed0", "BCE_seed1", "BCE_seed2"]], axis=0
)
errors = []
for name, result in paired["paired_head_minus_comparator"].items():
    difference = scores["frozen_head"] - scores[name]
    rng = np.random.default_rng(61006)
    draws = np.zeros(5000)
    for family in sorted(set(families)):
        values = difference[families == family]
        draws += values[rng.integers(len(values), size=(5000, len(values)))].sum(axis=1) / 400
    low, high = np.quantile(draws, [0.025, 0.975])
    errors.extend(
        [
            abs(result["mean"] - difference.mean()),
            abs(result["low"] - low),
            abs(result["high"] - high),
        ]
    )
staged = rows(stage_dir / "per_theorem.jsonl")
for saved in staged:
    tid = saved["theorem_id"]
    pred = predictions[tid]
    chosen, included, attempts = behavior(
        pred["candidate_validity_probabilities"], policy["policies"][saved["policy"]]
    )
    specification = policy["policies"][saved["policy"]]
    scored = included if specification.get("count") != 1 else [0] * 8
    if "mixture" in specification:
        mix = specification["mixture"]
        w = mix["right_weight"]
        scored = [
            (
                (1 - w) * mix["left_count"] * (mix["left_count"] != 1)
                + w * mix["right_count"] * (mix["right_count"] != 1)
            )
            / 8
        ] * 8
    generation = sum(
        p * (c["prompt_tokens"] + c["generated_tokens"] - len(c["raw_actions"]))
        for p, c in zip(included, public[tid]["candidates"], strict=True)
    )
    critic = sum(p * n for p, n in zip(scored, pred["candidate_input_tokens"], strict=True))
    expected = {
        "accuracy": sum(p * y for p, y in zip(chosen, valid[tid], strict=True)),
        "attempts": attempts,
        "generator_input_tokens_lower_bound": generation,
        "required_feature_replay_tokens_unpadded": 0,
        "critic_prefill_tokens_proxy": critic,
        "total_model_input_tokens_lower_bound": generation + critic,
    }
    errors.extend(abs(saved[k] - v) for k, v in expected.items())
summary = read(stage_dir / "summary.json")
for name, metrics in summary.items():
    selected = [r for r in staged if r["policy"] == name]
    for field, value in metrics.items():
        if isinstance(value, (int, float)):
            errors.append(abs(value - float(np.mean([r[field] for r in selected]))))
assert max(errors) < 1e-9
result = {
    "cached_linear_logit_max_error": linear_error,
    "arithmetic_max_error": max(errors),
    "policy_rows_checked": len(staged),
    "frozen_parameter_digest": config["parameter_sha256_after"],
    "paired": paired["paired_head_minus_comparator"],
    "cost03_threshold_name": policy["choices"]["0.03"]["threshold"],
    "cost03_threshold": summary[policy["choices"]["0.03"]["threshold"]],
    "independent_combinatorics_helper_sha256": sha(helper),
}
Path(__file__).with_name("main_frozen_head_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
