"""Independent frozen historical feature and permutation-policy arithmetic, no fitting."""

import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ANALYSIS = ART / "analysis"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
from state_env import Env  # noqa: E402 - runtime source path
from tokenizer import make_tokenizer  # noqa: E402 - runtime source path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(row, candidate):
    environment = Env(row["prompt"], canon=True, base=candidate["canonical_name_base"], assign=True)
    accepted = 0
    for action in candidate["raw_actions"]:
        tokens, ended = tok.decode_action(action)
        if not ended:
            break
        ok, _ = environment.apply(tokens)
        if not ok:
            break
        accepted += 1
        if environment.done:
            break
    count = len(candidate["raw_actions"])
    words = candidate["rendered_text"].split()
    return [
        candidate["mean_base_logprob"],
        candidate["mean_logprob"],
        candidate["mean_entropy"],
        math.log1p(candidate["generated_tokens"]),
        math.log1p(count),
        math.log1p(candidate["prompt_tokens"]),
        float(environment.done and accepted == count),
        accepted / max(count, 1),
        words.count("exact") / max(count, 1),
        words.count("Or.elim") / max(count, 1),
    ]


def fixed_probabilities(scores, count):
    # Enumerate unordered prefixes: each k-subset is equiprobable under random order.
    selected = [0.0] * 8
    weight = 1 / math.comb(8, count)
    for subset in itertools.combinations(range(8), count):
        best = max(scores[i] for i in subset)
        ties = [i for i in subset if scores[i] == best]
        for i in ties:
            selected[i] += weight / len(ties)
    return selected, [count / 8] * 8, count


def behavior(scores, policy):
    if "count" in policy:
        return fixed_probabilities(scores, policy["count"])
    if "mixture" in policy:
        mixture = policy["mixture"]
        left = fixed_probabilities(scores, mixture["left_count"])
        right = fixed_probabilities(scores, mixture["right_count"])
        weight = mixture["right_weight"]
        return (
            [(1 - weight) * a + weight * b for a, b in zip(left[0], right[0], strict=True)],
            [(1 - weight) * a + weight * b for a, b in zip(left[1], right[1], strict=True)],
            (1 - weight) * left[2] + weight * right[2],
        )
    qualified = [i for i, score in enumerate(scores) if score >= policy["threshold"]]
    if not qualified:
        return fixed_probabilities(scores, 8)
    count = len(qualified)
    return (
        [1 / count if i in qualified else 0 for i in range(8)],
        [1 / count if i in qualified else 1 / (count + 1) for i in range(8)],
        9 / (count + 1),
    )


tok = make_tokenizer("lean_staten")
model_path = ANALYSIS / "features1600_main_dev/feature_model.json"
policy_path = ANALYSIS / "threshold1600_main_dev_v3/frozen_policy.json"
model, frozen = read(model_path), read(policy_path)
registry_path = ART / "final_evaluation/historical_protocol.json"
registry = read(registry_path)
registered = [entry["signature"] for entry in registry["registered_analyses"]]
assert any(
    s.get("method") == "public_feature_thresholds"
    and s["feature_model_sha256"] == sha(model_path)
    and s["policy_sha256"] == sha(policy_path)
    for s in registered
)
public = rows(ART / "pools/historical_textbook72_test/public.jsonl")
private = {
    r["theorem_id"]: {c["candidate_id"]: c for c in r["labels"]}
    for r in rows(ART / "pools/historical_textbook72_test/private.jsonl")
}
baseline_path = ANALYSIS / "historical_test_features_v1"
staged_path = ANALYSIS / "historical_test_feature_staged_v1"
baseline = {r["theorem_id"]: r for r in rows(baseline_path / "per_theorem.jsonl")}
staged = {(r["theorem_id"], r["policy"]): r for r in rows(staged_path / "per_theorem.jsonl")}
summary = read(staged_path / "summary.json")
assert summary["frozen_test_protocol_sha256"] == sha(registry_path)
assert summary["selected_on_development"] == frozen["choices"]
assert not set(model["training_theorem_ids"]) & {r["theorem_id"] for r in public}
errors = []
for row in public:
    candidates = row["candidates"]
    labels = [private[row["theorem_id"]][c["candidate_id"]]["valid"] for c in candidates]
    scores = []
    for candidate in candidates:
        vector = features(row, candidate)
        logit = model["intercept"] + sum(
            (v - m) / s * w
            for v, m, s, w in zip(
                vector, model["mean"], model["scale"], model["weights"], strict=True
            )
        )
        scores.append(1 / (1 + math.exp(-max(-700, min(700, logit)))))
    best = max(scores)
    ties = [i for i in range(8) if scores[i] == best]
    accuracy = sum(labels[i] for i in ties) / len(ties)
    errors.append(abs(accuracy - baseline[row["theorem_id"]]["accuracy"]["learned_features"]))
    for name, policy in frozen["policies"].items():
        chosen, included, attempts = behavior(scores, policy)
        scoring = included if policy.get("count") != 1 else [0] * 8
        if "mixture" in policy:
            mix = policy["mixture"]
            weight = mix["right_weight"]
            scoring = [
                (
                    (1 - weight) * mix["left_count"] * (mix["left_count"] != 1)
                    + weight * mix["right_count"] * (mix["right_count"] != 1)
                )
                / 8
            ] * 8
        generation = sum(
            p * (c["prompt_tokens"] + c["generated_tokens"] - len(c["raw_actions"]))
            for p, c in zip(included, candidates, strict=True)
        )
        replay = sum(
            p * (c["prompt_tokens"] + c["generated_tokens"])
            for p, c in zip(scoring, candidates, strict=True)
        )
        expected = {
            "accuracy": sum(p * y for p, y in zip(chosen, labels, strict=True)),
            "attempts": attempts,
            "generated_tokens_proxy": sum(
                p * c["generated_tokens"] for p, c in zip(included, candidates, strict=True)
            ),
            "generator_prefill_tokens_proxy": sum(
                p * c["prompt_tokens"] for p, c in zip(included, candidates, strict=True)
            ),
            "generator_input_tokens_lower_bound": generation,
            "required_feature_replay_tokens_unpadded": replay,
            "total_model_input_tokens_lower_bound": generation + replay,
        }
        saved = staged[row["theorem_id"], name]
        for field, value in expected.items():
            errors.append(abs(value - saved[field]))
        if "count" in policy:
            k = policy["count"]
            invalid = 8 - sum(labels)
            oracle = 1 - (math.comb(invalid, k) / math.comb(8, k) if invalid >= k else 0)
            errors.append(abs(oracle - saved["oracle_accuracy"]))
for name, metrics in summary["metrics"].items():
    records = [r for (tid, policy), r in staged.items() if policy == name]
    for field, value in metrics.items():
        if isinstance(value, (int, float)):
            errors.append(abs(value - statistics.fmean(r[field] for r in records)))
    for cost, reward in metrics["rewards"].items():
        errors.append(abs(reward - metrics["accuracy"] + float(cost) * (metrics["attempts"] - 1)))
assert max(errors) < 1e-9
choice = frozen["choices"]["0.03"]
threshold = summary["metrics"][choice["threshold"]]
mixture = summary["metrics"][choice["matched_mixtures"]["total_model_input_tokens_lower_bound"]]
result = {
    "theorems": len(public),
    "policy_rows_checked": len(staged),
    "scalar_checks": len(errors),
    "max_absolute_error": max(errors),
    "feature_model_sha256": sha(model_path),
    "frozen_dev_policy_sha256": sha(policy_path),
    "threshold_cost03": threshold,
    "unchanged_dev_work_mixture": mixture,
    "test_work_difference": threshold["total_model_input_tokens_lower_bound"]
    - mixture["total_model_input_tokens_lower_bound"],
    "feature8_accuracy": statistics.fmean(
        r["accuracy"]["learned_features"] for r in baseline.values()
    ),
}
Path(__file__).with_name("historical_feature_staging_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
