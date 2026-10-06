"""Independent historical BCE order expectations and required scoring work."""

import ast
import itertools  # noqa: F401 - used by compiled independent functions
import json
import math  # noqa: F401 - used by compiled independent functions
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
from tokenizer import make_tokenizer  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


# Reuse only the reviewer's independent subset-enumeration mathematical functions.
tree = ast.parse(Path(__file__).with_name("audit_main_feature_staging.py").read_text())
functions = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in ("fixed_probabilities", "behavior")
]
exec(
    compile(
        ast.Module(body=functions, type_ignores=[]),
        "<independent-review-order-expectations>",
        "exec",
    )
)
public = rows(ART / "pools/historical_textbook72_test/public_transitions.jsonl")
private = {
    r["theorem_id"]: {x["candidate_id"]: x["valid"] for x in r["labels"]}
    for r in rows(ART / "pools/historical_textbook72_test/private.jsonl")
}
tok = make_tokenizer("lean_staten")
max_error = 0.0
checked = 0
for seed in (0, 1, 2):
    pred = {
        r["theorem_id"]: r
        for r in read(
            ART
            / "selectors/final_historical"
            / f"main_factorized_bce_transitions_s{seed}_perm77106/predictions.json"
        )
    }
    policy = read(ART / f"analysis/bce_threshold_main_dev_s{seed}_v3/frozen_policy.json")
    directory = ART / f"analysis/historical_test_bce_s{seed}_staged_v1"
    assert read(directory / "summary.json")["selected_on_development"] == policy["choices"]
    saved = {(r["theorem_id"], r["policy"]): r for r in rows(directory / "per_theorem.jsonl")}
    for row in public:
        candidates = sorted(row["candidates"], key=lambda c: c["candidate_id"])
        scores = pred[row["theorem_id"]]["candidate_validity_probabilities"]
        lengths = [
            len(tok.encode_prompt(row["prompt"])) + len(c["transition_tokens"]) + 5
            for c in candidates
        ]
        assert sum(lengths) == pred[row["theorem_id"]]["input_tokens"]
        charged = [n if n <= 8192 else 0 for n in lengths]
        labels = [private[row["theorem_id"]][i] for i in range(8)]
        for name, p in policy["policies"].items():
            chosen, included, attempts = behavior(scores, p)  # noqa: F821 - compiled above
            scoring = included if p.get("count") != 1 else [0.0] * 8
            if "mixture" in p:
                m = p["mixture"]
                w = m["right_weight"]
                scoring = [
                    (
                        (1 - w) * m["left_count"] * (m["left_count"] != 1)
                        + w * m["right_count"] * (m["right_count"] != 1)
                    )
                    / 8
                ] * 8
            generation = sum(
                q * (c["prompt_tokens"] + c["generated_tokens"] - len(c["raw_actions"]))
                for q, c in zip(included, candidates, strict=True)
            )
            critic = sum(q * n for q, n in zip(scoring, charged, strict=True))
            expected = {
                "accuracy": sum(q * y for q, y in zip(chosen, labels, strict=True)),
                "attempts": attempts,
                "generator_input_tokens_lower_bound": generation,
                "required_feature_replay_tokens_unpadded": 0.0,
                "critic_prefill_tokens_proxy": critic,
                "critic_dense_attention_positions_proxy": sum(
                    q * n * n for q, n in zip(scoring, charged, strict=True)
                ),
                "total_model_input_tokens_lower_bound": generation + critic,
            }
            target = saved[row["theorem_id"], name]
            for key, value in expected.items():
                error = abs(value - target[key]) / max(1.0, abs(value))
                max_error = max(max_error, error)
                assert error < 1e-12, (seed, name, key, error)
            checked += 1
result = {
    "seeds": 3,
    "pools_each": 72,
    "policy_rows": checked,
    "maximum_relative_error": max_error,
    "scope": "Frozen MAIN DEV policy application; no fitting or neural inference.",
}
Path(__file__).with_name("historical_bce_staging_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
