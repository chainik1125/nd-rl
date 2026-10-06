"""CPU-only independent arithmetic checks of public DEV confidence diagnostics."""

import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
SEL = ROOT / "code/experiments/current/learned_compute_allocation_20261006/selection"
ART = ROOT / "artifacts/learned_compute_allocation_20261006/selectors"
sys.path.insert(0, str(SEL))
import cached_confidence_diagnostic as primary  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


folder = ART / "cached_confidence_dev16_v1"
config = read(folder / "config.json")
for path, expected in config["source_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
panel = rows(folder / "public_panel.jsonl")
assert len(panel) == 16 and {r["split"] for r in panel} == {"dev"}
actions = rows(folder / "action_records.jsonl")
candidates = rows(folder / "candidate_discrepancies.jsonl")
groups = defaultdict(list)
for r in actions:
    assert r["tokens"] == len(r["raw_action"])
    groups[r["theorem_id"], r["candidate_id"]].append(r)
max_sum_error = 0.0
for r in candidates:
    group = groups[r["theorem_id"], r["candidate_id"]]
    assert r["tokens"] == sum(x["tokens"] for x in group)
    for source in ("cached", "replay"):
        total = np.sum([x[source] for x in group], axis=0)
        max_sum_error = max(max_sum_error, float(np.max(abs(total - r[source]))))
        assert np.allclose(total / r["tokens"], r[source + "_mean"], rtol=0, atol=1e-12)
    assert np.allclose(
        np.array(r["cached_mean"]) - r["replay_mean"],
        r["cached_minus_replay_mean"],
        rtol=0,
        atol=1e-12,
    )
differences = abs(np.array([r["cached_minus_replay_mean"] for r in candidates]))
summary = read(folder / "summary.json")
assert np.allclose(
    differences.max(0), summary["max_absolute_candidate_mean_difference"], atol=1e-15
)
assert np.allclose(
    differences.mean(0), summary["mean_absolute_candidate_mean_difference"], atol=1e-15
)

# Reconstruct every environment prefix and public feature without labels.
spec = importlib.util.spec_from_file_location(
    "audit_public_features", SEL.parent / "analysis/analyze_candidates.py"
)
feature = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature)
tok = make_tokenizer("lean_staten")
model = read(
    ROOT
    / "artifacts/learned_compute_allocation_20261006/analysis"
    / "features1600_main_dev/feature_model.json"
)
score_rows = rows(ART / "cached_confidence_feature_dev16_v1/candidate_scores.jsonl")
score_lookup = {(r["theorem_id"], r["candidate_id"]): r for r in score_rows}
lookup = {(r["theorem_id"], r["candidate_id"]): r for r in candidates}
checked_prefixes = 0
max_score_error = 0.0
score_vectors = defaultdict(list)
for index, row in enumerate(panel):
    for k in range(8):
        key = row["theorem_id"], k
        base = random.Random(124006 * 1000003 + index * 8 + k).randint(0, 32)
        env = primary.collect.Env(row["prompt"], canon=True, base=base, assign=True)
        effective = []
        records = sorted(groups[key], key=lambda r: r["wave"])
        assert [r["wave"] for r in records] == list(range(len(records)))
        for record in records:
            assert record["prefix"] == primary.collect.prompt_ids(tok, env)
            raw = record["raw_action"]
            assert tok.eos not in raw[:-1]
            assert record["sampled_pad_tokens"] == raw.count(tok.pad)
            checked_prefixes += 1
            action, ended = tok.decode_action(raw)
            if ended:
                ok, _ = env.apply(action)
                effective.append(tok.encode_toks(env._eff) + [tok.eos])
                closed = not ok or env.done or env.steps >= 48
            else:
                effective.append(raw)
                closed = True
            if closed:
                assert record is records[-1]
        assert closed
        candidate = {
            "canonical_name_base": base,
            "effective_actions": effective,
            "rendered_text": tok.text([tok.itos[t] for a in effective for t in a]),
            "generated_tokens": lookup[key]["tokens"],
            "prompt_tokens": sum(len(r["prefix"]) for r in records),
        }
        pair = []
        for source in ("replay", "cached"):
            lp, base_lp, entropy = lookup[key][source + "_mean"]
            candidate.update(mean_logprob=lp, mean_base_logprob=base_lp, mean_entropy=entropy)
            f = feature.public_features(row, candidate, tok, primary.collect.Env)
            value = float(
                model["intercept"]
                + np.dot((np.asarray(f) - model["mean"]) / model["scale"], model["weights"])
            )
            probability = float(1 / (1 + np.exp(-value)))
            max_score_error = max(
                max_score_error,
                abs(value - score_lookup[key][source + "_logit"]),
                abs(probability - score_lookup[key][source + "_probability"]),
            )
            pair.append((value, probability))
        assert (pair[0][1] >= 0.85) == (pair[1][1] >= 0.85)
        score_vectors[row["theorem_id"]].append(pair)
for pairs in score_vectors.values():
    a = np.array(pairs)[:, :, 0]
    assert np.array_equal(
        np.flatnonzero(a[:, 0] == a[:, 0].max()), np.flatnonzero(a[:, 1] == a[:, 1].max())
    )
assert max_score_error < 1e-12

fp = ART / "cached_confidence_fp32_dev8_v1"
for name in ("bf16_identity_raw.json", "fp32_identity_raw.json"):
    identity = read(fp / name)
    assert identity["original"] == identity["instrumented"]
fp_rows = rows(fp / "fp32_action_records.jsonl")
fp_errors = abs(np.array([(np.array(r["cached"]) - r["replay"]) / r["tokens"] for r in fp_rows]))
assert np.allclose(
    fp_errors.max(0), read(fp / "summary.json")["max_absolute_mean_difference"], rtol=0, atol=1e-15
)

# Synthetic wrapper exercise: sampled PAD, done-row callbacks, and noncontiguous original slots.
original = primary.ORIGINAL_PICK
schedule = iter([[2, 0], [1, 0], [2]])
primary.ORIGINAL_PICK = lambda *args: torch.tensor(next(schedule))
try:
    capture = primary.Capture(2)
    for step, slots in enumerate(([0, 1], [0, 1], [1])):
        capture.pick(torch.zeros((len(slots), 3)), 0.8, None, 0, step, 2, torch.tensor(slots))
    synthetic = capture.rows([[2], [0, 0, 2]])
    assert synthetic == read(folder / "cpu_eos_pad_compaction_check.json")["rows"]
finally:
    primary.ORIGINAL_PICK = original
result = {
    "actions": len(actions),
    "candidates": len(candidates),
    "prefixes_reconstructed": checked_prefixes,
    "max_candidate_sum_error": max_sum_error,
    "max_feature_score_error": max_score_error,
    "pool_tie_agreement": 16,
    "threshold_agreement": 128,
    "synthetic_bookkeeping_passed": True,
    "bf16_max_mean_difference": differences.max(0).tolist(),
    "fp32_max_mean_difference": fp_errors.max(0).tolist(),
    "precision_limitation": (
        "Autocast mode explicit in saved source; runtime matmul precision/TF32 flags not captured."
    ),
    "scope": "No model inference, verifier labels, or TEST outcomes read.",
}
Path(__file__).with_name("cached_confidence_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
