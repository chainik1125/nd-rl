"""Independent public-feature and DEV threshold recomputation; no final-test access."""

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
from state_env import Env  # noqa: E402 - runtime source path
from tokenizer import make_tokenizer  # noqa: E402 - runtime source path

tok = make_tokenizer("lean_staten")


def read(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


f = json.loads((ART / "analysis/features1600_main_dev/feature_model.json").read_text())
report = {}
X = []


def features(row, c):
    e = Env(row["prompt"], canon=True, base=c["canonical_name_base"], assign=True)
    accepted = 0
    for a in c["raw_actions"]:
        tokens, ended = tok.decode_action(a)
        if not ended:
            break
        ok, _ = e.apply(tokens)
        if not ok:
            break
        accepted += 1
        if e.done:
            break
    n = len(c["raw_actions"])
    words = c["rendered_text"].split()
    return [
        c["mean_base_logprob"],
        c["mean_logprob"],
        c["mean_entropy"],
        math.log1p(c["generated_tokens"]),
        math.log1p(n),
        math.log1p(c["prompt_tokens"]),
        float(e.done and accepted == n),
        accepted / max(n, 1),
        words.count("exact") / max(n, 1),
        words.count("Or.elim") / max(n, 1),
    ]


train = read(ART / "pools/main_train/public.jsonl")
for r in train:
    for c in r["candidates"]:
        X.append(features(r, c))
x = np.array(X)
report["training_feature_recompute"] = {
    "theorems": len(train),
    "candidates": len(X),
    "mean_max_error": float(np.max(np.abs(x.mean(0) - f["mean"]))),
    "scale_max_error": float(np.max(np.abs(np.maximum(x.std(0), 1e-6) - f["scale"]))),
    "training_id_match": set(f["training_theorem_ids"]) == {r["theorem_id"] for r in train},
}
private = {
    r["theorem_id"]: {c["candidate_id"]: c for c in r["labels"]}
    for r in read(ART / "pools/main_dev/private.jsonl")
}
acc = []
thracc = []
thrN = []
thrgen = []
thrpref = []
completed = collections.Counter()
for r in read(ART / "pools/main_dev/public.jsonl"):
    cs = r["candidates"]
    ys = [private[r["theorem_id"]][c["candidate_id"]]["valid"] for c in cs]
    fs = [features(r, c) for c in cs]
    z = (np.array(fs) - f["mean"]) / f["scale"]
    logits = z @ f["weights"] + f["intercept"]
    p = 1 / (1 + np.exp(-logits))
    best = np.flatnonzero(logits == max(logits))
    acc.append(sum(ys[i] for i in best) / len(best))
    qual = np.flatnonzero(p >= 0.85)
    m = len(qual)
    if m:
        chosen = qual
        N = 9 / (m + 1)
        inc = [1 / m if i in qual else 1 / (m + 1) for i in range(8)]
    else:
        chosen = best
        N = 8
        inc = [1] * 8
    thracc.append(sum(ys[i] for i in chosen) / len(chosen))
    thrN.append(N)
    thrgen.append(sum(w * c["generated_tokens"] for w, c in zip(inc, cs, strict=False)))
    thrpref.append(sum(w * c["prompt_tokens"] for w, c in zip(inc, cs, strict=False)))
    for v, y in zip(fs, ys, strict=False):
        completed[f"complete{int(v[6])}_valid{int(y)}"] += 1
report["dev_recompute"] = {
    "theorems": len(acc),
    "feature_accuracy": float(np.mean(acc)),
    "completion_table": dict(completed),
    "threshold": 0.85,
    "threshold_accuracy": float(np.mean(thracc)),
    "threshold_attempts": float(np.mean(thrN)),
    "threshold_reward_lambda03": float(np.mean(thracc) - 0.03 * (np.mean(thrN) - 1)),
    "threshold_generated_tokens_proxy": float(np.mean(thrgen)),
    "threshold_prefill_tokens_proxy": float(np.mean(thrpref)),
}
Path(__file__).with_name("feature_baseline_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
