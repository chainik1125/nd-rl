"""All-row arithmetic plus deterministic CPU/Lean sample of first main DEV probe."""

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
P = ART / "context/causal_main_nocost_s101_one_extra_v1"
A = P.with_name(P.name + "_analysis")
os.environ["ND_OFFLINE"] = "1"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
sys.path.insert(0, str(P / "runner_snapshot"))
from controller import Controller  # noqa: E402 - runtime source path
from lean_gate import check_sources  # noqa: E402 - runtime source path
from model import load_ckpt  # noqa: E402 - runtime source path


def rows(p):
    return [json.loads(line) for line in Path(p).open()]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


c = read(P / "config.json")
panel = rows(c["public"])
attempts = {r["attempt_uid"]: r for r in rows(P / "attempts_public.jsonl")}
priv = {r["attempt_uid"]: r for r in rows(P / "attempts_private_unjudged.jsonl")}
valid = read(P / "validity.json")
states = {r["state_id"]: r for r in rows(P / "states_scored.jsonl")}
branches = rows(P / "branches_scored.jsonl")
summary = read(A / "summary.json")
errs = []
checks = collections.Counter()
groups = collections.defaultdict(list)
assert [r["theorem_id"] for r in panel] == c["theorem_ids"]
assert sha(c["public"]) == c["public_sha256"]
assert len(panel) == 64 and collections.Counter(r["family"] for r in panel) == {
    "forward": 32,
    "ore2": 32,
}
assert len(states) == 128 and not any(s["forced"] for s in states.values())
for s in states.values():
    q = s["selection_probabilities"]
    value = sum(p * valid[u] for p, u in zip(q, s["incumbent_ids"], strict=False))
    errs.append(abs(value - s["accept_value"]))
for b in branches:
    s = states[b["state_id"]]
    old = s["incumbent_ids"]
    future = b["future_ids"]
    q = b["after"]["selection_probabilities"]
    om = sum(p * valid[u] for p, u in zip(q[: len(old)], old, strict=False))
    nm = sum(p * valid[u] for p, u in zip(q[len(old) :], future, strict=False))
    errs.extend(
        [
            abs(om - b["old_correct_mass"]),
            abs(nm - b["new_correct_mass"]),
            abs(om + nm - b["continue_value"]),
            abs(om + nm - s["accept_value"] - b["delta_continuation"]),
        ]
    )
    assert b["oracle_before"] == any(valid[u] for u in old)
    assert b["oracle_after"] == any(valid[u] for u in old + future)
    assert b["new_attempt_valid"] == valid[future[0]]
    assert len(future) == b["new_attempt_count"] == 1
    groups[b["state_id"], b["mode"]].append(b)
assert len(branches) == 1280 and all(len(v) == 8 for v in groups.values())
fields = list(summary["pooled_actual_history"])
state_means = {
    k: {
        f: np.mean(
            [
                float(b["after"]["forced"])
                if f == "terminal_hard_limit"
                else 0.0
                if f == "forced_count_shortfall"
                else b[f]
                for b in bs
            ]
        )
        for f in fields
    }
    for k, bs in groups.items()
}
tids = sorted(c["theorem_ids"])
fam = {r["theorem_id"]: r["family"] for r in panel}
rng = np.random.default_rng(81006)
staterrs = []
for f in fields:
    values = np.array(
        [
            np.mean([state_means[f"{tid}/state/{n}", "actual_history"][f] for n in [1, 2]])
            for tid in tids
        ]
    )
    indices = []
    for family in sorted(set(fam.values())):
        ix = np.array([i for i, t in enumerate(tids) if fam[t] == family])
        indices.append(ix[rng.integers(len(ix), size=(5000, len(ix)))])
    ci = np.quantile(values[np.concatenate(indices, axis=1)].mean(1), [0.025, 0.975])
    saved = summary["pooled_actual_history"][f]
    staterrs.extend(
        [abs(values.mean() - saved["mean"]), abs(ci[0] - saved["low"]), abs(ci[1] - saved["high"])]
    )
# Recompute conditional association from state means, independently of analyzer helpers.
x = []
y = []
for count in [1, 2]:
    for available in [False, True]:
        ss = (
            [
                s
                for s in states.values()
                if s["count"] == count and bool(s["oracle_before"]) == available
            ]
            if "oracle_before" in next(iter(states.values()))
            else [
                s
                for s in states.values()
                if s["count"] == count and any(valid[u] for u in s["incumbent_ids"]) == available
            ]
        )
        xx = np.array([s["accept_probability"] for s in ss])
        yy = np.array(
            [state_means[s["state_id"], "actual_history"]["delta_continuation"] for s in ss]
        )
        x.extend(xx - xx.mean())
        y.extend(yy - yy.mean())
correlation = float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))
# Four fixed panel positions, both counts, replicate zero.
# Includes both history modes where present.
selected_ids = {panel[i]["theorem_id"] for i in [0, 1, 32, 33]}
selected_states = [s for s in states.values() if s["theorem_id"] in selected_ids]
selected_branches = [b for b in branches if b["theorem_id"] in selected_ids and b["repeat"] == 0]
torch.set_num_threads(4)
base, tok, _ = load_ckpt(c["base"], "cpu")
model = Controller(base, c["max_attempts"])
model.load_state_dict(torch.load(c["checkpoint"], map_location="cpu", weights_only=True)["state"])
model.eval()
prob_errors = []
uids = set()


@torch.no_grad()
def check_decision(d, uu):
    ids = tok.encode_prompt(priv[uu[0]]["prompt"])
    for i, u in enumerate(uu):
        ids += (
            [tok.stoi["<st>"], tok.stoi[f"n{i + 1}"]]
            + attempts[u]["candidate"]["transition_tokens"]
            + [tok.eos]
        )
    ids += [tok.stoi["<act>"]]
    assert len(ids) == d["controller_tokens"]
    a, q = model.distributions(
        torch.tensor([ids]), torch.tensor([len(ids)]), torch.tensor([len(uu)])
    )
    prob_errors.append(abs(a.probs.item() - d["accept_probability"]))
    prob_errors.extend(
        abs(x - y)
        for x, y in zip(q.probs[0].tolist()[: len(uu)], d["selection_probabilities"], strict=False)
    )
    uids.update(uu)


for s in selected_states:
    check_decision(s, s["incumbent_ids"])
for b in selected_branches:
    check_decision(b["after"], states[b["state_id"]]["incumbent_ids"] + b["future_ids"])
for verdict in [False, True]:
    uids.update(sorted(u for u, v in valid.items() if v == verdict)[:8])
uids = sorted(uids)
sources = [
    tok.statement(priv[u]["prompt"]) + " " + attempts[u]["candidate"]["transition_text"]
    for u in uids
]
flags, _, _ = check_sources(sources)
disagreements = [u for u, f in zip(uids, flags, strict=False) if f != valid[u]]
# Compare panel/draw definitions, not unlike raw accuracies as paired observations.
D = ART / "allocation/no_cost/main_s101_dev200_paths_s52006"
dc = read(D / "config.json")
dp = rows(dc["data"])
dl = rows(D / "private_labels.jsonl")
overlap = set(c["theorem_ids"]) & {r["theorem_id"] for r in dp}
n1 = {s["theorem_id"]: s["accept_value"] for s in states.values() if s["count"] == 1}
comparison = {
    "checkpoint_matches": dc["checkpoint_sha256"] == c["checkpoint_sha256"],
    "causal_panel_theorems": 64,
    "path_panel_theorems": len(dp),
    "overlap_theorems": len(overlap),
    "causal_seed": c["seed"],
    "path_seed": dc["seed"],
    "causal_first_accuracy": float(np.mean(list(n1.values()))),
    "causal_first_on_overlapping32": float(np.mean([n1[t] for t in overlap])),
    "path_first_accuracy": float(np.mean([r["validities"][0] for r in dl])),
}
out = {
    "theorems": 64,
    "states": len(states),
    "branches": len(branches),
    "all_state_branch_arithmetic_max_error": max(errs),
    "all_pooled_means_and_intervals_max_error": max(staterrs),
    "conditional_correlation": correlation,
    "controller_decisions_recomputed": len(selected_states) + len(selected_branches),
    "controller_probability_max_CPU_GPU_error": max(prob_errors),
    "lean_proofs_rechecked": len(uids),
    "lean_disagreements": disagreements,
    "sample_theorem_ids": sorted(selected_ids),
    "sample_proof_ids": uids,
    "panel_comparison": comparison,
    "checkpoint_hash_matches": sha(c["checkpoint"]) == c["checkpoint_sha256"],
    "runner_hashes_match": all(
        sha(P / "runner_snapshot" / n) == h for n, h in c["runner_hashes"].items()
    ),
}
Path(__file__).with_name("main_causal_s101_independent_audit.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps({k: v for k, v in out.items() if k != "sample_proof_ids"}, indent=2))
