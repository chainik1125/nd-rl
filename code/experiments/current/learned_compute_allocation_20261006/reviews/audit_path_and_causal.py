"""Independent CPU scoring/arithmetic audit of development protocol smokes."""

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
os.environ["ND_OFFLINE"] = "1"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
from lean_gate import check_sources  # noqa: E402 - runtime source path
from tokenizer import make_tokenizer  # noqa: E402 - runtime source path

tok = make_tokenizer("lean_staten")


def read(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


P = ART / "allocation/live_path_protocol_smoke_s52006"
paths = read(P / "public_paths.jsonl")
labs = {(r["theorem_id"], r["repeat"]): r["validities"] for r in read(P / "private_labels.jsonl")}
saved = json.loads((P / "integrated.json").read_text())
patherrs = []
src = []
ys = []
for path in paths:
    key = (path["theorem_id"], path["repeat"])
    valid = labs[key]
    for c, y in zip(path["candidates"], valid, strict=False):
        src.append(tok.statement(path["prompt"]) + " " + c["transition_text"])
        ys.append(y)
    for method, records in saved.items():
        if method not in ["learned", "fixed1", "fixed2", "fixed4", "fixed8"]:
            continue
        target = next(v for v in records if (v["theorem_id"], v["repeat"]) == key)
        reach = 1.0
        acc = 0.0
        attempt = 0.0
        gentok = 0.0
        for n, (d, c) in enumerate(zip(path["decisions"], path["candidates"], strict=False), 1):
            h = d["accept_probability"] if method == "learned" else float(n >= int(method[5:]))
            h = 1.0 if d["forced"] or n == len(valid) else h
            acc += (
                reach
                * h
                * sum(p * y for p, y in zip(d["selection_probabilities"], valid[:n], strict=False))
            )
            attempt += reach
            gentok += reach * c["generated_tokens"]
            reach *= 1 - h
        patherrs.extend(
            [
                abs(acc - target["correct_probability"]),
                abs(attempt - target["attempts"]),
                abs(gentok - target["expected_policy_work"]["generated_tokens"]),
            ]
        )
flags, wall, _ = check_sources(src)
report = {
    "path_smoke": {
        "paths": len(paths),
        "proofs_rechecked": len(src),
        "Lean_disagreements": sum(a != b for a, b in zip(flags, ys, strict=False)),
        "max_quality_attempt_generated_token_error": max(patherrs),
        "seconds": wall,
    }
}
P = ART / "context/causal_protocol_smoke_v2"
attempts = {r["attempt_uid"]: r for r in read(P / "attempts_public.jsonl")}
private = {r["attempt_uid"]: r for r in read(P / "attempts_private_unjudged.jsonl")}
valid = json.loads((P / "validity.json").read_text())
states = {s["state_id"]: s for s in read(P / "states_scored.jsonl")}
branches = read(P / "branches_scored.jsonl")
errs = []
ct = collections.Counter()
paired = {}
for s in states.values():
    labels = [valid[u] for u in s["incumbent_ids"]]
    A = sum(p * y for p, y in zip(s["selection_probabilities"], labels, strict=False))
    errs.append(abs(A - s["accept_value"]))
    ct["forced_states"] += s["forced"]
for b in branches:
    s = states[b["state_id"]]
    q = b["after"]["selection_probabilities"]
    ys = [valid[u] for u in s["incumbent_ids"]] + [valid[b["future_id"]]]
    old = sum(p * y for p, y in zip(q[:-1], ys[:-1], strict=False))
    new = q[-1] * ys[-1]
    errs.extend(
        [
            abs(sum(q) - 1),
            abs(old - b["old_correct_mass"]),
            abs(new - b["new_correct_mass"]),
            abs(old + new - b["continue_value"]),
            abs(old + new - s["accept_value"] - b["delta_one"]),
        ]
    )
    ct["oracle_decrease"] += b["oracle_before"] and not b["oracle_after"]
    ct["nonextendible_state_used"] += s["forced"]
    paired.setdefault((b["state_id"], b["repeat"]), {})[b["mode"]] = b
for modes in paired.values():
    if len(modes) == 2:
        a, b = modes["actual_history"], modes["history_removed"]
        ct["paired_intervention_branches"] += 1
        ct["paired_seed_mismatch"] += a["seed"] != b["seed"]
        ca = attempts[a["future_id"]]["candidate"]
        cb = attempts[b["future_id"]]["candidate"]
        ct["paired_namebase_mismatch"] += ca["canonical_name_base"] != cb["canonical_name_base"]
        n = min(len(ca["generation_action_seeds"]), len(cb["generation_action_seeds"]))
        ct["paired_actionseed_prefix_mismatch"] += (
            ca["generation_action_seeds"][:n] != cb["generation_action_seeds"][:n]
        )
src = [
    tok.statement(private[u]["prompt"]) + " " + r["candidate"]["transition_text"]
    for u, r in attempts.items()
]
flags, wall, _ = check_sources(src)
report["causal_smoke"] = {
    "states": len(states),
    "branches": len(branches),
    "attempts": len(attempts),
    "saved_valid": sum(valid.values()),
    "proofs_rechecked": len(src),
    "Lean_disagreements": sum(f != valid[u] for f, u in zip(flags, attempts, strict=False)),
    "max_decomposition_probability_error": max(errs),
    "checks": dict(ct),
    "seconds": wall,
    "scope": "uniform untrainedheads; protocol only",
}
Path(__file__).with_name("path_causal_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
