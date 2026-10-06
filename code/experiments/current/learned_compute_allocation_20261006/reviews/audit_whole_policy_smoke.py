"""Independent CPU audit of the resumed-policy DEV protocol smoke."""

import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
P = ART / "context" / (sys.argv[1] if len(sys.argv) > 1 else "causal_policy_protocol_smoke_v3")
os.environ["ND_OFFLINE"] = "1"
sys.path.insert(0, str(ART / "source/nd-takehome-dan"))
sys.path.insert(0, str(P / "runner_snapshot"))
from controller import Controller  # noqa: E402 - runtime source path
from lean_gate import check_sources  # noqa: E402 - runtime source path
from model import load_ckpt  # noqa: E402 - runtime source path


def read(f):
    return [json.loads(line) for line in (P / f).read_text().splitlines()]


config = json.loads((P / "config.json").read_text())
attempts = {r["attempt_uid"]: r for r in read("attempts_public.jsonl")}
priv = {r["attempt_uid"]: r for r in read("attempts_private_unjudged.jsonl")}
valid = json.loads((P / "validity.json").read_text())
states = {r["state_id"]: r for r in read("states_scored.jsonl")}
branches = read("branches_scored.jsonl")
checks = collections.Counter()
errors = []
prob_errors = []


def keyed(seed, *parts):
    return int(hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()[:15], 16)


def ids_for(uids):
    prompt = priv[uids[0]]["prompt"]
    ids = tok.encode_prompt(prompt)
    for i, uid in enumerate(uids):
        ids += (
            [tok.stoi["<st>"], tok.stoi[f"n{i + 1}"]]
            + attempts[uid]["candidate"]["transition_tokens"]
            + [tok.eos]
        )
    return ids + [tok.stoi["<act>"]]


torch.set_num_threads(4)
base, tok, _ = load_ckpt(ROOT / config["base"])
model = Controller(base, config["max_attempts"])
ck = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=False)
model.load_state_dict(ck["state"])
model.eval()
del ck


@torch.no_grad()
def check_decision(d, uids):
    ids = ids_for(uids)
    checks["controller_length_mismatch"] += len(ids) != d["controller_tokens"]
    forced = (
        len(uids) == config["max_attempts"]
        or len(ids) + config["attempt_bound"] + config["state_reserve"] + 3 > config["context"]
    )
    checks["forced_flag_mismatch"] += forced != d["forced"]
    a, q = model.distributions(
        torch.tensor([ids]), torch.tensor([len(ids)]), torch.tensor([len(uids)])
    )
    prob_errors.append(abs(a.probs.item() - d["accept_probability"]))
    prob_errors.extend(
        abs(x - y)
        for x, y in zip(
            q.probs[0].tolist()[: len(uids)], d["selection_probabilities"], strict=False
        )
    )
    checks["decisions_recomputed"] += 1


for s in states.values():
    check_decision(s, s["incumbent_ids"])
    A = sum(
        q * valid[u] for q, u in zip(s["selection_probabilities"], s["incumbent_ids"], strict=False)
    )
    errors.append(abs(A - s["accept_value"]))
for b in branches:
    s = states[b["state_id"]]
    future = b["future_ids"]
    path = b["decision_path"]
    checks["future_count_mismatch"] += len(future) != len(path) or b["new_attempt_count"] != len(
        future
    )
    rng = random.Random(keyed(b["seed"], "stop_decisions"))
    checks["multifuture_branches"] += len(future) > 1
    for j, (uid, d) in enumerate(zip(future, path, strict=False), 1):
        ids = s["incumbent_ids"] + future[:j]
        check_decision(d, ids)
        checks["candidate_identity_mismatch"] += (
            attempts[uid]["candidate"]["candidate_id"] != s["count"] + j - 1
        )
        expected_seed = b["seed"] if j == 1 else keyed(b["seed"], "extra", j)
        checks["generation_seed_mismatch"] += expected_seed != d["generation_seed"]
        u = rng.random()
        checks["stop_uniform_mismatch"] += u != d["stop_uniform"]
        force_continue = j < config["forced_extra"] and not d["forced"]
        accept = d["forced"] or (not force_continue and u < d["accept_probability"])
        checks["forced_continuation_mismatch"] += force_continue != d["forced_continuation"]
        checks["stop_decision_mismatch"] += accept != d["accepted"]
        checks["nonterminal_accept_or_terminal_continue"] += (j < len(path) and accept) or (
            j == len(path) and not accept
        )
    q = b["after"]["selection_probabilities"]
    old = sum(p * valid[u] for p, u in zip(q[: s["count"]], s["incumbent_ids"], strict=False))
    new = sum(p * valid[u] for p, u in zip(q[s["count"] :], future, strict=False))
    errors.extend(
        [
            abs(old - b["old_correct_mass"]),
            abs(new - b["new_correct_mass"]),
            abs(old + new - b["continue_value"]),
            abs(old + new - s["accept_value"] - b["delta_continuation"]),
        ]
    )
    checks["oracle_decrease"] += b["oracle_before"] and not b["oracle_after"]
sources = [
    tok.statement(priv[u]["prompt"]) + " " + a["candidate"]["transition_text"]
    for u, a in attempts.items()
]
flags, wall, _ = check_sources(sources)
checks["Lean_disagreement"] = sum(f != valid[u] for f, u in zip(flags, attempts, strict=False))
out = {
    "states": len(states),
    "branches": len(branches),
    "proofs_rechecked": len(sources),
    "valid_proofs": sum(valid.values()),
    "checks": dict(checks),
    "max_selection_value_arithmetic_error": max(errors),
    "max_CPU_vs_GPU_controller_probability_error": max(prob_errors),
    "runner_hashes_match": all(
        hashlib.sha256((P / "runner_snapshot" / n).read_bytes()).hexdigest() == h
        for n, h in config["runner_hashes"].items()
    ),
    "checkpoint_hash_matches": hashlib.sha256(
        (ROOT / config["checkpoint"]).read_bytes()
    ).hexdigest()
    == config["checkpoint_sha256"],
    "scope": "two-theorem DEV integration smoke only",
}
Path(__file__).with_name(
    "forced4_smoke_independent_audit.json"
    if config["forced_extra"] == 4
    else "whole_policy_smoke_independent_audit.json"
).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
