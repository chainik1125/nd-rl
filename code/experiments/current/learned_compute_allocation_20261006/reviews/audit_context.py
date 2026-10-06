"""CPU replay and first-action context sensitivity audit; no generation or training."""

import collections
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from model import load_ckpt  # noqa: E402 - runtime source path
from state_env import Env  # noqa: E402 - runtime source path
from state_sample import prompt_ids  # noqa: E402 - runtime source path

base = ROOT / "artifacts/learned_compute_allocation_20261006/context/base_dev48_v1"


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


prior = {
    x["theorem_id"]: x
    for x in read(
        ROOT
        / "artifacts/learned_compute_allocation_20261006/pools/dev_pilot_v2/public_normalized.jsonl"
    )
}
torch.set_num_threads(4)
model, tok, _ = load_ckpt(
    ROOT / "artifacts/learned_compute_allocation_20261006/checkpoints/la_T1_best12_s0_r16.pt"
)
report = {}
for count in [0, 1]:
    rows = read(base / f"history_{count}/public.jsonl")
    labs = {
        r["theorem_id"]: {x["candidate_id"]: x for x in r["labels"]}
        for r in read(base / f"history_{count}/private.jsonl")
    }
    ct = collections.Counter()
    examples = []
    for r in rows:
        old = prior[r["theorem_id"]]["candidates"][0]["effective_tokens"]
        oldnames = {tok.itos[x] for x in old if re.fullmatch("n[0-9]+", tok.itos[x])}
        for c in r["candidates"]:
            e = Env(r["prompt"], canon=True, base=c["canonical_name_base"], assign=True)
            why = "complete"
            for ai, ids in enumerate(c["raw_actions"]):
                a, ended = tok.decode_action(ids)
                if not ended:
                    why = "truncated"
                    break
                ok, why = e.apply(a)
                if not ok:
                    defined = {t for j, t in enumerate(a[:-1]) if a[j + 1] == ":"}
                    names = {t for t in a if re.fullmatch("n[0-9]+", t)} - defined
                    unknown = {n for n in names if e.lookup(n) is None}
                    ct["failure_action_" + str(ai + 1)] += 1
                    if why == "unbound" and unknown & oldnames:
                        ct["unbound_has_prior_name"] += 1
                    if len(examples) < 3:
                        examples.append(
                            {
                                "theorem_id": r["theorem_id"],
                                "candidate_id": c["candidate_id"],
                                "failure_action": ai + 1,
                                "reason": why,
                                "raw_action": " ".join(a),
                                "unknown_names": sorted(unknown),
                                "prior_name_overlap": sorted(unknown & oldnames),
                            }
                        )
                    break
            why = why or ("complete" if e.done else "step_cap")
            label = labs[r["theorem_id"]][c["candidate_id"]]
            ct["valid"] += label["valid"]
            ct["complete"] += label["parse_complete"]
            ct["completion_mismatch"] += e.done != label["parse_complete"]
            ct["reason_" + (why or "complete")] += 1
    report[str(count)] = {"counts": dict(ct), "examples": examples}
spot = []
for row in read(base / "history_1/public.jsonl")[:4]:
    c = row["candidates"][0]
    e = Env(row["prompt"], canon=True, base=c["canonical_name_base"], assign=True)
    state = prompt_ids(tok, e)
    old = prior[row["theorem_id"]]["candidates"][0]["effective_tokens"]
    prefix = [tok.stoi["<st>"], tok.stoi["n1"]] + old + [tok.eos]
    full = prefix + state
    n = len(full)
    p = len(prefix)
    with torch.no_grad():
        clean = model(torch.tensor([state]))[0, -1].float()
        hist = model(torch.tensor([full]))[0, -1].float()
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        mask[p:, :p] = False
        masked = model(torch.tensor([full]), mask=mask[None, None])[0, -1].float()
        q = clean.softmax(-1)
        ph = hist.log_softmax(-1)
        pc = clean.log_softmax(-1)
    spot.append(
        {
            "theorem_id": row["theorem_id"],
            "prefix_tokens": p,
            "clean_top": tok.itos[int(clean.argmax())],
            "history_top": tok.itos[int(hist.argmax())],
            "masked_top": tok.itos[int(masked.argmax())],
            "max_abs_masked_clean_logits": float((masked - clean).abs().max()),
            "KL_clean_to_history": float((q * (pc - ph)).sum()),
        }
    )
report["first_action_logit_spotcheck"] = spot
out = Path(__file__).with_name("context_independent_audit.json")
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
