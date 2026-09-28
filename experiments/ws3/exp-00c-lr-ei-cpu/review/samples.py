"""Reviewer sample reads for exp-00c (own code). Run from code/:

    uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/samples.py

Writes:
  samples_random.txt   20 random rows per model (frozen + 4 arms) from the evaluate dumps
                       (long_pool, greedy_heldout, textbook72 concatenated), seed 20260927
  long_proofs.txt      every DISTINCT accepted proof (after renumbering lines) whose problem has
                       length >= 8 (eval-set L_true, or textbook reference_lines) or whose proof
                       has >= 8 lines, from every eval dump and every training completion
  structure.json       per-rule-set lift on eval-set L7 (rules of the generator's own proof),
                       and rule-set counts in the pool, to see where the lift comes from
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "ws3"))
import ndpath  # noqa: E402,F401

from strict_verify import safe_verify  # noqa: E402

sys.path.insert(0, HERE)
from recount_lib import read_jsonl, renumber  # noqa: E402

ARMS = ["lr3e-5", "lr1e-4", "lr3e-4", "ei"]
MODELS = {"base": os.path.join(EXP, "run_lr3e-5", "baseline", "dumps")}
MODELS.update({a: os.path.join(EXP, f"run_{a}", "evaluate", "dumps") for a in ARMS})
PARTS = ["long_pool.jsonl.gz", "greedy_heldout.jsonl.gz", "textbook72.jsonl.gz"]
TB = {r["name"]: r for f in ("textbook_dev.jsonl", "textbook_train.jsonl")
      for r in read_jsonl(os.path.join(ndpath.TEXTBOOK72, f))}
rl = {r["prompt"]: r for r in read_jsonl(os.path.join(ndpath.FORK_DATA, "rl_targets.jsonl"))}

# ---- 20 random per model
rng = random.Random(20260927)
lines = []
for m, d in MODELS.items():
    rows = []
    for p in PARTS:
        rows += read_jsonl(os.path.join(d, p))
    idx = sorted(rng.sample(range(len(rows)), 20))
    lines.append(f"==================== {m}: rows {idx} of {len(rows)} (long_pool, greedy, textbook concatenated)")
    for i in idx:
        r = rows[i]
        ok, reason, n = safe_verify(r["prompt"], r["completion"])
        lines.append(f"--- [{i}] {r['label']} {r['name']} len={r['length']} mine={ok} ({reason}) stored={r['ok']}")
        lines.append(r["prompt"])
        lines.append(r["completion"])
open(os.path.join(HERE, "samples_random.txt"), "w").write("\n".join(lines) + "\n")

# ---- distinct accepted proofs at length >= 8
seen = {}
for m, d in MODELS.items():
    for p in PARTS:
        for r in read_jsonl(os.path.join(d, p)):
            L = r["length"] if r["length"] is not None else -1
            ok, _, n = safe_verify(r["prompt"], r["completion"])
            if ok and (L >= 8 or n >= 8):
                key = (r["prompt"], renumber(r["completion"]))
                seen.setdefault(key, {"where": Counter(), "L": L, "n": n, "name": r["name"]})
                seen[key]["where"][f"{m}/{p.split('.')[0]}"] += 1
for a in ARMS:
    for r in read_jsonl(os.path.join(EXP, f"run_{a}", "completions.jsonl.gz")):
        if r["reward"] < 0.5:
            continue
        ok, _, n = safe_verify(r["prompt"], r["completion"])
        L = int(rl[r["prompt"]]["L_true"]) if r["prompt"] in rl else 6
        if ok and (L >= 8 or n >= 8):
            key = (r["prompt"], renumber(r["completion"]))
            seen.setdefault(key, {"where": Counter(), "L": L, "n": n, "name": "train-pool"})
            seen[key]["where"][f"{a}/train"] += 1
out = [f"{len(seen)} distinct accepted proofs with problem length >= 8 or >= 8 proof lines",
       f"by problem length: {dict(Counter(v['L'] for v in seen.values()))}",
       f"by proof lines: {dict(Counter(v['n'] for v in seen.values()))}",
       f"distinct problems: {len({k[0] for k in seen})}", ""]
for (prompt, proof), v in sorted(seen.items(), key=lambda x: (x[1]["name"], x[0])):
    out.append(f"--- {v['name']} L={v['L']} lines={v['n']} seen in {dict(v['where'])}")
    out.append(prompt)
    out.append(proof)
open(os.path.join(HERE, "long_proofs.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[:4]))

# ---- structure of the L7 lift: by generator proof rule set
es = read_jsonl(os.path.join(EXP, "eval_set.jsonl"))
pool = read_jsonl(os.path.join(EXP, "pool.jsonl"))


def rules_of(prompt):
    r = rl.get(prompt)
    if not r:
        return "L6:train"
    if r.get("rules"):
        return "+".join(sorted(set(r["rules"]) - {"PR", "AS"}))
    return f"schema:{r.get('schema')}"


rate = {}
for m, d in MODELS.items():
    g = defaultdict(list)
    for r in read_jsonl(os.path.join(d, "long_pool.jsonl.gz")):
        g[r["prompt"]].append(safe_verify(r["prompt"], r["completion"])[0])
    rate[m] = {p: sum(v) / len(v) for p, v in g.items()}
struct = {"pool_rule_sets": dict(Counter(rules_of(r["prompt"]) for r in pool).most_common()), "eval_L7_by_rule_set": {}, "eval_L8_by_rule_set": {}}
for L, key in ((7, "eval_L7_by_rule_set"), (8, "eval_L8_by_rule_set")):
    groups = defaultdict(list)
    for r in es:
        if int(r["L_true"]) == L:
            groups[rules_of(r["prompt"])].append(r["prompt"])
    for rs, ps in sorted(groups.items(), key=lambda x: -len(x[1])):
        struct[key][rs] = {"n": len(ps), **{m: round(sum(rate[m][p] for p in ps) / len(ps), 3) for m in MODELS}}
json.dump(struct, open(os.path.join(HERE, "structure.json"), "w"), indent=1)
print(json.dumps(struct, indent=1))
