import json, sys, collections
import feat
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
R = {r["name"]: r for r in map(json.loads, open(f"{E}/data/rl.jsonl"))}
def split_imp(f):
    f = f.strip()
    if feat.top(f) != ">": return None
    inner = f[1:-1].strip().split(); d = 0
    for i, t in enumerate(inner):
        if t == "(": d += 1
        elif t == ")": d -= 1
        elif d == 0 and t == ">": return " ".join(inner[:i]), " ".join(inner[i+1:])
def spine(f):
    n = 0
    while (s := split_imp(f)): n += 1; f = s[1]
    return n
goal = lambda r: r["prompt"].split(" SEQ ")[1].replace(" PRF", "")
def first(fn):
    d = {}
    for r in map(json.loads, open(fn)):
        d[r["name"]] = min(d.get(r["name"], 10**9), r["round"])
    return d
runs = {"grpo_s0": f"{A}/v3/early/untied-grpo_s0.found_targets.jsonl", "ei_s1": f"{A}/v2/early/untied-ei_s1.found_targets.jsonl"}
import os
if os.path.exists(f"{A}/v3/early/untied-grpo_s1.found_targets.jsonl"): runs["grpo_s1"] = f"{A}/v3/early/untied-grpo_s1.found_targets.jsonl"
F = {k: first(v) for k, v in runs.items()}
lad = [r for r in R.values() if r["src"].startswith("ladder")]
print("ladder targets by goal implication-spine length: solved by grpo_s0 before call 84 / by end; others by end")
c = collections.defaultdict(collections.Counter)
for r in lad:
    s = min(spine(goal(r)), 5)
    c[s]["n"] += 1
    c[s]["s0<84"] += F["grpo_s0"].get(r["name"], 10**9) < 84
    for k in F: c[s][k] += r["name"] in F[k]
for s in sorted(c): print(s, dict(c[s]))
