"""Print a few accepted trained proofs that cite no premise, and ones with repeated identical lines."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import rd, body_lines, used_lines, E
rows = [r for r in rd(os.path.join(E, "run_lr3e-5/evaluate/dumps/long_pool.jsonl")) if r["ok"]]
np, rep = [], []
for r in rows:
    ls = body_lines(r["completion"]); u = used_lines(ls)
    pr = [l["idx"] for l in ls if l["rule"] == "PR"]
    if pr and not set(pr) & u: np.append(r)
    k = [(l["depth"], l["formula"], l["rule"], tuple(l["refs"])) for l in ls]
    if len(set(k)) < len(k): rep.append(r)
print("no-premise distinct prompts", len({r["prompt"] for r in np}))
for r in {r["prompt"]: r for r in np}.values():
    print(r["prompt"], "\n   ", r["completion"])
print("\nrepeated-line examples")
for r in rep[:3]: print(r["prompt"], "\n   ", r["completion"])
