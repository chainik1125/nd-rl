"""Swap the disjuncts of the OUTER case-split premise so the nested split moves from the first branch to the second."""
import json, random, sys, collections
sys.path.insert(0, ".")
from feat import parse_prompt, parse_proof, show
d = [json.loads(l) for l in open("feats2.jsonl")]
def info(r):
    L = parse_proof(r["gold"]); by = {l["idx"]: l for l in L}
    last = L[-1]
    if last["rule"] != "ORE": return None
    disj, a1, e1, a2, e2 = last["refs"]
    inner_first = any(l["rule"] == "ORE" and a1 < l["idx"] <= e1 for l in L)
    inner_second = any(l["rule"] == "ORE" and a2 < l["idx"] <= e2 for l in L)
    if by[disj]["rule"] != "PR": return None
    return disj, inner_first, inner_second
pool = [r for r in d if r["pool"] == "rl" and r["src"] in ("s_f20.jsonl", "s2_f20.jsonl") and 12 <= r["body"] <= 14 and r["n_ore"] == 2]
c = collections.Counter()
for r in pool:
    i = info(r)
    if i: c[(i[1], i[2], r["solved"])] += 1
print("(inner in first branch, inner in second, solved): count", sorted(c.items()))
sel = [r for r in pool if (i := info(r)) and i[1] and not i[2] and not r["solved"]]
random.Random(0).shuffle(sel); sel = sel[:30]
orig, swp = [], []
for r in sel:
    prems, goal = parse_prompt(r["prompt"])
    k = info(r)[0] - 1
    p = list(prems); assert p[k][0] == "v"; p[k] = ("v", p[k][2], p[k][1])
    orig.append(r["prompt"]); swp.append("THM " + " , ".join(show(x) for x in p) + " SEQ " + show(goal) + " PRF")
json.dump(orig, open("int_s_orig.json", "w")); json.dump(swp, open("int_s_swap.json", "w"))
print(len(sel)); print(orig[0]); print(swp[0])
