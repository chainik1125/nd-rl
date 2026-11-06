"""Interventions: rewrite premises to remove a structural need, keep the theorem logically equivalent.
 curry:   premise ((A & B) > C)  ->  (A > (B > C))   (recursively for nested conj antecedents)  [c family]
 nand:    premise ~(A & B)       ->  (A > ~B)                                                     [d family]
"""
import json, random, sys
sys.path.insert(0, ".")
from feat import parse_prompt, show
def curry(f):
    if isinstance(f, tuple) and f[0] == ">" and isinstance(f[1], tuple) and f[1][0] == "&":
        a, b = f[1][1], f[1][2]
        return curry((">", a, (">", b, f[2])))
    return f
def nand(f):
    if isinstance(f, tuple) and f[0] == "~" and isinstance(f[1], tuple) and f[1][0] == "&":
        return (">", f[1][1], ("~", f[1][2]))
    return f
def rewrite(prompt, fn):
    prems, goal = parse_prompt(prompt)
    new = [fn(p) for p in prems]
    return "THM " + " , ".join(show(p) for p in new) + " SEQ " + show(goal) + " PRF", sum(a != b for a, b in zip(prems, new))
d = [json.loads(l) for l in open("feats2.jsonl")]
rng = random.Random(0)
c = [r for r in d if r["pool"] == "rl" and r["src"] == "c_f20.jsonl" and r["body"] <= 16 and r.get("g_andi_to_impe")]
dd = [r for r in d if r["pool"] == "rl" and r["src"] == "d_f20.jsonl" and 9 <= r["body"] <= 13 and r.get("g_andi_to_nege") and not r["solved"]]
rng.shuffle(c); rng.shuffle(dd)
c, dd = c[:30], dd[:30]
out = {"c_orig": [r["prompt"] for r in c], "c_curry": [rewrite(r["prompt"], curry)[0] for r in c],
       "d_orig": [r["prompt"] for r in dd], "d_nand": [rewrite(r["prompt"], nand)[0] for r in dd]}
print([rewrite(r["prompt"], nand)[1] for r in dd]); print([rewrite(r["prompt"], curry)[1] for r in c])
print(out["c_orig"][0]); print(out["c_curry"][0]); print(out["d_orig"][0]); print(out["d_nand"][0])
for k, v in out.items(): json.dump(v, open(f"int_{k}.json", "w"))
json.dump({"c": [r["name"] for r in c], "d": [r["name"] for r in dd]}, open("int_names.json", "w"))
