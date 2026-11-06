import json, sys, collections
sys.path.insert(0, ".")
from feat import parse_proof
d = [json.loads(l) for l in open("feats2.jsonl")]
def flags(p):
    L = parse_proof(p)
    ores = [l for l in L if l["rule"] == "ORE" and len(l["refs"]) == 5]
    first = second = False
    for o in ores:
        _, a1, e1, a2, e2 = o["refs"]
        for l in ores:
            if l is o: continue
            if a1 < l["idx"] <= e1: first = True
            if a2 < l["idx"] <= e2: second = True
    return first, second
tab = collections.defaultdict(lambda: [0, 0])
for r in d:
    if r["pool"] not in ("rl", "dev") or not r.get("gold") or r["src"] == "ladder": continue
    f, s = flags(r["gold"])
    if not (f or s): key = "no nesting"
    elif f: key = "nested in FIRST branch"
    else: key = "nested in second branch only"
    b = "7-14" if r["body"] <= 14 else "15-19" if r["body"] <= 19 else "20-24" if r["body"] <= 24 else "25-34" if r["body"] <= 34 else "35+"
    t = tab[(r["pool"], r["src"][:3].rstrip("_"), b, key)]; t[0] += r["solved"]; t[1] += 1
for k in sorted(tab):
    s, n = tab[k]
    if n >= 10: print(k, f"{s}/{n} = {s/n:.2f}")
