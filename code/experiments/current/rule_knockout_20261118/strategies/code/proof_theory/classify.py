# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import sys, json, gzip, collections, time
from flags import parse_prompt, flags
from g4ip import prove

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
tab = collections.Counter()
t0 = time.time()
out = {}
for l in op(path, "rt"):
    r = json.loads(l)
    prem, concl = parse_prompt(r["prompt"])
    i = prove(prem, concl, budget=300000)
    if i is None:
        lab = "?"
    elif not i:
        lab = "C"
    else:
        m = prove(prem, concl, minimal=True, budget=300000)
        lab = "?" if m is None else ("M" if m else "I")
    rules = ()
    if r.get("proof"):
        rs = set(flags(r["prompt"], r["proof"])["rules"])
        rules = ("DN" if "DN" in rs else "-", "BOTE" if "BOTE" in rs else "-")
    else:
        rules = ("noref", "noref")
    tab[(lab,) + rules] += 1
    out[r["name"]] = lab
print(path, "secs", round(time.time() - t0))
for k, v in sorted(tab.items()):
    print("  ", k, v)
json.dump(out, open(sys.argv[2], "w"))
