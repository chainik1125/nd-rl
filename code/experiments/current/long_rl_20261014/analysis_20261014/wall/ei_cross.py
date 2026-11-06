import json, collections, sys
sys.argv = ["x"]
exec(open("spine.py").read().split("print(\"ladder")[0])
for run in ("ei_s1",):
    f = F[run]; c = collections.Counter()
    for r in lad:
        if spine(goal(r)) == 3 and r["name"] in f: c[f[r["name"]]] += 1
    cum = 0
    for k in sorted(c): cum += c[k]; print(run, "call", k, "+", c[k], "cum", cum)
