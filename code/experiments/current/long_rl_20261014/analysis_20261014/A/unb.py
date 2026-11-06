import json, sys, re, collections
def kind(raw):
    toks = raw.split(); seen = set(); hdr = set(); out = collections.Counter()
    for i, t in enumerate(toks):
        if not re.fullmatch(r"[nh]\d+", t): continue
        prev = toks[i - 1] if i else ""
        if prev == "have": hdr.add(t); seen.add(t); continue
        if prev == "(" and i + 1 < len(toks) and toks[i + 1] == ":": seen.add(t); continue
        if t.startswith("h"): continue
        if t not in seen: out["never-defined"] += 1
    return out
c = collections.Counter()
for fn in sys.argv[1:]:
    for r in json.load(open(fn)):
        for s in r["samples"]:
            if s["nd"].startswith("LEANPARSE unbound"):
                k = kind(s["raw"])
                c["never-defined name" if k["never-defined"] else "defined-but-out-of-scope (incl. self/closed box)"] += 1
print(c)
