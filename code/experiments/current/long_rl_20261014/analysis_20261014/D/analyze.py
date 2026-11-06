import collections, json, sys
from common import *

rl = load_rl()
byname = {r["name"]: r for r in rl}
S = solved_sets()
arms = ARMS
gold = [r for r in rl if "proof" in r]
feat = {r["name"]: features(r) for r in gold}
json.dump(feat, open("feat.json", "w"))

def nsolv(n):
    return sum(1 for a in arms if n in S[a])

cnt = collections.Counter(nsolv(r["name"]) for r in rl)
print("targets by #arms solving:", sorted(cnt.items()))
print("\nby band x #arms")
for band in ["0-6", "7-9", "10-14", "15-19", "20-24", "25-34", "35-49", "50+"]:
    recs = [r for r in rl if r["band"] == band]
    c = collections.Counter(nsolv(r["name"]) for r in recs)
    print(f"{band:6s} n={len(recs):5d} " + " ".join(f"{k}:{c[k]:5d}" for k in range(5)))
print("\nby family x #arms (all bands)")
for fam in sorted({src_family(r["src"]) for r in rl}):
    recs = [r for r in rl if src_family(r["src"]) == fam]
    c = collections.Counter(nsolv(r["name"]) for r in recs)
    bodies = sorted(r["body"] for r in recs)
    print(f"{fam:8s} n={len(recs):5d} body med={bodies[len(bodies)//2]:3d} " + " ".join(f"{k}:{c[k]:5d}" for k in range(5)))

# feature comparison within short bands, gold-proof targets only, body <= 19
print("\nFeature means by #arms solving, gold targets, body 7-19, per family")
keys = ["body", "n_prem", "distractors", "max_depth", "n_box", "ore", "negi", "nege", "impi", "ori", "andi", "ande", "bote", "dn", "impe", "reuse", "as_neg"]
for fam in ["s", "s2", "d", "c"]:
    recs = [r for r in gold if src_family(r["src"]) == fam and 7 <= r["body"] <= 19]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[min(nsolv(r["name"]), 4)].append(feat[r["name"]])
    print(f"\n{fam}: n={len(recs)}  " + " ".join(f"{k}:{len(groups[k])}" for k in sorted(groups)))
    print("      " + " ".join(f"{k:>6s}" for k in keys))
    for g in sorted(groups):
        fs = groups[g]
        print(f"  {g}: " + " ".join(f"{sum(f[k] for f in fs)/len(fs):6.2f}" for k in keys))

# never-solved short gold targets
print("\nNever solved, gold, body<=12:")
ns = [r for r in gold if nsolv(r["name"]) == 0 and r["body"] <= 12]
print(len(ns))
for r in sorted(ns, key=lambda r: r["body"])[:25]:
    f = feat[r["name"]]
    print(r["name"], r["body"], src_family(r["src"]), f["n_prem"], "dist", f["distractors"], "ore", f["ore"], "negi", f["negi"], "depth", f["max_depth"], "|", r["prompt"])
