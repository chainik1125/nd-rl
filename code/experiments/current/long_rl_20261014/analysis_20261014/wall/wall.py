"""python wall.py <found_targets.jsonl> [lo hi]: targets by first-solved round window, by family/band and proof features."""
import json, sys, collections
import feat
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
R = {r["name"]: r for r in map(json.loads, open(f"{E}/data/rl.jsonl"))}
fam = lambda r: "ladder" if r["src"].startswith("ladder") else r["src"].replace("_f20.jsonl", "").replace(".jsonl", "")
found = collections.defaultdict(list)
for r in map(json.loads, open(sys.argv[1])):
    found[r["name"]].append(r)
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (84, 102)
first = {n: min(x["round"] for x in v) for n, v in found.items()}
grp = lambda n: "never" if n not in first else "pre" if first[n] < lo else "wall" if first[n] <= hi else "post"
G = collections.defaultdict(list)
for n, r in R.items():
    G[grp(n)].append(r)
print({k: len(v) for k, v in G.items()})
def tab(key, groups=("pre", "wall", "post", "never")):
    c = {g: collections.Counter(key(r) for r in G[g]) for g in groups}
    keys = sorted(set().union(*c.values()), key=str)
    print(f"{'':>22}" + "".join(f"{g:>9}" for g in groups))
    for k in keys:
        print(f"{str(k):>22}" + "".join(f"{c[g][k]:>9}" for g in groups))
tab(lambda r: fam(r))
tab(lambda r: r["band"])
tab(lambda r: r["n_prem"])
def goaltop(r):
    return feat.top(r["prompt"].split(" SEQ ")[1].replace(" PRF", ""))
tab(goaltop)
# features of the model's own (shortest) found proof, wall vs pre
def best(n):
    return min(found[n], key=lambda x: x["written"])
for g in ("pre", "wall", "post"):
    rs = collections.Counter(); deps = collections.Counter(); n = 0; F = 0; wr = []
    for r in G[g]:
        f = feat.feats(r, best(r["name"])["proof"]); n += 1
        for k in ("ORE", "NEGI", "NEGE", "IMPI", "ORI1", "ORI2", "ANDI", "R", "BOTE", "DN"):
            rs[k] += f[k] > 0
        rs.update({"other:" + o: 1 for o in f["other_rules"]})
        deps[f["maxdepth"]] += 1; F += f["has_F"]; wr.append(f["body"])
    print(g, n, "share of proofs using rule:", {k: round(v / max(n, 1), 2) for k, v in sorted(rs.items())})
    print("   maxdepth", dict(sorted(deps.items())), "has F", round(F / max(n, 1), 2), "mean written body", round(sum(wr) / max(n, 1), 1))
