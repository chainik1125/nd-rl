import collections, json
from common import *
rl = load_rl(); S = solved_sets(); feat = json.load(open("feat.json"))
gold = [r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
def lev(a, b):
    a, b = a.split(), b.split()
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]
groups = collections.defaultdict(list)
for r in gold:
    if r["body"] <= 14: groups[(src_family(r["src"]), r["body"], r["n_prem"])].append(r)
pairs=[]
for g, recs in groups.items():
    s4 = [r for r in recs if nsolv(r["name"])==4]; s0=[r for r in recs if nsolv(r["name"])==0]
    for a in s4:
        for b in s0:
            pairs.append((lev(a["key"], b["key"]), g, a, b))
pairs.sort(key=lambda p: (p[0]/len(p[2]["key"].split()), p[1][1]))
print("pairs", len(pairs))
seen=set(); shown=0
for d, g, a, b in pairs:
    if b["name"] in seen or shown>=14: continue
    seen.add(b["name"]); shown+=1
    print(f"\n-- dist {d} fam={g[0]} body={g[1]} nprem={g[2]}")
    for tag, r in (("SOLVED", a), ("UNSOLV", b)):
        print(f"  {tag} {r['name']}: {r['key']}")
        print("     ", " ; ".join(ln.split(" : ")[1] + " " + ("|"*int(ln.count('|'))) if False else ln for ln in r["proof"].split(" ; ") if ": PR" not in ln))
