import collections, json
from common import *
rl = load_rl(); S = solved_sets(); feat = json.load(open("feat.json"))
gold = [r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
def branch_info(r):
    """For the LAST (outermost, top-level) ORE in the gold proof: (lines in branch1, lines in branch2, hard rules in b1, in b2)."""
    lines = parse_proof_tokens(r["proof"].split()); by = {ln["idx"]: ln for ln in lines}
    ores = [ln for ln in lines if ln["rule"] == "ORE" and ln["depth"] == 0]
    if not ores: return None
    o = ores[-1]; j, s1, e1, s2, e2 = o["refs"]
    b1 = [ln for ln in lines if s1 <= ln["idx"] <= e1]; b2 = [ln for ln in lines if s2 <= ln["idx"] <= e2]
    hard = lambda b: sum(1 for ln in b if ln["rule"] in ("ORE", "NEGI"))
    return len(b1), len(b2), hard(b1), hard(b2)
for fam in ["s+s2", "d"]:
    print(f"\n### {fam}: top-level ORE; where is the nested box rule (ORE/NEGI)? solve-by-all / never, by body")
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in gold:
        if src_family(r["src"]) not in fam.split("+") or r["body"] > 16: continue
        bi = branch_info(r)
        if bi is None: continue
        l1, l2, h1, h2 = bi
        cls = "none" if h1 == 0 and h2 == 0 else "first" if h2 == 0 else "second" if h1 == 0 else "both"
        tab[r["body"]][cls].append(nsolv(r["name"]))
    for b in sorted(tab):
        print(f"  body {b:2d}: " + " | ".join(f"{c:6s} n={len(v):4d} all={sum(x==4 for x in v)/len(v):.2f} none={sum(x==0 for x in v)/len(v):.2f}" for c, v in sorted(tab[b].items())))
    print("  also: longer first branch (l1>l2) vs shorter, among 'none'-class")
    tab2 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in gold:
        if src_family(r["src"]) not in fam.split("+") or r["body"] > 16: continue
        bi = branch_info(r)
        if bi is None or bi[2] or bi[3]: continue
        tab2[r["body"]]["b1 longer" if bi[0] > bi[1] else "b1 shorter" if bi[0] < bi[1] else "equal"].append(nsolv(r["name"]))
    for b in sorted(tab2):
        print(f"  body {b:2d}: " + " | ".join(f"{c:10s} n={len(v):4d} all={sum(x==4 for x in v)/len(v):.2f} none={sum(x==0 for x in v)/len(v):.2f}" for c, v in sorted(tab2[b].items())))
