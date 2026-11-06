import collections, json, itertools
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

# ---- hypothesis tests at matched body length, within family
def goal_of(r): return r["key"].split("|-")[1].strip()
def parse_lines(r):
    return [ln for ln in parse_proof_tokens(r["proof"].split()) if ln["rule"] != "PR"]
def h_negi_intermediate(r):
    """gold has a NEGI whose conclusion is not the theorem goal (an intermediate negated lemma)"""
    goal = r["prompt"].split("SEQ")[1].replace("PRF","").strip()
    return any(ln["rule"]=="NEGI" and fmt_formula(ln["formula"]) != goal for ln in parse_lines(r))
def h_bote(r): return feat[r["name"]]["bote"] > 0
def h_nested_ore(r):
    """an ORE whose branches sit at depth>=2 (ORE inside another box)"""
    return any(ln["rule"]=="ORE" and ln["depth"]>=1 for ln in parse_lines(r))
def h_goal_neg(r): return goal_of(r).startswith("( ~")
def h_neg_as(r): return feat[r["name"]]["as_neg"]>0
def h_depth3(r): return feat[r["name"]]["max_depth"]>=3
def h_reuse(r): return feat[r["name"]]["reuse"]>0
def h_dn(r): return feat[r["name"]]["dn"]>0
def h_nege_on_neg(r):
    """NEGE where the contradicted formula is itself a negation (~~ pattern)"""
    return any(ln["rule"]=="NEGE" and (ln["formula"][0]=="not") for ln in parse_lines(r)) or any(ln["rule"]=="AS" and ln["formula"][0]=="not" and ln["formula"][1][0]=="not" for ln in parse_lines(r))
def h_impe_on_neg_prem(r):
    """an IMPE whose major premise is (~A > B): must first derive ~A (a negation) to fire it"""
    lines = parse_lines(r); by={ln["idx"]:ln for ln in parse_proof_tokens(r["proof"].split())}
    return any(ln["rule"]=="IMPE" and by[ln["refs"][1]]["formula"][0]=="not" for ln in lines)
HYP = dict(negi_intermediate=h_negi_intermediate, bote=h_bote, nested_ore=h_nested_ore, goal_neg=h_goal_neg, neg_as=h_neg_as, depth3=h_depth3, reuse=h_reuse, dn=h_dn, nege_on_neg=h_nege_on_neg, impe_minor_is_neg=h_impe_on_neg_prem)
for fam in ["s+s2", "d", "c"]:
    recs = [r for r in gold if (src_family(r["src"]) in fam.split("+")) and 7 <= r["body"] <= 16]
    print(f"\n### family {fam} body 7-16, n={len(recs)}: solve-by-all rate (and never-solved rate) with / without feature, matched by body")
    for hname, h in HYP.items():
        bybody = collections.defaultdict(lambda: {True: [], False: []})
        for r in recs: bybody[r["body"]][bool(h(r))].append(nsolv(r["name"]))
        rows=[]
        for b in sorted(bybody):
            w, wo = bybody[b][True], bybody[b][False]
            if len(w)>=5 and len(wo)>=5:
                rows.append((b, len(w), sum(x==4 for x in w)/len(w), sum(x==0 for x in w)/len(w), len(wo), sum(x==4 for x in wo)/len(wo), sum(x==0 for x in wo)/len(wo)))
        if not rows: 
            nw = sum(h(r) for r in recs); print(f"  {hname:20s} no matched rows (with={nw})"); continue
        print(f"  {hname:20s} " + " | ".join(f"b{b}: with n={nw} all={a4:.2f} none={a0:.2f} / without n={nwo} all={b4:.2f} none={b0:.2f}" for b,nw,a4,a0,nwo,b4,b0 in rows))

# ---- minimal pairs: same family, same body, same skeleton, 4 vs 0 solved, small key edit distance
print("\n### minimal pairs (same family+body+skeleton; one solved by all, other by none)")
groups = collections.defaultdict(list)
for r in gold:
    if r["body"] <= 20: groups[(src_family(r["src"]), r["body"], feat[r["name"]]["skel"])].append(r)
pairs=[]
for g, recs in groups.items():
    s4 = [r for r in recs if nsolv(r["name"])==4]; s0=[r for r in recs if nsolv(r["name"])==0]
    for a in s4:
        for b in s0:
            pairs.append((lev(a["key"], b["key"]), g, a, b))
pairs.sort(key=lambda p: (p[0], p[1][1]))
print("mixed skeleton groups:", sum(1 for g,recs in groups.items() if any(nsolv(r["name"])==4 for r in recs) and any(nsolv(r["name"])==0 for r in recs)), "pairs:", len(pairs))
seen=set()
for d, g, a, b in pairs[:40]:
    if b["name"] in seen: continue
    seen.add(b["name"])
    print(f"\n-- dist {d} fam={g[0]} body={g[1]}\n  SOLVED {a['name']}: {a['key']}\n  UNSOLV {b['name']}: {b['key']}")
json.dump([(d, g, a["name"], b["name"]) for d,g,a,b in pairs[:200]], open("pairs.json","w"))
