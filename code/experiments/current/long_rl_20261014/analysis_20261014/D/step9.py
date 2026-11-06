import json, collections
from common import *
rl = load_rl(); S = solved_sets(); gold=[r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
BOX = ("ORE","NEGI","IMPI")
def classify(r):
    lines = parse_proof_tokens(r["proof"].split()); by={ln["idx"]:ln for ln in lines}
    goal = r["prompt"].split("SEQ")[1].replace("PRF","").strip()
    boxfirst = False
    for o in [ln for ln in lines if ln["rule"]=="ORE"]:
        j,s1,e1,s2,e2 = o["refs"]
        if any(s1<=ln["idx"]<=e1 and ln["rule"] in BOX for ln in lines): boxfirst=True
    inter = any(ln["rule"]=="NEGI" and fmt_formula(ln["formula"]) != goal and ln["depth"]>=1 for ln in lines)
    classical = any(ln["rule"]=="DN" for ln in lines)
    if r["body"] >= 20: return "E long (>=20)"
    if boxfirst: return "A box in 1st ORE branch"
    if classical: return "C classical (DN)"
    if inter: return "B intermediate ~lemma"
    return "D other"
tab = collections.defaultdict(lambda: collections.Counter()); fam_tab=collections.defaultdict(collections.Counter)
for r in gold:
    c = classify(r); n = nsolv(r["name"])
    tab[c]["n"]+=1; tab[c]["never"]+= n==0; tab[c]["all4"]+= n==4; tab[c]["some"]+= 0<n<4
    if n==0: fam_tab[c][src_family(r["src"])]+=1
print("### taxonomy over 13,946 gold RL targets (first matching class)")
for c in sorted(tab): print(f"{c:28s} n={tab[c]['n']:5d} never={tab[c]['never']:5d} ({tab[c]['never']/tab[c]['n']:.0%}) some={tab[c]['some']:4d} all4={tab[c]['all4']:5d}  never by family: {dict(fam_tab[c])}")
print("\n### same, restricted to body 7-14 (where length alone is not the story)")
tab = collections.defaultdict(lambda: collections.Counter())
for r in gold:
    if not 7<=r["body"]<=14: continue
    c = classify(r); n = nsolv(r["name"]); tab[c]["n"]+=1; tab[c]["never"]+= n==0; tab[c]["all4"]+= n==4
for c in sorted(tab): print(f"{c:28s} n={tab[c]['n']:5d} never={tab[c]['never']:5d} ({tab[c]['never']/tab[c]['n']:.0%}) all4={tab[c]['all4']:5d} ({tab[c]['all4']/tab[c]['n']:.0%})")
print("\n### 'D other' never-solved, body 7-14: by family and body")
c2=collections.Counter()
for r in gold:
    if 7<=r["body"]<=14 and classify(r)=="D other" and nsolv(r["name"])==0: c2[(src_family(r["src"]), r["body"])]+=1
print(sorted(c2.items()))
# dev set the same way
dev = load_dev(); DS = dev_solved()
tab = collections.defaultdict(lambda: collections.Counter())
for r in dev:
    if "proof" not in r: continue
    c = classify(r); n = sum(1 for a in ARMS if DS[a][r["name"]]>0); tab[c]["n"]+=1; tab[c]["never"]+= n==0; tab[c]["all4"]+= n==4
print("\n### dev.jsonl (k=64 eval) same taxonomy")
for c in sorted(tab): print(f"{c:28s} n={tab[c]['n']:4d} never={tab[c]['never']:4d} ({tab[c]['never']/tab[c]['n']:.0%}) all4={tab[c]['all4']:4d}")
# d-probe items: inter or not
d=json.load(open("probe_swap_results.json")); byname={r["name"]:r for r in rl}
print("\n### d-family swap items: inter?, untied orig/swap n_ok")
for p in d["probes"]:
    if p["fam"]!="d": continue
    r=byname[p["name"]]; lines=parse_proof_tokens(r["proof"].split()); goal=r["prompt"].split("SEQ")[1].replace("PRF","").strip()
    inter = any(ln["rule"]=="NEGI" and fmt_formula(ln["formula"]) != goal for ln in lines)
    print(p["name"], p["tag"], "inter" if inter else "direct", d["results"][p["name"]+"/orig"]["n_ok"], d["results"][p["name"]+"/swap"]["n_ok"])
# failure modes of negi_inter / excluded_middle
res=json.load(open("probe_synth_results.json"))
for k in ["negi_inter/0","negi_inter/1","excluded_middle/0","peirce/0","impi_first/0","impi_second/1"]:
    r=res[k]; print("\n", k, r["prompt"], collections.Counter(t[:25] for t in r["tags"])); print("  RAW:", r["raw"][0][-420:])
