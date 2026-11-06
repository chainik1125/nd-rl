import collections, json, gzip, random
from common import *
rl = load_rl(); S = solved_sets(); feat = json.load(open("feat.json"))
gold = [r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
def show(r):
    print(f"== {r['name']} body={r['body']} fam={src_family(r['src'])} solved_by={[a for a in ARMS if r['name'] in S[a]]}")
    print(r["prompt"]); 
    for ln in r["proof"].split(" ; "): print("   ", ln)
print("### d family never-solved 9-liners")
for r in [r for r in gold if src_family(r["src"])=="d" and nsolv(r["name"])==0 and r["body"]==9][:2]: show(r)
print("### d family solved-by-all, body>=9")
for r in [r for r in gold if src_family(r["src"])=="d" and nsolv(r["name"])==4 and r["body"]>=9][:2]: show(r)
print("### c family never solved, shortest")
for r in sorted([r for r in gold if src_family(r["src"])=="c" and nsolv(r["name"])==0], key=lambda r:r["body"])[:2]: show(r)
print("### c family solved")
for r in sorted([r for r in gold if src_family(r["src"])=="c" and nsolv(r["name"])>=2], key=lambda r:r["body"])[:2]: show(r)
print("### s family never-solved shortest")
for r in sorted([r for r in gold if src_family(r["src"]) in("s","s2") and nsolv(r["name"])==0], key=lambda r:r["body"])[:3]: show(r)

# pretraining rule frequencies
print("### pretraining rule / structure stats")
rules = collections.Counter(); n=0; depth=collections.Counter(); nprem=collections.Counter(); asneg=0; bodyc=collections.Counter()
with gzip.open(f"{E}/data/pretrain.jsonl.gz","rt") as f:
    for i,l in enumerate(f):
        if i>=200000: break
        r=json.loads(l); n+=1
        try: fe=features(r)
        except Exception as e: continue
        for k in ["ore","negi","nege","impi","ori","andi","ande","bote","dn","impe"]: rules[k]+=fe[k]
        depth[fe["max_depth"]]+=1; nprem[fe["n_prem"]]+=1; asneg+=fe["as_neg"]; bodyc[fe["body"]]+=1
print("n",n,"rules per proof",{k:round(v/n,3) for k,v in rules.items()},"as_neg/proof",round(asneg/n,3))
print("depth",sorted(depth.items())); print("nprem",sorted(nprem.items())); print("body",sorted(bodyc.items()))
print("pretrain keys:", list(r.keys()))
