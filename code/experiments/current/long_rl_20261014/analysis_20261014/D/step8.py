import os, sys, json, collections, hashlib
os.environ["ND_ARCH"]="untied"; os.environ["LEAN_GATE_LOG"]=os.path.join(os.getcwd(),"lean_gate.jsonl")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(2)
from common import *
rl = load_rl(); S = solved_sets(); gold=[r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
model, tok = M.load(f"{E}/resume/untied-ei_s1/final.pt")
def cls(r):
    lines = parse_proof_tokens(r["proof"].split())
    ores = [ln for ln in lines if ln["rule"] == "ORE" and ln["depth"] == 0]
    if not ores: return "no-top-ORE"
    j, s1, e1, s2, e2 = ores[-1]["refs"]
    h = lambda a,b: sum(1 for ln in lines if a <= ln["idx"] <= b and ln["rule"] in ("ORE","NEGI"))
    h1,h2=h(s1,e1),h(s2,e2)
    return "none" if not h1 and not h2 else "first" if not h2 else "second" if not h1 else "both"
def worst3(r):
    rows = M.token_logp(model, tok, r["prompt"], r["proof"])
    idx = sorted(range(len(rows)), key=lambda i: rows[i][1])[:3]
    return rows, idx
# s first-class: how much of the deficit is the one token?
print("### s/s2 'first' class, body 11-14: total logp, logp without worst token, 2nd/3rd worst tokens")
agg=collections.Counter(); tots=[]; 
for r in [r for r in gold if src_family(r["src"]) in ("s","s2") and 11<=r["body"]<=14 and cls(r)=="first"][:60]:
    rows, idx = worst3(r)
    tot=sum(l for _,l,_ in rows); tots.append((tot, tot-rows[idx[0]][1], tot-rows[idx[0]][1]-rows[idx[1]][1]))
    for k in idx[1:3]: agg[(rows[k][0], rows[k][2], f"{rows[k][1]:.0f}")]+=1
print("mean total", sum(t[0] for t in tots)/len(tots), "without worst", sum(t[1] for t in tots)/len(tots), "without 2 worst", sum(t[2] for t in tots)/len(tots))
print("2nd/3rd worst (gold, argmax, lp):", agg.most_common(8))
# c family: where does gold logp collapse?
print("\n### c family body 13-15 (all unsolved): class counts and worst tokens")
agg=collections.Counter(); ccount=collections.Counter(); tots=[]; ex=[]
for r in [r for r in gold if src_family(r["src"])=="c" and 13<=r["body"]<=15][:80]:
    c=cls(r); ccount[c]+=1
    rows, idx = worst3(r); tot=sum(l for _,l,_ in rows); tots.append(tot)
    i=idx[0]; agg[(rows[i][0], rows[i][2])]+=1
    if len(ex)<6: ex.append(f"{r['name']} {c} tot={tot:.0f} ..." + " ".join(t for t,_,_ in rows[max(0,i-8):i]) + f" [[{rows[i][0]} lp={rows[i][1]:.1f} argmax={rows[i][2]}]]")
print("classes", ccount, "mean total logp", sum(tots)/len(tots))
print("worst", agg.most_common(8)); print("\n".join(ex))
# c-family: proofs with IMPI wrapping: where is the first fun-block? Count 'first'-class generalised to any box rule (IMPI/NEGI/ORE) in first branch
# baseline: s 'none' class body 13 solved by all for comparison
print("\n### s none-class body 13 solved-by-all: mean total logp")
tots=[]
for r in [r for r in gold if src_family(r["src"]) in ("s","s2") and r["body"]==13 and cls(r)=="none" and nsolv(r["name"])==4][:40]:
    rows,_=worst3(r); tots.append(sum(l for _,l,_ in rows))
print(sum(tots)/len(tots))
# old dev schema counts
F=f"{ND_FORK}/data/ladder/transfer.jsonl"
recs=[json.loads(l) for l in open(F)]
old=[r for r in recs if int(hashlib.sha1(r["key"].encode()).hexdigest(),16)%2==0]
ev={arm:jl(f"{A}/long-rl-20261014-v2_{arm}/eval_old_dev.jsonl") for arm in ARMS}
print("\n### old dev: by (source, schema) -> n, never, all")
tab=collections.defaultdict(lambda:[0,0,0])
for i,r in enumerate(old):
    n=sum(ev[a][i]["solved"] for a in ARMS); k=(r.get("source"), r.get("schema"))
    tab[k][0]+=1; tab[k][1]+= n==0; tab[k][2]+= n==4
for k,v in sorted(tab.items(), key=lambda kv:-kv[1][1]): print(k, "n",v[0],"never",v[1],"all",v[2])
