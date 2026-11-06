import os, sys, json, collections, time, hashlib
os.environ["ND_ARCH"]="untied"; os.environ["LEAN_GATE_WORKERS"]="2"; os.environ["LEAN_GATE_LOG"]=os.path.join(os.getcwd(),"lean_gate.jsonl")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(3)
from common import *
rl = load_rl(); S = solved_sets(); byname={r["name"]:r for r in rl}
gold = [r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
model, tok = M.load(f"{E}/resume/untied-ei_s1/final.pt")
for n in ["rl_11321","rl_10517","rl_5198","rl_13846"]:
    r=byname[n]; ids=tok.encode_proof(r["proof"]); tok.decode(ids); print(f"LEAN TEXT {n} (solved by {nsolv(n)}):\n ", tok.last_text, "\n")
# d family: localisation
def dcls(r):
    lines = parse_proof_tokens(r["proof"].split())
    ores = [ln for ln in lines if ln["rule"] == "ORE" and ln["depth"] == 0]
    goal = r["prompt"].split("SEQ")[1].replace("PRF","").strip()
    inter = any(ln["rule"]=="NEGI" and fmt_formula(ln["formula"]) != goal for ln in lines)
    if not ores: return None
    j, s1, e1, s2, e2 = ores[-1]["refs"]
    h = lambda a,b: sum(1 for ln in lines if a <= ln["idx"] <= b and ln["rule"] in ("ORE","NEGI"))
    h1,h2=h(s1,e1),h(s2,e2)
    pos = "none" if not h1 and not h2 else "first" if not h2 else "second" if not h1 else "both"
    return pos + ("+inter" if inter else "")
res = collections.defaultdict(list); worst = collections.defaultdict(collections.Counter); ex=collections.defaultdict(list)
for r in gold:
    if src_family(r["src"]) != "d" or not (9 <= r["body"] <= 11): continue
    c = dcls(r)
    if c is None: continue
    rows = M.token_logp(model, tok, r["prompt"], r["proof"])
    tot = sum(l for _,l,_ in rows); mn = min(rows, key=lambda x:x[1]); i = rows.index(mn)
    res[c].append((tot, len(rows), mn[1], nsolv(r["name"])))
    worst[c][(mn[0], mn[2])] += 1
    if len(ex[c])<2: ex[c].append(r["name"]+" "+" ".join(t for t,_,_ in rows[max(0,i-8):i]) + " [[" + mn[0] + f" lp={mn[1]:.1f} argmax={mn[2]}]]")
print("### d family body 9-11, class -> n, mean total logp, mean min logp, solve-by-all, never")
for c in sorted(res):
    v=res[c]; print(f"{c:14s} n={len(v):3d} tot={sum(x[0] for x in v)/len(v):6.1f} min={sum(x[2] for x in v)/len(v):6.1f} all={sum(x[3]==4 for x in v)/len(v):.2f} none={sum(x[3]==0 for x in v)/len(v):.2f}", worst[c].most_common(4))
    for e in ex[c]: print("     ", e)
# old dev 9-liners
F=f"{ND_FORK}/data/ladder/transfer.jsonl"
recs=[json.loads(l) for l in open(F)]
old=[r for r in recs if int(hashlib.sha1(r["key"].encode()).hexdigest(),16)%2==0]
ev={arm:jl(f"{A}/long-rl-20261014-v2_{arm}/eval_old_dev.jsonl") for arm in ARMS}
print("\n### old dev: body 9 by #arms, rules field")
cnt=collections.defaultdict(collections.Counter)
for i,r in enumerate(old):
    b=r["n_lines"]-r["n_prem"]; n=sum(ev[a][i]["solved"] for a in ARMS)
    if b in (8,9,10,11): cnt[(b, "none" if n==0 else "all" if n==4 else "some")][str(r.get("rules"))[:60]]+=1
for k in sorted(cnt): print(k, cnt[k].most_common(5))
print("examples never-solved body 9:")
k=0
for i,r in enumerate(old):
    b=r["n_lines"]-r["n_prem"]; n=sum(ev[a][i]["solved"] for a in ARMS)
    if b==9 and n==0 and k<8: k+=1; print("  ", r["prompt"], r.get("rules"), r.get("source"), r.get("schema"))
print("examples solved-by-all body 9:")
k=0
for i,r in enumerate(old):
    b=r["n_lines"]-r["n_prem"]; n=sum(ev[a][i]["solved"] for a in ARMS)
    if b==9 and n==4 and k<4: k+=1; print("  ", r["prompt"], r.get("rules"), r.get("source"), r.get("schema"))
