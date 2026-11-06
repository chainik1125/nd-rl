import os, sys, json, collections, time
os.environ["ND_ARCH"]="untied"; os.environ["LEAN_GATE_WORKERS"]="2"; os.environ["LEAN_GATE_LOG"]=os.path.join(os.getcwd(),"lean_gate.jsonl")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe")
import ndmodel as M, torch
torch.set_num_threads(3)
from common import *
rl = load_rl(); S = solved_sets(); byname={r["name"]:r for r in rl}
gold = [r for r in rl if "proof" in r]
def nsolv(n): return sum(1 for a in ARMS if n in S[a])
def branch_cls(r):
    lines = parse_proof_tokens(r["proof"].split())
    ores = [ln for ln in lines if ln["rule"] == "ORE" and ln["depth"] == 0]
    if not ores: return None
    j, s1, e1, s2, e2 = ores[-1]["refs"]
    h = lambda a,b: sum(1 for ln in lines if a <= ln["idx"] <= b and ln["rule"] in ("ORE","NEGI"))
    h1,h2=h(s1,e1),h(s2,e2)
    return "none" if not h1 and not h2 else "first" if not h2 else "second" if not h1 else "both"
model, tok = M.load(f"{E}/resume/untied-ei_s1/final.pt")
# 1. what the model must emit for a 'first' gold proof
r = byname["rl_11321"]; ids = tok.encode_proof(r["proof"]); print("LEAN TEXT (first-class, rl_11321):\n", tok.last_text, "\n")
r = byname["rl_10517"]; ids = tok.encode_proof(r["proof"]); print("LEAN TEXT (second-class, rl_10517):\n", tok.last_text, "\n")
# 2. gold logp by class, s family body 11-14; find the min-logp token position pattern
res = collections.defaultdict(list); worst = collections.defaultdict(collections.Counter)
t0=time.time()
for r in gold:
    if src_family(r["src"]) not in ("s","s2") or not (11 <= r["body"] <= 14): continue
    c = branch_cls(r)
    if c not in ("first","second"): continue
    rows = M.token_logp(model, tok, r["prompt"], r["proof"])
    tot = sum(l for _,l,_ in rows); mn = min(rows, key=lambda x:x[1])
    i = rows.index(mn)
    ctx = " ".join(t for t,_,_ in rows[max(0,i-6):i]) + " [[" + mn[0] + f" lp={mn[1]:.1f} argmax={mn[2]}]]"
    res[c].append((tot, len(rows), mn[1]))
    worst[c][(mn[0], mn[2])] += 1
    if len(res[c]) <= 4: print(c, r["name"], nsolv(r["name"]), f"total={tot:.1f} n={len(rows)} min={mn[1]:.1f}", ctx)
print("time", time.time()-t0)
for c in res:
    v=res[c]; print(c, "n", len(v), "mean total logp", sum(x[0] for x in v)/len(v), "mean per-token", sum(x[0] for x in v)/sum(x[1] for x in v), "mean min", sum(x[2] for x in v)/len(v))
    print("   worst-token (gold tok, argmax):", worst[c].most_common(6))
