"""Matched distractor check (2026-10-09): 200 random rows of the k12_distract10 slice and their undistracted K12 originals,
pass@32 on Robbie's frozen lean_seq base, split by distractor count, proof length and total premise count.
    cd code && uv run python pretrain_data/matched_distract.py > ../data/pretrain_data/screens/matched_distract.log
Writes data/pretrain_data/screens/matched_distract.json. Finding: the wall is the prompt's premise count, not premise use:
distracted problems with 4-5 total premises 0 of 71 solved, their originals 0.83; with <= 3 total premises 0.74-1.00."""
import sys, os, json, random, collections
sys.path[:0] = ["/Users/cyu/Desktop/github/nd-rl/code/pretrain_data", "/Users/cyu/Desktop/github/nd-rl/code/textbook_problem"]
import pool_common as PC
import screen_ckpt as SC
import eval_ckpt as EC
import torch
from strict_verify import safe_verify
rng = random.Random(0)
rows = [r for r in PC.read(os.path.join(PC.DATA, "pools", "scan", "k12_distract10.jsonl.gz")) if r.get("slice")]
samp = rng.sample(rows, 200)
k12 = {r["prompt"]: r for r in PC.load_k12()}
base = [k12[r["base_prompt"]] for r in samp]
model, tok, meta, fork_sample, lean_tok = EC.load_robbie(SC.RUN, "pretrain", "cpu")
def hits(rs, k=32, seed=0):
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    prompts = [tok.encode_prompt(r["prompt"]) for r in rs]
    flat = [(i, j) for i in range(len(rs)) for j in range(k)]
    order = sorted(range(len(flat)), key=lambda x: len(prompts[flat[x][0]]))
    outs = [None] * len(flat)
    for s in range(0, len(order), 256):
        chunk = order[s:s+256]
        res = fork_sample.generate_ids(model, tok, [prompts[flat[x][0]] for x in chunk], greedy=False, temperature=1.0, max_new=400, gen=gen)
        for x, o in zip(chunk, res): outs[x] = o
    h = collections.Counter()
    for (i, j), o in zip(flat, outs):
        body = tok.decode(o)
        if not body.startswith("LEANPARSE") and safe_verify(rs[i]["prompt"], body)[0]: h[i] += 1
    return [h[i] for i in range(len(rs))]
hd = hits(samp); hb = hits(base)
n = len(samp)
def p(hs, k=32): return round(sum(hs)/(n*k), 3), round(sum(1 for x in hs if x > 0)/n, 3)
print("distracted pass@1/pass@32", p(hd)); print("same problems undistracted", p(hb))
both = sum(1 for a, b in zip(hd, hb) if a > 0 and b > 0); only_b = sum(1 for a, b in zip(hd, hb) if a == 0 and b > 0); only_d = sum(1 for a, b in zip(hd, hb) if a > 0 and b == 0); neither = n - both - only_b - only_d
print(f"n={n}: solved both {both}, undistracted only {only_b}, distracted only {only_d}, neither {neither}")
for k in (1, 2):
    idx = [i for i, r in enumerate(samp) if r["n_distract"] == k]
    print(f"k={k} distractors: n={len(idx)} distracted pass@32 {sum(1 for i in idx if hd[i]>0)/len(idx):.2f} undistracted {sum(1 for i in idx if hb[i]>0)/len(idx):.2f} per-sample {sum(hd[i] for i in idx)/(32*len(idx)):.2f} vs {sum(hb[i] for i in idx)/(32*len(idx)):.2f}")
byL = collections.defaultdict(lambda: [0,0,0])
for i, r in enumerate(samp):
    b = "2-6" if r["n_lines"] <= 7 else "7-14"
    byL[b][0] += 1; byL[b][1] += hd[i] > 0; byL[b][2] += hb[i] > 0
print({b: f"n={v[0]} distracted {v[1]/v[0]:.2f} undistracted {v[2]/v[0]:.2f}" for b, v in byL.items()})
json.dump(dict(n=n, hd=hd, hb=hb, k=[r["n_distract"] for r in samp], lines=[r["n_lines"] for r in samp]), open("/Users/cyu/Desktop/github/nd-rl/data/pretrain_data/screens/matched_distract.json", "w"))
