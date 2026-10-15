# ruff: noqa: E501,E741,SIM115,UP031  (research scripts: long result lines, bare open(), short loop names)
"""At each pop decision (the `exact` after a box-closing `) ;`, which must be followed by the name of the `have` that
opened the just-closed box), how much does each head attend to the correct opener's `have nX` (name token and the token
after it) vs the opener one level further out ('up')? Spine-3 ladder proofs written by EI s1.
python attn_pop.py [n] [models] -> out/attn_pop.json"""

import collections
import contextlib
import io
import json
import random
import sys

import nd
import torch

torch.set_num_threads(6)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as S
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MS = sys.argv[2:] or ["g0_pre", "g0_fin", "g1_pre", "g1_fin", "e1_pre", "e1_fin"]
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
F = {}
for r in map(json.loads, open(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")):
    if r["name"] not in F or r["written"] < F[r["name"]]["written"]:
        F[r["name"]] = r
names = sorted(
    n for n, r in R.items() if r["src"].startswith("ladder") and S.spine(S.goal(r)) == 3 and n in F
)
names = random.Random(0).sample(names, N)


def pops(tok, ids, n_p):
    """[(j = position of `exact`, k, pos of correct opener's name token, pos of up opener's name token or None)]"""
    s = [tok.itos[x] for x in ids]
    ev, stack, opener = [], [], []
    hv = {}  # depth -> last `have` seen at that depth
    for i in range(n_p, len(s)):
        t = s[i]
        if t == "have":
            hv[len(opener)] = i + 1
        if t == "(":
            box = s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            if box:
                opener.append(hv[len(opener)])
        elif t == ")" and stack and stack.pop():
            opener.pop()
            if s[i + 1] == ";" and s[i + 2] == "exact":
                k = len(opener)
                if k >= 1:
                    ev.append((i + 2, k, opener[-1], opener[-2] if k >= 2 else None))
    return ev


tok = nd.load(nd.CKPT[MS[0]])[1]
J = {}
for m in MS:
    model = nd.load(nd.CKPT[m])[0]
    acc = collections.defaultdict(lambda: torch.zeros(6, 8))
    cnt = collections.Counter()
    for n in names:
        f = F[n]
        ids, n_p = nd.encode(tok, f["prompt"], f["proof"], 0)
        x = torch.tensor([ids])
        att = nd.attn_maps(model, x)  # L,H,T,T
        for j, k, good, up in pops(tok, ids, n_p):
            g = att[:, :, j, good] + att[:, :, j, good + 1]
            acc[(k, "good")] += g
            if up is not None:
                acc[(k, "up")] += att[:, :, j, up] + att[:, :, j, up + 1]
            # the top-level opener (depth 0 have) too
            cnt[k] += 1
    print(
        f"\n== {m}: mean attention from pop `exact` to opener name tokens (good / up), per layer x head"
    )
    for k in (1, 2):
        G = acc[(k, "good")] / cnt[k]
        U = acc[(k, "up")] / cnt[k] if (k, "up") in acc else None
        print(f" k={k} (n={cnt[k]})  good:")
        for l in range(6):
            print("   L%d " % l + " ".join(f"{v:4.2f}" for v in G[l].tolist()))
        if U is not None:
            print(f" k={k} up:")
            for l in range(6):
                print("   L%d " % l + " ".join(f"{v:4.2f}" for v in U[l].tolist()))
        J[f"{m}|{k}|good"] = G.tolist()
        if U is not None:
            J[f"{m}|{k}|up"] = U.tolist()
json.dump(J, open("out/attn_pop.json", "w"))
