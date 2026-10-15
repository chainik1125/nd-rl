# ruff: noqa: B007,E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Linear probe for box depth (0,1,2,3,4+) on the residual stream after each block, per checkpoint.
Tokens: Lean-accepted EI-s1 proofs of ladder targets (goal spine 1-4) + other RL targets; split by proof 80/20.
Stratified: every depth>=3 token, and the same number per shallower depth. Report test recall per depth, and how
depth-3 tokens are misread. python probe_depth.py [models...] -> out/probe_depth.json
"""

import collections
import json
import random
import sys

import nd
import torch

torch.set_num_threads(6)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
MS = sys.argv[1:] or ["g0_pre", "g0_fin", "g1_pre", "g1_fin", "e1_pre", "e1_fin"]
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
F = {}
for r in map(json.loads, open(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")):
    if r["name"] not in F or r["written"] < F[r["name"]]["written"]:
        F[r["name"]] = r
rng = random.Random(0)
names = sorted(F)
rng.shuffle(names)
tok = nd.load(nd.CKPT[MS[0]])[1]
# pick proofs: enough to get ~4k depth>=3 tokens
recs = []
cnt = collections.Counter()
for n in names:
    f = F[n]
    if len(f["proof"]) > 3000:
        continue
    ids, n_p = nd.encode(tok, f["prompt"], f["proof"], rng.randint(0, 40))
    if len(ids) > 700:
        continue
    L = nd.labels(tok, ids, n_p)
    d = collections.Counter(min(l["depth"], 4) for l in L)
    recs.append((ids, n_p, [min(l["depth"], 4) for l in L]))
    cnt.update(d)
    if len(recs) >= 900:
        break
print("proofs", len(recs), "tokens by depth", dict(sorted(cnt.items())))
split = int(0.8 * len(recs))
# stratified token selection (same positions for every model)
sel = []  # (rec index, token offset in proof, depth)
per = collections.defaultdict(list)
for ri, (ids, n_p, ds) in enumerate(recs):
    for j, d in enumerate(ds):
        per[d].append((ri, j, d))
cap = len(per[3]) + len(per[4])
for d, v in per.items():
    rng.shuffle(v)
    sel += v[:cap]
sel.sort()
print("selected per depth", dict(collections.Counter(d for _, _, d in sel)))
y = torch.tensor([d for _, _, d in sel])
tr = torch.tensor([ri < split for ri, _, _ in sel])


def feats(model):
    X = torch.empty(7, len(sel), nd.P.D)
    by = collections.defaultdict(list)
    for k, (ri, j, _d) in enumerate(sel):
        by[ri].append((k, j))
    for ri, lst in by.items():
        ids, n_p, _ = recs[ri]
        h = nd.resid(model, torch.tensor([ids]))[:, 0]
        for k, j in lst:
            X[:, k] = h[:, n_p + j]
    return X


# decision positions: the `exact` right after a box-closing `) ;`, labelled with the depth k it returns to
pops = []  # (rec index, offset of the `exact` token, k)
for ri, (ids, n_p, ds) in enumerate(recs):
    if ri < split:
        continue
    st = [tok.itos[x] for x in ids]
    for j in range(len(ds) - 1):
        i = n_p + j
        if (
            st[i] == "exact"
            and st[i - 1] == ";"
            and st[i - 2] == ")"
            and j >= 2
            and ds[j] < ds[j - 2]
        ):
            pops.append((ri, j, ds[j]))
print("pop positions by k", dict(collections.Counter(k for _, _, k in pops)))


def pop_feats(model):
    X = torch.empty(7, len(pops), nd.P.D)
    for k, (ri, j, _d) in enumerate(pops):
        ids, n_p, _ = recs[ri]
        X[:, k] = nd.resid(model, torch.tensor([ids[: n_p + j + 1]]))[:, 0, -1]
    return X


def probe(X):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Z = (X - mu) / sd
    W = torch.zeros(Z.shape[1], 5, requires_grad=True)
    b = torch.zeros(5, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z[tr] @ W + b, y[tr]) + 1e-3 * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    probe.last = (mu, sd, W.detach(), b.detach())
    pred = (Z[~tr] @ W + b).argmax(-1)
    yt = y[~tr]
    rec = [float((pred[yt == d] == d).float().mean()) for d in range(5)]
    conf3 = [int((pred[yt == 3] == d).sum()) for d in range(5)]
    return rec, conf3


J = {}
print(f"{'model':>7} {'layer':>5}  recall d0   d1   d2   d3   d4+ | depth-3 tokens read as 0..4")
for m in MS:
    model = nd.load(nd.CKPT[m])[0]
    X = feats(model)
    XP = pop_feats(model)
    kp = torch.tensor([k for _, _, k in pops])
    for l in range(7):
        rec, conf3 = probe(X[l])
        mu, sd, W, b = probe.last
        pp = (((XP[l] - mu) / sd) @ W + b).argmax(-1)
        popread = {k: [int((pp[kp == k] == d).sum()) for d in range(5)] for k in (0, 1, 2, 3)}
        J[f"{m}|{l}"] = {"recall": rec, "conf3": conf3, "pop_read": popread}
        print(
            f"{m:>7} {l:>5}  "
            + " ".join(f"{r:5.0%}" for r in rec)
            + f" | {conf3} | pops k=1,2,3 read as {popread[1]} {popread[2]} {popread[3]}"
        )
json.dump(J, open("out/probe_depth.json", "w"), indent=1)
