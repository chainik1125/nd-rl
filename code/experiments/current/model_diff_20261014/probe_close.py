# ruff: noqa: E501,E741  (research scripts: long result lines, bare open(), short loop names)
"""Ridge probes at pop tokens (the `exact` after a box close) for r (close index in the run), D_start (depth before the
run = k + r) and k (true depth after the close), per layer, per model. Split by record. Report test R^2 and, for k,
accuracy of the rounded prediction on 'off-diagonal' pops (k != max(0, 2 - r)), where the pretrained heuristic is wrong.
python probe_close.py [models]"""

import collections
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(6)
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
MS = sys.argv[1:] or ["g0_pre", "g0_fin", "g1_fin", "e1_pre", "e1_fin"]
C = torch.load("out/corpus.pt")
starts = C["starts"].tolist() + [len(C["ids"])]
itos = C["itos"]
ids = C["ids"].long()
pk = C["pop_k"].long()
recs = [
    i
    for i in range(len(starts) - 1)
    if starts[i + 1] - starts[i] <= 700 and (pk[starts[i] : starts[i + 1]] >= 1).any()
]
random.Random(1).shuffle(recs)
recs = recs[:2500]
ev = []
for i in recs:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    p = pk[starts[i] : starts[i + 1]]
    for j in (p >= 0).nonzero().flatten().tolist():
        r, q = 1, j - 2
        while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
            r += 1
            q -= 4
        ev.append((i, j, int(p[j]), r))
split = set(recs[: int(0.75 * len(recs))])
tr = torch.tensor([e[0] in split for e in ev])
K = torch.tensor([e[2] for e in ev]).float()
Rr = torch.tensor([e[3] for e in ev]).float()
Ds = K + Rr
off = torch.tensor([e[2] != max(0, 2 - e[3]) for e in ev])
print("pops", len(ev), "off-diagonal", int(off.sum()), "test off-diag", int((off & ~tr).sum()))
by = collections.defaultdict(list)
for n_, e in enumerate(ev):
    by[e[0]].append((n_, e[1]))


def ridge(X, y, lam=10.0):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Z = (X - mu) / sd
    Zt = torch.cat([Z[tr], torch.ones(int(tr.sum()), 1)], 1)
    w = torch.linalg.solve(Zt.T @ Zt + lam * torch.eye(Zt.shape[1]), Zt.T @ y[tr])
    pred = torch.cat([Z, torch.ones(len(Z), 1)], 1) @ w
    te = ~tr
    r2 = 1 - ((pred[te] - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return float(r2), pred


for mname in MS:
    run, st = mname.split("_")
    m = Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[run]}/{'pretrain' if st == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    X = torch.empty(7, len(ev), 384)
    with torch.no_grad():
        for i, lst in by.items():
            h = m.resid(ids[starts[i] : starts[i + 1]][None])[:, 0]
            for n_, j in lst:
                X[:, n_] = h[:, j]
    print(
        f"\n{mname}: test R^2 for r / D_start / k, and k-probe accuracy on off-diagonal test pops (heuristic-wrong)"
    )
    for l in range(1, 7):
        r2r, _ = ridge(X[l], Rr)
        r2d, _ = ridge(X[l], Ds)
        r2k, pk_ = ridge(X[l], K)
        te = ~tr & off
        acc_off = float((pk_[te].round() == K[te]).float().mean())
        heur = float((pk_[te].round() == (2 - Rr[te]).clamp_min(0)).float().mean())
        print(
            f"  after block {l}: R2 r {r2r:5.2f}  D_start {r2d:5.2f}  k {r2k:5.2f} | off-diag k-probe = true k {acc_off:4.0%}, = heuristic {heur:4.0%}"
        )
