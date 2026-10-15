# ruff: noqa: E402,E501  (research scripts: long result lines, bare open(), short loop names)
"""Do the rotated decoders point along depth? Ridge probes on g0_fin's (and g0_pre's) layer-L residual at close-run
tokens for D_start, r and k_after (= depth after this close). Cosine between each probe weight and Delta d_j =
d_j^fin - d_j^pre (raw space) for the contribution-ranked latents, vs 500 random alive latents (null).
python align.py runs/g0_L3"""

import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(8)
import importlib.util

spec = importlib.util.spec_from_file_location("it", "interpret_top.py")
run = sys.argv[1]
cc = torch.load(f"{run}/cc.pt", map_location="cpu")
L = cc["args"]["layer"]
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
M = [
    Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[n.split('_')[0]]}/{'pretrain' if n.endswith('pre') else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    for n in cc["names"]
]
C = torch.load("out/corpus.pt")
starts = C["starts"].tolist() + [len(C["ids"])]
itos = C["itos"]
ids = C["ids"].long()
npr = C["n_prompt"].tolist()
W_dec, scale = cc["W_dec"], cc["scale"]
recs = [
    i
    for i in range(len(starts) - 1)
    if C["src"][i] in (C["SRCS"].index("lad3"), C["SRCS"].index("ei1"), C["SRCS"].index("gold"))
]
random.Random(0).shuffle(recs)
recs = recs[:900]
X = [[], []]
Y = []
for i in recs:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    T = len(s)
    depth = 0
    stack = []
    dep = []
    for t in range(T):
        dep.append(depth)
        if t >= npr[i] and s[t] == "(":
            b = t + 2 < T and s[t + 1] == "fun" and s[t + 2] == "("
            stack.append(b)
            depth += b
        elif t >= npr[i] and s[t] == ")" and stack and stack.pop():
            depth -= 1
    pos = []
    t = npr[i]
    while t < T - 2:
        if s[t] == "exact" and s[t + 2] == ")" and dep[t] >= 1:
            D0, r, j = dep[t], 0, t
            while j + 2 < T and s[j] == "exact" and s[j + 2] == ")" and dep[j] >= 1:
                r += 1
                pos.append((j + 2, D0, r, D0 - r))  # the closing `)`
                if j + 4 < T and s[j + 3] == ";" and s[j + 4] == "exact":
                    j += 4
                else:
                    break
            t = j + 1
        else:
            t += 1
    if not pos:
        continue
    with torch.no_grad():
        hs = [m.resid(x[None])[L][0] for m in M]
    for p, D0, r, k in pos:
        for a in range(2):
            X[a].append(hs[a][p])
        Y.append((D0, r, k))
X = [torch.stack(v) for v in X]
Y = torch.tensor(Y).float()
print(
    f"{len(Y)} closing `)` tokens; D_start dist",
    {int(d): int((Y[:, 0] == d).sum()) for d in Y[:, 0].unique()},
)
dd = W_dec[:, 1] / scale[1] - W_dec[:, 0] / scale[0]  # raw-space decoder delta (m, D)
top = [9, 2922, 5099, 4176, 3799, 2115, 6125, 570, 7057, 2294]
alive = torch.nonzero(cc["freq"] > 0).flatten()
rnd = alive[torch.randperm(len(alive), generator=torch.Generator().manual_seed(0))[:500]]
for a, nm in enumerate(cc["names"]):
    Xc = X[a] - X[a].mean(0)
    for col, lab in enumerate(("D_start", "r", "k_after")):
        y = Y[:, col] - Y[:, col].mean()
        w = torch.linalg.solve(Xc.T @ Xc + 10.0 * torch.eye(Xc.shape[1]), Xc.T @ y)
        r2 = 1 - ((Xc @ w - y) ** 2).sum() / (y**2).sum()
        cs = torch.nn.functional.cosine_similarity(dd, w[None], dim=-1)
        null = cs[rnd].abs()
        print(
            f"{nm} probe {lab:>8} (train R2 {float(r2):.2f}): |cos(Delta d_j, w)| top latents "
            + " ".join(f"{j}:{float(cs[j]):+.2f}" for j in top)
            + f" | null |cos| median {float(null.median()):.2f}, q95 {float(null.quantile(0.95)):.2f}"
        )
