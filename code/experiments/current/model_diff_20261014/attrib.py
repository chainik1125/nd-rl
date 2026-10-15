# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Attribution patching (first-order) of the pre->fin change at closing tokens onto crosscoder latents.
Metric per hard pop: logit(correct opener) - logit(off-by-one opener) in the PRE model. g = d metric / d h_L at every
close-run position (pre model, layer L residual). Total first-order effect of the change: sum (h_fin - h_pre) . g;
latent j's share: sum f_j (d_j^fin/s_fin - d_j^pre/s_pre) . g; plus the b_dec delta and the reconstruction-error delta.
python attrib.py runs/g0_L3 [n_records]"""

import collections
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(8)
run = sys.argv[1]
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
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
W_enc, W_dec, b_enc, b_dec, scale, thr = (
    cc["W_enc"],
    cc["W_dec"],
    cc["b_enc"],
    cc["b_dec"],
    cc["scale"],
    cc["thr"],
)
dn = W_dec.norm(dim=-1).sum(-1)
Dd = W_dec[:, 1] / scale[1] - W_dec[:, 0] / scale[0]
lad = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("lad3")]
random.Random(1).shuffle(lad)
lad = lad[:NR]
A = torch.zeros(len(Dd))
tot = 0.0
bdec_t = 0.0
err_t = 0.0
n_pop = 0
per_cell = collections.Counter()
for i in lad:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    # pops with their correct and 'up' names; close-run positions
    hv, opener, stack = {}, [], []
    pops = []
    close = torch.zeros(len(s), dtype=torch.bool)
    for t in range(npr[i], len(s) - 3):
        if s[t] == "have":
            hv[len(opener)] = t + 1
        if s[t] == "(":
            b = s[t + 1] == "fun" and s[t + 2] == "("
            stack.append(b)
            if b:
                opener.append(hv[len(opener)])
        elif s[t] == ")" and stack and stack.pop():
            opener.pop()
            close[t - 2 : t + 1] = True
            if s[t + 1] == ";" and s[t + 2] == "exact":
                close[t + 1 : t + 3] = True
                if len(opener) >= 1:  # off-by-one alternative = the enclosing box's opener
                    pops.append((t + 2, int(x[t + 3]), int(x[opener[-1]])))
    if not pops:
        continue
    with torch.no_grad():
        hp = M[0].resid(x[None])[L][0]
        hf = M[1].resid(x[None])[L][0]
    h = hp.clone().requires_grad_(True)
    lg = Tiny.logits_from.__wrapped__(M[0], h[None], L)[0]  # grad-enabled
    metric = sum(lg[j, good] - lg[j, up] for j, good, up in pops)
    (g,) = torch.autograd.grad(metric, h)
    g = g * close[:, None]
    with torch.no_grad():
        z = torch.stack([hp, hf], 1) * scale.view(1, 2, 1)
        pa = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
        f = pa * (pa * dn > thr)
        xh = torch.einsum("bm,mnd->bnd", f, W_dec) + b_dec
        delta = hf - hp
        tot += float((delta * g).sum())
        A += (f * (g @ Dd.T)).sum(0)
        bdec_t += float((g.sum(0) * (b_dec[1] / scale[1] - b_dec[0] / scale[0])).sum())
        e_f = hf - xh[:, 1] / scale[1]
        e_p = hp - xh[:, 0] / scale[0]
        err_t += float(((e_f - e_p) * g).sum())
    n_pop += len(pops)
print(f"{run}: {n_pop} pops with an off-by-one alternative, {len(lad)} spine-3 records")
print(
    f"first-order total (h_fin - h_pre).g at close tokens: {tot:.1f}  = latents {float(A.sum()):.1f} + b_dec {bdec_t:.1f} + recon-error {err_t:.1f}"
)
o = A.argsort(descending=True)
cum = A[o].cumsum(0)
for n in (1, 2, 5, 10, 20, 50, 100, 200, 500):
    print(
        f"  top {n:>3} latents: {float(cum[n - 1]):7.1f} ({float(cum[n - 1]) / tot:5.0%} of total)"
    )
print(
    "  top 15:",
    " ".join(f"{int(j)}:{float(A[j]):.1f}(cos {float(cc['cos'][j]):.2f})" for j in o[:15]),
)
print("  most negative 5:", " ".join(f"{int(j)}:{float(A[j]):.1f}" for j in A.argsort()[:5]))
torch.save({"A": A, "tot": tot}, f"{run}/attrib.pt")
