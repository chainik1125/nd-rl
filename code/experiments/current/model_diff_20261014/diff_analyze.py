# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Diff-SAE (SAE on h_fin - h_pre) analysis: attribution of the hard-pop metric onto diff latents (first order, as
attrib.py), then causal steering g0_pre + sum_{j in top-N} f_j d_j at close-run positions (or all proof positions),
pop accuracy by (k, r). Also prints each top latent's firing profile over close-run (role, D_start, r).
python diff_analyze.py runs/diff_g0_L3"""

import collections
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(8)
run = sys.argv[1]  # add --mag to rank by change magnitude
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
pk = C["pop_k"].long()
W_enc, W_dec, b_enc, b_dec, scale, thr = (
    cc["W_enc"],
    cc["W_dec"],
    cc["b_enc"],
    cc["b_dec"],
    cc["scale"],
    cc["thr"],
)
dn = W_dec.norm(dim=-1).sum(-1)
D0 = W_dec[:, 0] / scale[0]  # raw-space decoder (m, D)


def code(hp, hf):
    z = ((hf - hp) * scale[0])[:, None]
    pa = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
    return pa * (pa * dn > thr)


def close_mask(s, n_p):
    m = torch.zeros(len(s), dtype=torch.bool)
    for t in range(n_p, len(s) - 2):
        if s[t] == ")" and t >= 2 and s[t - 2] == "exact":
            m[t - 2 : t + 1] = True
            if t + 2 < len(s) and s[t + 1] == ";" and s[t + 2] == "exact":
                m[t + 1 : t + 3] = True
    return m


lad = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("lad3")]
random.Random(1).shuffle(lad)
rank_set, eval_pool = lad[:250], lad[250:]
# 1. attribution on the rank set
A = torch.zeros(W_dec.shape[0])
tot = 0.0
for i in rank_set:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    hv, opener, stack, pops = {}, [], [], []
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
            if (
                s[t + 1] == ";" and s[t + 2] == "exact" and len(opener) >= 1
            ):  # alternative = enclosing box's opener
                pops.append((t + 2, int(x[t + 3]), int(x[opener[-1]])))
    if not pops:
        continue
    with torch.no_grad():
        hp = M[0].resid(x[None])[L][0]
        hf = M[1].resid(x[None])[L][0]
    h = hp.clone().requires_grad_(True)
    lg = Tiny.logits_from.__wrapped__(M[0], h[None], L)[0]
    (g,) = torch.autograd.grad(sum(lg[j, a] - lg[j, b] for j, a, b in pops), h)
    g = g * close_mask(s, npr[i])[:, None]
    with torch.no_grad():
        f = code(hp, hf)
        tot += float(((hf - hp) * g).sum())
        A += (f * (g @ D0.T)).sum(0)
o = A.argsort(descending=True)
cum = A[o].cumsum(0)
if (
    "--mag" in sys.argv
):  # rank by magnitude of the reconstructed change at close tokens instead of first-order attribution
    Mg = torch.zeros(W_dec.shape[0])
    for i in rank_set:
        x = ids[starts[i] : starts[i + 1]]
        s = [itos[t] for t in x.tolist()]
        with torch.no_grad():
            f = code(M[0].resid(x[None])[L][0], M[1].resid(x[None])[L][0])
        Mg += (f[close_mask(s, npr[i])] * D0.norm(dim=-1)).sum(0)
    o = Mg.argsort(descending=True)
print(
    f"{run}: first-order total {tot:.1f}, diff latents explain {float(A.sum()):.1f};",
    " ".join(f"top{n}: {float(cum[n - 1]) / tot:.0%}" for n in (1, 2, 5, 10, 20, 50)),
)
print("  top 10:", " ".join(f"{int(j)}:{float(A[j]):.1f}" for j in o[:10]))
# 2. causal steering on held-out records (other corpus records with k>=1 pops, excluding the rank set)
others = [
    i
    for i in range(len(starts) - 1)
    if starts[i + 1] - starts[i] <= 700
    and (pk[starts[i] : starts[i + 1]] >= 1).any()
    and i not in set(rank_set)
]
random.Random(0).shuffle(others)
others = others[:1000]


def evaluate(lat, where):
    hit = collections.Counter()
    totc = collections.Counter()
    for i in others:
        x = ids[starts[i] : starts[i + 1]]
        s = [itos[t] for t in x.tolist()]
        with torch.no_grad():
            hp = M[0].resid(x[None])[L][0]
            if lat is not None:
                hf = M[1].resid(x[None])[L][0]
                f = code(hp, hf)
                fl = torch.zeros_like(f)
                fl[:, lat] = f[:, lat]
                pos = close_mask(s, npr[i]) if where == "close" else torch.arange(len(s)) >= npr[i]
                hp = hp + (fl @ D0) * pos[:, None]
            am = M[0].logits_from(hp[None], L)[0].argmax(-1)
        p = pk[starts[i] : starts[i + 1]]
        for j in (p >= 1).nonzero().flatten().tolist():
            r, q = 1, j - 2
            while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
                r += 1
                q -= 4
            key = "easy" if int(p[j]) == max(0, 2 - r) else "hard"
            totc[key] += 1
            hit[key] += int(am[j] == x[j + 1])
            totc[(int(p[j]), min(r, 3))] += 1
            hit[(int(p[j]), min(r, 3))] += int(am[j] == x[j + 1])
    cells = [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2)]
    return (
        f"easy {hit['easy'] / totc['easy']:4.0%} hard {hit['hard'] / totc['hard']:4.0%} | "
        + " ".join(f"k{k}r{r} {hit[(k, r)] / totc[(k, r)]:4.0%}" for k, r in cells)
    )


print("  held-out pops, g0_pre clean:", evaluate(None, None))
for n in (5, 20, 100, 400) if "--mag" in sys.argv else (1, 2, 5, 10, 20, 50, 4096):
    lat = o[:n]
    print(f"  + top {n:>4} diff latents at close tokens: {evaluate(lat, 'close')}", flush=True)
if "--mag" not in sys.argv:
    print(f"  + all diff latents at all proof tokens: {evaluate(o, 'all')}")
torch.save({"A": A, "order": o}, f"{run}/attrib.pt")
