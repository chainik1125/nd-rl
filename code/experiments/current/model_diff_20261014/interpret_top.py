# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Interpret the latents that carry the pre->fin change at closing tokens (contribution-ranked, as ablate_cc.py contrib:N).
For each: decoder cos and |d_fin - d_pre| / |d_pre|; at close-run tokens, firing rate by token role (exact, name, `)`,
`;`, pop exact) and by (D_start, r); and whether it fires off close runs. Also: does the fin-minus-pre decoder delta of
the top latents point along a 'D_start' direction (ridge probe on fin's close tokens)?
python interpret_top.py runs/g0_L3 20"""

import collections
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(8)
run, N = sys.argv[1], int(sys.argv[2])
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
W_enc, W_dec, b_enc, scale, thr = cc["W_enc"], cc["W_dec"], cc["b_enc"], cc["scale"], cc["thr"]
dn = W_dec.norm(dim=-1).sum(-1)
dd = (W_dec[:, 1] - W_dec[:, 0]).norm(dim=-1)
lad = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("lad3")]
other = [
    i
    for i in range(len(starts) - 1)
    if C["src"][i] in (C["SRCS"].index("ei1"), C["SRCS"].index("pre"))
]
random.Random(0).shuffle(other)


def roles(s, n_p):
    """per position: (role, D_start, r) inside close runs; role in exact / name / close / semi / pop"""
    out = {}
    T = len(s)
    depth = 0
    stack = []
    dep = [0] * T
    for i in range(T):
        dep[i] = depth
        if i >= n_p and s[i] == "(":
            box = i + 2 < T and s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            depth += box
        elif i >= n_p and s[i] == ")" and stack and stack.pop():
            depth -= 1
    i = n_p
    while i < T - 2:
        if s[i] == "exact" and s[i + 2] == ")" and dep[i] >= 1:
            # start of a close run: walk forward over `exact n ) ;` groups that close boxes
            D0 = dep[i]
            r = 0
            j = i
            while j + 2 < T and s[j] == "exact" and s[j + 2] == ")" and dep[j] >= 1:
                r += 1
                out[j] = ("exact", D0, r)
                out[j + 1] = ("name", D0, r)
                out[j + 2] = ("close", D0, r)
                if j + 4 < T and s[j + 3] == ";" and s[j + 4] == "exact":
                    out[j + 3] = ("semi", D0, r)
                    out[j + 4] = ("pop", D0, r)
                    j += 4
                    if not (j + 2 < T and s[j + 2] == ")"):
                        break
                else:
                    break
            i = j + 1
        else:
            i += 1
    return out


def acts(i):
    x = ids[starts[i] : starts[i + 1]]
    with torch.no_grad():
        z = torch.stack([m.resid(x[None])[L][0] for m in M], 1) * scale.view(1, 2, 1)
        pa = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
        return x, pa * (pa * dn > thr), z


# rank as in ablate_cc contrib: on the first 150 lad3 records (rank set); describe on the rest
score = torch.zeros(len(dd))
for i in lad[:150]:
    x, f, _ = acts(i)
    s = [itos[t] for t in x.tolist()]
    R = roles(s, npr[i])
    pos = torch.zeros(len(s), dtype=torch.bool)
    for p, (ro, _, _) in R.items():
        if ro in ("exact", "name", "close"):
            pos[p] = True
        if ro in ("semi", "pop"):
            pos[p] = True
    score += (f[pos] * dd).sum(0)
top = score.argsort(descending=True)[:N]
fire = collections.defaultdict(lambda: torch.zeros(N))
cnt = collections.Counter()
off_fire = torch.zeros(N)
off_n = 0
for i in lad[150:600] + other[:400]:
    x, f, _ = acts(i)
    s = [itos[t] for t in x.tolist()]
    R = roles(s, npr[i])
    on = (f[:, top] > 0).float()
    for p in range(npr[i], len(s)):
        if p in R:
            ro, D0, r = R[p]
            key = (ro, min(D0, 4), min(r, 4))
            fire[key] += on[p]
            cnt[key] += 1
        else:
            off_fire += on[p]
            off_n += 1
print(
    f"{run}: top {N} latents by pre->fin change at closing tokens. cols = latents; rows = (role, D_start, r) firing rate"
)
print(" " * 18 + " ".join(f"{int(j):>5}" for j in top))
print(f"{'cos(d_pre,d_fin)':>18}" + " ".join(f"{float(cc['cos'][j]):5.2f}" for j in top))
print(f"{'|dd|/|d_pre|':>18}" + " ".join(f"{float(dd[j] / W_dec[j, 0].norm()):5.2f}" for j in top))
print(f"{'off-run rate':>18}" + " ".join(f"{float(v / off_n):5.2f}" for v in off_fire))
for key in sorted(
    cnt, key=lambda k: (["exact", "name", "close", "semi", "pop"].index(k[0]), k[1], k[2])
):
    if cnt[key] < 30:
        continue
    v = fire[key] / cnt[key]
    print(f"{str(key):>18}" + " ".join(f"{float(a):5.2f}" for a in v) + f"   n={cnt[key]}")
