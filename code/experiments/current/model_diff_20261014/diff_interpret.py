# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Firing profile of the top diff-SAE latents (ranked by change magnitude at close tokens on the rank set, as
diff_analyze.py --mag) over close-run (role, D_start, r), on held-out spine-3 + EI records, with off-run firing rate.
python diff_interpret.py runs/diff_g0_L3 20"""

import collections
import random
import sys

import torch
from roles_lib import roles
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
D0n = (W_dec[:, 0] / scale[0]).norm(dim=-1)
lad = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("lad3")]
random.Random(1).shuffle(lad)
other = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("ei1")]
random.Random(0).shuffle(other)


def code(i):
    x = ids[starts[i] : starts[i + 1]]
    with torch.no_grad():
        z = ((M[1].resid(x[None])[L][0] - M[0].resid(x[None])[L][0]) * scale[0])[:, None]
        pa = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
    return x, pa * (pa * dn > thr)


mag = torch.zeros(len(dn))
for i in lad[:250]:
    x, f = code(i)
    s = [itos[t] for t in x.tolist()]
    R = roles(s, npr[i])
    pos = torch.zeros(len(s), dtype=torch.bool)
    for p in R:
        pos[p] = True
    mag += (f[pos] * D0n).sum(0)
top = mag.argsort(descending=True)[:N]
fire = collections.defaultdict(lambda: torch.zeros(N))
cnt = collections.Counter()
off = torch.zeros(N)
n_off = 0
for i in lad[250:650] + other[:300]:
    x, f = code(i)
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
            off += on[p]
            n_off += 1
print(
    f"{run}: top {N} diff latents by change magnitude at close tokens; rows (role, D_start, r): firing rate"
)
print(" " * 18 + " ".join(f"{int(j):>5}" for j in top))
print(f"{'off-run rate':>18}" + " ".join(f"{float(v / n_off):5.2f}" for v in off))
for key in sorted(
    cnt, key=lambda k: (["exact", "name", "close", "semi", "pop"].index(k[0]), k[1], k[2])
):
    if cnt[key] < 30:
        continue
    print(
        f"{str(key):>18}"
        + " ".join(f"{float(a):5.2f}" for a in fire[key] / cnt[key])
        + f"   n={cnt[key]}"
    )
