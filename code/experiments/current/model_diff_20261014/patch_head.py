# ruff: noqa: E501,E741  (research scripts: long result lines, bare open(), short loop names)
"""Query vs key: for one head (layer, head) of the BASE model, replace its query and/or keys with what the DONOR model
computes there (donor residual -> donor ln1/qkv), keep base values and everything else. Pop accuracy by (k, r).
modes: q (queries at all positions), k (keys at all positions), qk, out (the head's whole output from the donor).
python patch_head.py g0 3 2      (base g0_pre <- donor g0_fin, and reverse)"""

import collections
import random
import sys

import torch
import torch.nn.functional as F
from tiny import N_HEAD, D, Tiny

torch.set_num_threads(6)
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
pair, LH, HH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
M = {
    s: Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[pair]}/{'pretrain' if s == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    for s in ("pre", "fin")
}
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
random.Random(0).shuffle(recs)
recs = recs[:800]
dh = D // N_HEAD


def qkv(m, h, l):
    T = h.shape[1]
    a = m._ln(h, f"blocks.{l}.ln1")
    return (
        m._lin(a, f"blocks.{l}.qkv").view(1, T, 3, N_HEAD, dh).permute(2, 0, 3, 1, 4)
    )  # 3,1,H,T,dh


@torch.no_grad()
def run(base, donor, x, mode):
    mb, md = M[base], M[donor]
    T = x.shape[1]
    bias = mb.bias(T, x.device)
    hb = mb.resid(x)
    hd = md.resid(x)
    h = hb[LH]
    q, k, v = qkv(mb, h, LH)
    qd, kd, vd = qkv(md, hd[LH], LH)
    if "q" in mode:
        q = q.clone()
        q[:, HH] = qd[:, HH]
    if "k" in mode:
        k = k.clone()
        k[:, HH] = kd[:, HH]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    if mode == "out":
        yd = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=bias)
        y = y.clone()
        y[:, HH] = yd[:, HH]
    b = f"blocks.{LH}."
    y = mb._lin(y.transpose(1, 2).reshape(1, T, D), b + "proj")
    h = h + mb._ln(y, b + "ln_attn_out")
    h = h + mb._ln(
        mb._lin(F.gelu(mb._lin(mb._ln(h, b + "ln2"), b + "fc1")), b + "fc2"), b + "ln_mlp_out"
    )
    return mb.logits_from(h, LH + 1)[0]


for base, donor in (("pre", "fin"), ("fin", "pre")):
    for mode in ("none", "q", "k", "qk", "out"):
        hit = collections.Counter()
        tot = collections.Counter()
        for i in recs:
            x = ids[starts[i] : starts[i + 1]]
            s = [itos[t] for t in x.tolist()]
            lg = (
                M[base].logits_from(M[base].resid(x[None])[-1], 6)[0]
                if mode == "none"
                else run(base, donor, x[None], mode)
            )
            p = pk[starts[i] : starts[i + 1]]
            for j in (p >= 1).nonzero().flatten().tolist():
                r, q_ = 1, j - 2
                while (
                    q_ - 4 >= 0 and s[q_ - 2] == "exact" and s[q_ - 3] == ";" and s[q_ - 4] == ")"
                ):
                    r += 1
                    q_ -= 4
                key = "easy" if int(p[j]) == max(0, 2 - r) else "hard"
                tot[key] += 1
                hit[key] += int(lg[j].argmax() == x[j + 1])
        print(
            f"{pair}_{base} with head {LH}.{HH} {mode:>4} from {pair}_{donor}: hard pops {hit['hard'] / tot['hard']:4.0%} (n {tot['hard']}), easy {hit['easy'] / tot['easy']:4.0%}",
            flush=True,
        )
