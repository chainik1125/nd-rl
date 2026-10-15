# ruff: noqa: E501,E741  (research scripts: long result lines, bare open(), short loop names)
"""For every head: at pops (k >= 1), the box header whose `fun` (or `(`/binder) the head attends to most -> pointed opener.
Rank heads by P(model argmax == pointed) (is it the copy head?) and report P(pointed == truth) (does it point right?).
Pops split into 'easy' (k = max(0, 2 - r), heuristic right) and 'hard' (heuristic wrong). python headscan.py <models>"""

import collections
import random
import sys

import torch
from tiny import N_HEAD, N_LAYER, D, Tiny

torch.set_num_threads(6)
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
C = torch.load("out/corpus.pt")
starts = C["starts"].tolist() + [len(C["ids"])]
itos = C["itos"]
ids = C["ids"].long()
pk = C["pop_k"].long()
npr = C["n_prompt"].tolist()
recs = [
    i
    for i in range(len(starts) - 1)
    if starts[i + 1] - starts[i] <= 700 and (pk[starts[i] : starts[i + 1]] >= 1).any()
]
random.Random(0).shuffle(recs)
recs = recs[:700]


@torch.no_grad()
def all_attn(m, x):
    T = x.shape[1]
    bias = m.bias(T, x.device)
    hs = m.resid(x)
    out = []
    for l in range(N_LAYER):
        a = m._ln(hs[l], f"blocks.{l}.ln1")
        q, k, _ = (
            m._lin(a, f"blocks.{l}.qkv").view(1, T, 3, N_HEAD, D // N_HEAD).permute(2, 0, 3, 1, 4)
        )
        out.append(torch.softmax((q @ k.transpose(-1, -2)) / (D // N_HEAD) ** 0.5 + bias, -1)[0])
    return torch.stack(out), m.logits_from(hs[-1], 6)[0]


for mname in sys.argv[1:]:
    r_, st = mname.split("_")
    m = Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[r_]}/{'pretrain' if st == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    agg = collections.defaultdict(
        lambda: torch.zeros(3, N_LAYER, N_HEAD)
    )  # [n, pred=pointed, pointed=truth]
    for i in recs:
        x = ids[starts[i] : starts[i + 1]]
        s = [itos[t] for t in x.tolist()]
        A, lg = all_attn(m, x[None])
        hv, opener, stack, hdr = (
            {},
            [],
            [],
            [],
        )  # hdr: (positions of `(`, `fun`, `(`, binder) , opener name id
        for t in range(npr[i], len(s) - 3):
            if s[t] == "have":
                hv[len(opener)] = t + 1
            if s[t] == "(":
                box = s[t + 1] == "fun" and s[t + 2] == "("
                stack.append(box)
                if box:
                    opener.append(hv[len(opener)])
                    hdr.append(([t, t + 1, t + 2, t + 3], int(x[hv[len(opener) - 1]])))
            elif s[t] == ")" and stack and stack.pop():
                opener.pop()
                if s[t + 1] == ";" and s[t + 2] == "exact" and len(opener) >= 1:
                    j = t + 2
                    k = len(opener)
                    r, q = 1, j - 2
                    while (
                        q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")"
                    ):
                        r += 1
                        q -= 4
                    hs_ = [h for h in hdr if h[0][0] < j]
                    w = torch.stack([A[:, :, j, h[0]].sum(-1) for h in hs_], -1)  # L,H,nh
                    pointed = torch.tensor([h[1] for h in hs_])[w.argmax(-1)]  # L,H
                    truth, pred = int(x[j + 1]), int(lg[j].argmax())
                    g = agg["easy" if k == max(0, 2 - r) else "hard"]
                    g[0] += 1
                    g[1] += (pointed == pred).float()
                    g[2] += (pointed == truth).float()
    print(
        f"\n== {mname}: top heads by P(argmax = pointed) on hard pops; cells: hard pred=pointed / pointed=truth | easy same"
    )
    H, E = agg["hard"], agg["easy"]
    ph = H[1] / H[0]
    tr_h = H[2] / H[0]
    pe = E[1] / E[0]
    tr_e = E[2] / E[0]
    order = (ph + pe).flatten().argsort(descending=True)[:6]
    print(f"   n hard {int(H[0, 0, 0])}, n easy {int(E[0, 0, 0])}")
    for o in order.tolist():
        l, h = divmod(o, N_HEAD)
        print(
            f"   {l}.{h}: hard {ph[l, h]:4.0%} / {tr_h[l, h]:4.0%} | easy {pe[l, h]:4.0%} / {tr_e[l, h]:4.0%}"
        )
