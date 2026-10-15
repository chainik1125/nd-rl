# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Is the pointer head the copy mechanism? At every pop (k >= 1) in 1,000 corpus records, take the box header whose
`fun` token gets the most attention from head (layer, head) at the pop `exact`; call its opener name the head's 'pointed
name'. Report how often model argmax == pointed name, == truth, and pointed == truth, by (k, r).
python pointer.py g0_pre 3 2"""

import collections
import random
import sys

import torch
from tiny import N_HEAD, D, Tiny

torch.set_num_threads(6)
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
mname, LH = sys.argv[1], int(sys.argv[2])
HH = int(sys.argv[3])
r_, st = mname.split("_")
m = Tiny(
    torch.load(
        f"ckpt/runs/long-rl-20261014-{CK[r_]}/{'pretrain' if st == 'pre' else 'final'}.pt",
        map_location="cpu",
    )["state"]
)
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
recs = recs[:1000]


@torch.no_grad()
def attn_row(x, layer, head):
    T = x.shape[1]
    bias = m.bias(T, x.device)
    h = m.resid(x)[layer]
    a = m._ln(h, f"blocks.{layer}.ln1")
    q, k, _ = (
        m._lin(a, f"blocks.{layer}.qkv").view(1, T, 3, N_HEAD, D // N_HEAD).permute(2, 0, 3, 1, 4)
    )
    s = (q @ k.transpose(-1, -2)) / (D // N_HEAD) ** 0.5 + bias
    return torch.softmax(s, -1)[0, head]


agg = collections.defaultdict(collections.Counter)
for i in recs:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    A = attn_row(x[None], LH, HH)
    lg = m.logits_from(m.resid(x[None])[-1], 6)[0]
    hv, opener, stack, funs = {}, [], [], {}  # funs: fun position -> opener name id
    for t in range(npr[i], len(s) - 3):
        if s[t] == "have":
            hv[len(opener)] = t + 1
        if s[t] == "(":
            box = s[t + 1] == "fun" and s[t + 2] == "("
            stack.append(box)
            if box:
                opener.append(hv[len(opener)])
                funs[t + 1] = int(x[hv[len(opener) - 1]])
        elif s[t] == ")" and stack and stack.pop():
            opener.pop()
            if s[t + 1] == ";" and s[t + 2] == "exact" and len(opener) >= 1:
                j = t + 2
                k = len(opener)
                r, q = 1, j - 2
                while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
                    r += 1
                    q -= 4
                fp = [p for p in funs if p < j]
                best = max(fp, key=lambda p: float(A[j, p]))
                pointed, truth, pred = funs[best], int(x[j + 1]), int(lg[j].argmax())
                c = agg[(k, min(r, 3))]
                c["n"] += 1
                c["pred=pointed"] += pred == pointed
                c["pred=truth"] += pred == truth
                c["pointed=truth"] += pointed == truth
print(
    f"{mname} head {LH}.{HH}: by (k, r): n, P(argmax = pointed opener), P(argmax = truth), P(pointed = truth)"
)
for key in sorted(agg):
    c = agg[key]
    if c["n"] >= 15:
        print(
            f"  k{key[0]} r{key[1]}: n {c['n']:4d}  pred=pointed {c['pred=pointed'] / c['n']:4.0%}  pred=truth {c['pred=truth'] / c['n']:4.0%}  pointed=truth {c['pointed=truth'] / c['n']:4.0%}"
        )
