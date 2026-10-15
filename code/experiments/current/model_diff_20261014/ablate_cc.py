# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Causal test of crosscoder latents on the pop behaviour.
  ablate: in the FINAL model's forward, at the chosen positions of layer L, subtract sum_j f_j d_j^fin / scale_fin
  steer:  in the PRETRAINED model's forward, add sum_j f_j (d_j^fin - d_j^pre) / scale_pre  (f from both models' acts)
  random: same as ablate with as many random alive latents of matched frequency (control)
Latent sets: 'fin_only' (r > 0.9), 'top:<label>:<n>' (the n latents most enriched on a label among fin-leaning r > 0.7),
or a comma list of ids. Positions: 'all' proof tokens or 'close' (the `exact nE ) ;` run and pop `exact`).
Metric: pop accuracy by (k, r) on 1,500 corpus records with pops, and proof CE on 300 pretraining records.
python ablate_cc.py runs/g0_L4 fin_only close
"""

import collections
import os
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(6)
run, which, where = sys.argv[1], sys.argv[2], sys.argv[3]
cc = torch.load(f"{run}/cc.pt", map_location="cpu")
L = cc["args"]["layer"]
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
pre_n, fin_n = cc["names"]
M = {}
for nm in cc["names"]:
    r_, st = nm.split("_")
    M[nm] = Tiny(
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
W_enc, W_dec, b_enc, b_dec, scale, thr = (
    cc["W_enc"],
    cc["W_dec"],
    cc["b_enc"],
    cc["b_dec"],
    cc["scale"],
    cc["thr"],
)
dn = W_dec.norm(dim=-1).sum(-1)
rel, freq = cc["rel"], cc["freq"]
alive = freq > 0


def pick(spec):
    if spec == "fin_only":
        return torch.nonzero(alive & (rel > 0.9)).flatten()
    if spec == "pre_only":
        return torch.nonzero(alive & (rel < 0.1)).flatten()
    if spec.startswith("top:"):
        _, lab, n = spec.split(":")
        key, vals = {
            "pop2": ("pop_k", [2]),
            "pop1": ("pop_k", [1]),
            "deep": ("depth", [3, 4, 5]),
            "close1": ("close_k", [1]),
        }[lab]
        p = cc["cnt"][key][vals].sum(0) / cc["tot"][key][vals].sum()
        base = cc["cnt"]["src"].sum(0) / cc["tot"]["src"].sum()
        e = torch.where(
            alive & (rel > 0.7) & (p > 0.1), p / base.clamp_min(1e-9), torch.zeros_like(p)
        )
        return e.argsort(descending=True)[: int(n)]
    if spec.startswith("chg:"):
        # latents that fire on the label set, ranked by P(fire | label) * |d_fin - d_pre| (how much their write changed)
        _, lab, n = spec.split(":")
        key, vals = {
            "pop2": ("pop_k", [2]),
            "pop1": ("pop_k", [1]),
            "pop12": ("pop_k", [1, 2]),
            "deep": ("depth", [3, 4, 5]),
            "close12": ("close_k", [1, 2]),
        }[lab]
        p = cc["cnt"][key][vals].sum(0) / cc["tot"][key][vals].sum()
        dd = (W_dec[:, 1] - W_dec[:, 0]).norm(dim=-1)
        return (torch.where(alive & (p > 0.05), p * dd, torch.zeros_like(p))).argsort(
            descending=True
        )[: int(n)]
    if spec == "all":
        return torch.nonzero(alive).flatten()
    if spec.startswith("contrib:"):
        # rank by summed |f_j| * |d_j^fin - d_j^pre| at close positions of 150 spine-3 ladder records
        n_ = int(spec.split(":")[1])
        dd = (W_dec[:, 1] - W_dec[:, 0]).norm(dim=-1)
        score = torch.zeros(len(dd))
        lad = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("lad3")][:150]
        with torch.no_grad():
            for i in lad:
                x = ids[starts[i] : starts[i + 1]]
                s_ = [itos[t] for t in x.tolist()]
                z = torch.stack(
                    [M[pre_n].resid(x[None])[L][0], M[fin_n].resid(x[None])[L][0]], 1
                ) * scale.view(1, 2, 1)
                pa = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
                f = pa * (pa * dn > thr)
                score += (f[closepos(s_, npr[i])] * dd).sum(0)
        return score.argsort(descending=True)[:n_]
    if spec.startswith("lowcos:"):
        return torch.nonzero(alive & (cc["cos"] < float(spec.split(":")[1]))).flatten()
    return torch.tensor([int(x) for x in spec.split(",")])


def ctrl(sel):
    """random alive latents matched on log-frequency (greedy nearest, without replacement, excluding sel)"""
    pool = torch.nonzero(alive).flatten()
    pool = pool[~torch.isin(pool, sel)][
        torch.randperm(
            int((~torch.isin(torch.nonzero(alive).flatten(), sel)).sum()),
            generator=torch.Generator().manual_seed(0),
        )
    ]
    lf = freq[pool].log()
    used = torch.zeros(len(pool), dtype=torch.bool)
    out = []
    for j in sel.tolist():
        d = (lf - torch.log(freq[j])).abs().masked_fill(used, float("inf"))
        b = int(d.argmin())
        used[b] = True
        out.append(int(pool[b]))
    return torch.tensor(out)


recs = [
    i
    for i in range(len(starts) - 1)
    if starts[i + 1] - starts[i] <= 700 and (pk[starts[i] : starts[i + 1]] >= 1).any()
]
random.Random(0).shuffle(recs)
recs = recs[:1500]
pre_recs = [i for i in range(len(starts) - 1) if C["src"][i] == C["SRCS"].index("pre")][:300]


def closepos(s, n_p):
    T = len(s)
    m = torch.zeros(T, dtype=torch.bool)
    for i in range(n_p, T - 1):
        if s[i] == ")" and i >= 2 and s[i - 2] == "exact":
            m[i - 2 : i + 1] = True
            if i + 2 < T and s[i + 1] == ";" and s[i + 2] == "exact":
                m[i + 1 : i + 3] = True
    return m


sel = pick(which)
print(f"{run}: {len(sel)} latents ({which}), positions {where}, layer {L}")


@torch.no_grad()
def forward(i, mode, lat):
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    hp, hf = M[pre_n].resid(x[None])[L][0], M[fin_n].resid(x[None])[L][0]
    z = torch.stack([hp, hf], 1) * scale.view(1, 2, 1)
    pre_act = torch.relu(torch.einsum("bnd,ndm->bm", z, W_enc) + b_enc)
    f = pre_act * (pre_act * dn > thr)
    pos = torch.zeros(len(s), dtype=torch.bool)
    pos[npr[i] :] = True
    if where == "close":
        pos &= closepos(s, npr[i])
    fl = torch.zeros_like(f)
    fl[:, lat] = f[:, lat]
    fl[~pos] = 0
    if mode == "clean_fin":
        return x, M[fin_n].logits_from(hf[None], L)[0]
    if mode == "clean_pre":
        return x, M[pre_n].logits_from(hp[None], L)[0]
    if mode == "ablate":
        h = hf - (fl @ W_dec[:, 1]) / scale[1]
        return x, M[fin_n].logits_from(h[None], L)[0]
    if mode == "steer":
        h = hp + (fl @ (W_dec[:, 1] - W_dec[:, 0])) / scale[0]
        if os.environ.get("STEER_BDEC"):
            h = h + pos[:, None] * (b_dec[1] - b_dec[0]) / scale[0]
        return x, M[pre_n].logits_from(h[None], L)[0]


def evaluate(mode, lat):
    hit = collections.Counter()
    tot = collections.Counter()
    for i in recs:
        x, lg = forward(i, mode, lat)
        s = [itos[t] for t in x.tolist()]
        p = pk[starts[i] : starts[i + 1]]
        for j in (p >= 0).nonzero().flatten().tolist():
            r, q = 1, j - 2
            while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
                r += 1
                q -= 4
            key = (int(p[j]), min(r, 3))
            tot[key] += 1
            hit[key] += int(lg[j].argmax() == x[j + 1])
    ce = nt = 0
    for i in pre_recs:
        x, lg = forward(i, mode, lat)
        a = npr[i]
        ce += float(torch.nn.functional.cross_entropy(lg[a - 1 : -1], x[a:], reduction="sum"))
        nt += len(x) - a
    cells = "  ".join(
        f"k{k}r{r} {hit[(k, r)] / tot[(k, r)]:4.0%}" for k, r in sorted(tot) if tot[(k, r)] >= 20
    )
    print(f"  {mode:>10} ({len(lat)} lat): {cells} | CE pretrain {ce / nt:.4f}", flush=True)


MODES = (
    sys.argv[4].split(",") if len(sys.argv) > 4 else ["clean_pre", "clean_fin", "ablate", "steer"]
)
cs = ctrl(sel) if len(sel) < 2000 else None
for mode in MODES:
    if mode.startswith("clean"):
        evaluate(mode, sel)
        continue
    evaluate(mode, sel)
    if cs is not None:
        print("    control (frequency-matched random latents):", end="")
        evaluate(mode, cs)
