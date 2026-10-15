# ruff: noqa: E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Residual-stream patching between a pretrained model (pre) and its RL'd final model (fin) on spine-3 pops.
Take fin's residual stream after block l at a SET of positions, write it into pre's forward, finish with pre's blocks
l..5 (and the reverse direction). Position sets:
  all      every token
  hdr      box-header tokens: `have nX : <type> := ( fun ( nB : <type> ) => by` of every box
  close    the closing run `exact nE )` `;` and the pop `exact` itself
  pop      the pop `exact` only
  rest     all - hdr - close
Metric: pop accuracy at k = 1 and k = 2 (k = depth returned to). python patch_resid.py g0 -> out/patch_resid_g0.json
"""

import json
import sys

import torch
from tiny import Tiny

torch.set_num_threads(6)
pair = sys.argv[1] if len(sys.argv) > 1 else "g0"
NREC = int(sys.argv[2]) if len(sys.argv) > 2 else 150
C = torch.load("out/corpus.pt")
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
M = {
    s: Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[pair]}/{'pretrain' if s == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    for s in ("pre", "fin")
}
starts = C["starts"].tolist() + [len(C["ids"])]
itos = C["itos"]
LAD3 = C["SRCS"].index("lad3")
lad = [i for i in range(len(starts) - 1) if C["src"][i] == LAD3][:NREC]
ids = C["ids"].long()
pk = C["pop_k"].long()
npr = C["n_prompt"].tolist()


def possets(x, n_p):
    s = [itos[t] for t in x.tolist()]
    T = len(s)
    hdr = torch.zeros(T, dtype=torch.bool)
    close = torch.zeros(T, dtype=torch.bool)
    pop = torch.zeros(T, dtype=torch.bool)
    last_have = None
    for i in range(n_p, T - 3):
        if s[i] == "have":
            last_have = i
        if s[i] == "(" and s[i + 1] == "fun" and s[i + 2] == "(":
            j = i
            while s[j] != "by":
                j += 1
            hdr[last_have : j + 1] = True
        if s[i] == "exact" and s[i - 1] == ";" and s[i - 2] == ")":
            pop[i] = True
            close[i] = True
            close[i - 1] = True
        if s[i] == ")" and s[i - 2] == "exact":
            close[i] = True
            close[i - 1] = True
            close[i - 2] = True
    allp = torch.ones(T, dtype=torch.bool)
    return {"all": allp, "hdr": hdr, "close": close, "pop": pop, "rest": allp & ~hdr & ~close}


@torch.no_grad()
def run():
    res = {}
    cache = []
    for i in lad:
        x = ids[starts[i] : starts[i + 1]]
        cache.append(
            (
                x,
                possets(x, npr[i]),
                {s: M[s].resid(x[None])[:, 0] for s in M},
                pk[starts[i] : starts[i + 1]],
            )
        )
    for base, donor in (("pre", "fin"), ("fin", "pre")):
        for l in range(0, 6):
            for ps in ("all", "hdr", "close", "pop", "rest"):
                hit = {1: 0, 2: 0}
                tot = {1: 0, 2: 0}
                for x, P, H, p in cache:
                    h = torch.where(P[ps][:, None], H[donor][l], H[base][l])
                    lg = M[base].logits_from(h[None], l)[0]
                    for j in (p >= 1).nonzero().flatten().tolist():
                        k = min(int(p[j]), 2)
                        tot[k] += 1
                        hit[k] += int(lg[j].argmax() == x[j + 1])
                res[f"{base}<-{donor}|L{l}|{ps}"] = [hit[1] / tot[1], hit[2] / tot[2]]
            print(
                f"{base} <- {donor} resid after block {l}: "
                + "  ".join(
                    f"{ps} {res[f'{base}<-{donor}|L{l}|{ps}'][0]:.0%}/{res[f'{base}<-{donor}|L{l}|{ps}'][1]:.0%}"
                    for ps in ("all", "hdr", "close", "pop", "rest")
                ),
                flush=True,
            )
    json.dump(res, open(f"out/patch_resid_{pair}.json", "w"), indent=1)


print(
    "cells: pop accuracy k=1 / k=2 after patching that position set (resid index l = after block l-1; 0 = embeddings)"
)
run()
