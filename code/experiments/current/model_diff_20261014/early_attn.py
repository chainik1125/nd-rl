# ruff: noqa: E501,E741  (research scripts: long result lines, bare open(), short loop names)
"""Blocks 0-3 attention at close-run tokens (the `)` that closes a box, and the following `;`), pre vs fin.
Key positions are grouped by role relative to the box stack at the query:
  run      tokens of the current close run before the query (earlier `exact n )` `;` groups of this run)
  match    the `( fun (` header tokens of the box this `)` closes (for `;`: of the box just closed)
  open     header tokens of boxes still open after this close
  closedB  header tokens of boxes closed earlier (outside this run)
  self, other
Heads 0-3 have ALiBi (recency bias), 4-7 NoPE (content only). Only runs with D_start >= 2. Reports mean mass per head
for r = 1 and r = 2 queries. python early_attn.py g0_pre g0_fin ...
"""

import collections
import random
import sys

import torch
from tiny import N_HEAD, D, Tiny

torch.set_num_threads(8)
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
npr = C["n_prompt"].tolist()
recs = [
    i
    for i in range(len(starts) - 1)
    if C["src"][i] in (C["SRCS"].index("lad3"), C["SRCS"].index("ei1"))
]
random.Random(0).shuffle(recs)
recs = recs[:400]
ROLES = ["run", "match", "open", "closedB", "self", "other"]


def analyse(s, n_p):
    """-> list of (query pos, kind ')' or ';', D_start, r, role array over key positions)"""
    T = len(s)
    out = []
    stack, boxes, open_ids = [], [], []
    run_start, run_D, run_r = None, None, 0
    for t in range(n_p, T):
        if s[t] == "(":
            b = t + 2 < T and s[t + 1] == "fun" and s[t + 2] == "("
            stack.append(b)
            if b:
                bid = len(boxes)
                boxes.append((t, t + 3))
                open_ids.append(bid)
        elif s[t] == ")" and stack and stack.pop():
            bid = open_ids.pop()
            # close run bookkeeping: a run continues if the previous close was 4 tokens back (`) ; exact n )`)
            if (
                run_start is not None
                and t - 4 >= 0
                and s[t - 4] == ")"
                and s[t - 3] == ";"
                and s[t - 2] == "exact"
            ):
                run_r += 1
            else:
                run_start, run_D, run_r = t - 2, len(open_ids) + 1, 1
            role = torch.full((t + 2,), ROLES.index("other"))
            for b2, (a, e) in enumerate(boxes):
                if b2 == bid:
                    role[a : e + 1] = ROLES.index("match")
                elif b2 in open_ids:
                    role[a : e + 1] = ROLES.index("open")
                elif a < run_start:
                    role[a : e + 1] = ROLES.index("closedB")
            role[run_start:t] = ROLES.index("run")
            role_semi = role.clone()
            role[t] = ROLES.index("self")
            if run_D >= 2:
                out.append((t, ")", run_D, run_r, role[: t + 1]))
                if t + 1 < T and s[t + 1] == ";":
                    role_semi[t] = ROLES.index("run")
                    role_semi[t + 1] = ROLES.index("self")
                    out.append((t + 1, ";", run_D, run_r, role_semi[: t + 2]))
    return out


@torch.no_grad()
def attn(m, x, layers):
    T = x.shape[1]
    bias = m.bias(T, x.device)
    hs = m.resid(x)
    out = {}
    for l in layers:
        a = m._ln(hs[l], f"blocks.{l}.ln1")
        q, k, _ = (
            m._lin(a, f"blocks.{l}.qkv").view(1, T, 3, N_HEAD, D // N_HEAD).permute(2, 0, 3, 1, 4)
        )
        out[l] = torch.softmax((q @ k.transpose(-1, -2)) / (D // N_HEAD) ** 0.5 + bias, -1)[0]
    return out


LAYERS = [0, 1, 2, 3]
for mname in sys.argv[1:]:
    r_, st = mname.split("_")
    m = Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[r_]}/{'pretrain' if st == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    agg = collections.defaultdict(lambda: torch.zeros(len(LAYERS), N_HEAD, len(ROLES)))
    n = collections.Counter()
    for i in recs:
        x = ids[starts[i] : starts[i + 1]]
        s = [itos[t] for t in x.tolist()]
        ev = analyse(s, npr[i])
        if not ev:
            continue
        A = attn(m, x[None], LAYERS)
        for q, kind, _D0, r, role in ev:
            key = (kind, min(r, 2))
            n[key] += 1
            for li, l in enumerate(LAYERS):
                w = A[l][:, q, : len(role)]  # H, keys
                for ri in range(len(ROLES)):
                    agg[key][li, :, ri] += w[:, role == ri].sum(-1)
    print(
        f"\n== {mname}: mean attention mass at close-run tokens (D_start>=2). heads 0-3 ALiBi, 4-7 NoPE"
    )
    for key in sorted(n):
        v = agg[key] / n[key]
        print(
            f" query {key[0]!r} r={key[1]}{'+' if key[1] == 2 else ''} (n={n[key]}): heads with >=0.25 on run/match/open/closedB"
        )
        cells = []
        for li, l in enumerate(LAYERS):
            for h in range(N_HEAD):
                for ri, rn in enumerate(ROLES[:4]):
                    if v[li, h, ri] >= 0.25:
                        cells.append(
                            f"{l}.{h}{'A' if h < 4 else 'N'} {rn} {float(v[li, h, ri]):.2f}"
                        )
        print("   " + "  ".join(cells))
