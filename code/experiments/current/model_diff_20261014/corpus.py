# ruff: noqa: E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Build the shared token corpus (ids + per-token structural labels) that every SAE / crosscoder is trained and scored on.

Sources (each record is prompt + a Lean-accepted proof, names shifted by a random lean_seq offset as in training):
  pre   pretraining records (<= 6 body lines, with distractors)                      25,000
  gold  gold proofs of RL-pool targets (7-80 body lines; long, never seen in training)  8,000
  ei1   EI s1's RL-found proofs (shortest per target)                                   12,000
  g0    GRPO s0's (wall) RL-found proofs                                                  8,000
  g1    GRPO s1's (no wall) RL-found proofs                                               5,000
plus `lad3`: every ladder target with a 3-deep implication goal, with EI s1's proof (the wall family), ~1,800.
Records longer than 1,024 tokens are dropped.

Output out/corpus.pt: ids (int16, concatenated), starts (record offsets), n_prompt, src (record source index), and per
token (proof tokens only meaningful): depth, kind index, pop_k (k for the `exact` after a box close, else -1),
close_k (k for the `)` closing box k after a pop name, else -1), pos (offset in proof), spine (goal implication depth
of the record's theorem, -1 unknown).
"""

import collections
import contextlib
import gzip
import io
import json
import random
import sys

import nd
import torch

A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as S

rng = random.Random(0)
tok = nd.load(nd.CKPT["g0_pre"])[1]
KIND = sorted(set(nd.KINDS.values()) | {"name", "atom", "conn", "prem"})
SRCS = ["pre", "gold", "ei1", "g0", "g1", "lad3"]


def found(fn):
    d = {}
    for r in map(json.loads, open(fn)):
        if r["name"] not in d or r["written"] < d[r["name"]]["written"]:
            d[r["name"]] = r
    return d


R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
pre = [json.loads(l) for l in gzip.open(f"{nd.E}/data/pretrain.jsonl.gz", "rt")]
ei1 = found(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")
g0 = found(f"{A}/v3/long-rl-20261014-v3_untied-grpo_s0/found_targets.jsonl")
g1 = found(f"{A}/v3/long-rl-20261014-v3_untied-grpo_s1/found_targets.jsonl")
lad3 = sorted(
    n
    for n, r in R.items()
    if r["src"].startswith("ladder") and S.spine(S.goal(r)) == 3 and n in ei1
)
lad3_set = set(lad3)


def pick(d, n, exclude=()):
    keys = sorted(k for k in d if k not in exclude)
    return [d[k] for k in rng.sample(keys, min(n, len(keys)))]


plan = [
    ("pre", rng.sample(pre, 25000)),
    ("gold", [R[k] for k in rng.sample(sorted(R), 8000)]),
    ("ei1", pick(ei1, 12000, lad3_set)),
    ("g0", pick(g0, 8000, lad3_set)),
    ("g1", pick(g1, 5000, lad3_set)),
    ("lad3", [ei1[n] for n in lad3]),
]


def spine_of(prompt):
    try:
        return S.spine(prompt.split(" SEQ ")[1].replace(" PRF", ""))
    except Exception:
        return -1


def stack_events(s, n_p):
    """pop_k / close_k per position (see module doc)"""
    pop = {}
    close = {}
    stack, opener = [], []
    for i in range(n_p, len(s) - 3):
        t = s[i]
        if t == "(":
            box = s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            if box:
                opener.append(i)
        elif t == ")" and stack and stack.pop():
            opener.pop()
            if s[i + 1] == ";" and s[i + 2] == "exact":
                k = len(opener)
                pop[i + 2] = k
                if k >= 1:
                    close[i + 4] = k
    return pop, close


ids_all, starts, n_prompts, srcs, spines = [], [], [], [], []
lab = collections.defaultdict(list)
n_drop = 0
for src, recs in plan:
    for r in recs:
        try:
            ids, n_p = nd.encode(tok, r["prompt"], r["proof"], 0)
        except Exception:
            n_drop += 1
            continue
        ids, n_p = nd.encode(tok, r["prompt"], r["proof"], nd.rand_shift(tok, ids, rng))
        if len(ids) > 1024:
            n_drop += 1
            continue
        L = nd.labels(tok, ids, n_p)
        s = [tok.itos[x] for x in ids]
        pop, close = stack_events(s, n_p)
        starts.append(len(ids_all))
        n_prompts.append(n_p)
        srcs.append(SRCS.index(src))
        spines.append(spine_of(r["prompt"]))
        ids_all += ids
        lab["depth"] += [0] * n_p + [l["depth"] for l in L]
        lab["kind"] += [-1] * n_p + [KIND.index(l["kind"]) for l in L]
        lab["pop_k"] += [pop.get(i, -1) for i in range(len(ids))]
        lab["close_k"] += [close.get(i, -1) for i in range(len(ids))]
        lab["pos"] += [-1] * n_p + list(range(len(ids) - n_p))
        lab["in_f"] += [0] * n_p + [int(l["in_f"]) for l in L]
out = {
    "ids": torch.tensor(ids_all, dtype=torch.int16),
    "starts": torch.tensor(starts),
    "n_prompt": torch.tensor(n_prompts),
    "src": torch.tensor(srcs),
    "spine": torch.tensor(spines),
    "SRCS": SRCS,
    "KIND": KIND,
    "itos": tok.itos,
    **{k: torch.tensor(v, dtype=torch.int16) for k, v in lab.items()},
}
torch.save(out, "out/corpus.pt")
print(
    f"records {len(starts)} (dropped {n_drop}), tokens {len(ids_all):,}",
    "by src",
    dict(collections.Counter(SRCS[s] for s in srcs)),
)
d = out["depth"]
print(
    "proof tokens by depth",
    {k: int((d == k).sum()) for k in range(6)},
    "pops by k",
    {k: int((out["pop_k"] == k).sum()) for k in range(5)},
)
