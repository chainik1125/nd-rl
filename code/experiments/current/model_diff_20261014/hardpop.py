# ruff: noqa: E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Pop accuracy on easy (k = max(0, 2 - r)) vs hard pops, plus key (k, r) cells, for any list of checkpoints.
Same 1,000 corpus records for every checkpoint. python hardpop.py <ckpt.pt> ...  (paths relative to ckpt/runs/long-rl-20261014-)"""

import collections
import random
import sys

import torch
from tiny import Tiny

torch.set_num_threads(8)
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
recs = recs[:1000]
ev = collections.defaultdict(list)
for i in recs:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    p = pk[starts[i] : starts[i + 1]]
    for j in (p >= 0).nonzero().flatten().tolist():
        if int(p[j]) == 0:
            continue
        r, q = 1, j - 2
        while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
            r += 1
            q -= 4
        ev[i].append((j, int(p[j]), min(r, 3)))
CELLS = [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1)]
print(f"{'checkpoint':>34} {'easy':>5} {'hard':>5} " + " ".join(f"k{k}r{r:<3}" for k, r in CELLS))
OUT = {}
JSON = None
args = sys.argv[1:]
if args and args[0].startswith("--json="):
    JSON = args.pop(0).split("=", 1)[1]
for path in args:
    c = torch.load(f"ckpt/runs/long-rl-20261014-{path}", map_location="cpu")
    m = Tiny(c.get("state", c))
    hit = collections.Counter()
    tot = collections.Counter()
    with torch.no_grad():
        for i, lst in ev.items():
            x = ids[starts[i] : starts[i + 1]]
            am = m.logits_from(m.resid(x[None])[-1], 6)[0].argmax(-1)
            for j, k, r in lst:
                ok = int(am[j] == x[j + 1])
                tot[(k, r)] += 1
                hit[(k, r)] += ok
                key = "easy" if k == max(0, 2 - r) else "hard"
                tot[key] += 1
                hit[key] += ok
    OUT[path] = {
        "easy": hit["easy"] / tot["easy"],
        "hard": hit["hard"] / tot["hard"],
        **{f"k{k}r{r}": hit[(k, r)] / tot[(k, r)] for k, r in CELLS},
    }
    print(
        f"{path:>34} {hit['easy'] / tot['easy']:5.0%} {hit['hard'] / tot['hard']:5.0%} "
        + " ".join(f"{hit[c_] / tot[c_]:6.0%}" for c_ in CELLS),
        flush=True,
    )

if JSON:
    import json

    json.dump(OUT, open(JSON, "w"), indent=1)
