# ruff: noqa: E402,E501  (research scripts: long result lines, bare open(), short loop names)
"""Pop accuracy by (k = true depth returned to, r = position of this close in a run of consecutive box closes).
Hypothesis (pretrained heuristic): after the r-th consecutive close the model acts as if depth = D0 - r for some fixed
D0 (2, from depth<=2 pretraining proofs), instead of tracking opens - closes. Then pre is right iff k == D0 - r.
python runpos.py [models] -> table over all corpus pops with k>=0 (skips records > 700 tokens for speed)"""

import collections
import sys

import torch
from tiny import Tiny

torch.set_num_threads(6)
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
MS = sys.argv[1:] or ["g0_pre", "g0_fin", "g1_fin", "e1_pre", "e1_fin"]
C = torch.load("out/corpus.pt")
starts = C["starts"].tolist() + [len(C["ids"])]
itos = C["itos"]
ids = C["ids"].long()
pk = C["pop_k"].long()
# records with at least one pop at k>=1, sample
recs = [
    i
    for i in range(len(starts) - 1)
    if starts[i + 1] - starts[i] <= 700 and (pk[starts[i] : starts[i + 1]] >= 1).any()
]
import random

random.Random(0).shuffle(recs)
recs = recs[:1500]
events = []  # (rec, j, k, r)
for i in recs:
    x = ids[starts[i] : starts[i + 1]]
    s = [itos[t] for t in x.tolist()]
    p = pk[starts[i] : starts[i + 1]]
    for j in (p >= 0).nonzero().flatten().tolist():
        # count consecutive closes: pattern ... exact nA ) ; exact nB ) ; exact  -> walk back over ") ; exact name"
        r, q = 1, j - 2  # s[q] == ')'
        while q - 4 >= 0 and s[q - 2] == "exact" and s[q - 3] == ";" and s[q - 4] == ")":
            r += 1
            q -= 4
        events.append((i, j, int(p[j]), r))
print("pops", len(events), collections.Counter((k, r) for _, _, k, r in events).most_common(12))
for mname in MS:
    run, st = mname.split("_")
    m = Tiny(
        torch.load(
            f"ckpt/runs/long-rl-20261014-{CK[run]}/{'pretrain' if st == 'pre' else 'final'}.pt",
            map_location="cpu",
        )["state"]
    )
    hit = collections.Counter()
    tot = collections.Counter()
    by = collections.defaultdict(list)
    for e in events:
        by[e[0]].append(e)
    with torch.no_grad():
        for i, evs in by.items():
            x = ids[starts[i] : starts[i + 1]]
            lg = m.logits_from(m.resid(x[None])[-1], 6)[0]
            for _, j, k, r in evs:
                tot[(k, r)] += 1
                hit[(k, r)] += int(lg[j].argmax() == x[j + 1])
    print(f"\n{mname}: pop accuracy by k (rows) x r = close index in run (cols), n in brackets")
    for k in range(0, 4):
        print(
            f"  k={k} "
            + "  ".join(
                f"r={r}: {hit[(k, r)] / tot[(k, r)]:4.0%} ({tot[(k, r)]:4d})"
                if tot[(k, r)] >= 5
                else f"r={r}:    -      "
                for r in range(1, 5)
            )
        )
