"""Gate summary of the base read: propensity and capability under tag A, B and no tag (rates only).

    python tag_gate_summary.py

Reads AT/reads/pc/results/tagAB_s{0,1,2}@{A,B,none}.jsonl. Per measure and seed: mean over theorems under each
tag, and A - B with a 95 % paired bootstrap over theorems. Written after the read (2026-11-10), not pre-registered code.
"""

# ruff: noqa: E501, E701, E702
import json
import math

import numpy as np

R = "/scratch/cowifornia/nd-rl/author_tags/reads/pc/results/"
rng = np.random.default_rng(0)


def load(s, t):
    d = {}
    for line in open(f"{R}tagAB_s{s}@{t}.jsonl"):  # noqa: SIM115
        r = json.loads(line)
        d[(r["set"], r["name"], r["arm"], r["direction"])] = r
    return d


def vec(d, st, arm, dr, f):
    return {
        k[1]: f(r)
        for k, r in d.items()
        if k[0] == st and k[2] == arm and k[3] == dr and r.get("applicable", True)
    }


M = {
    "solve U": ("U", "any", lambda r: r["solved"] / r["k"]),
    "uses red": ("U", "reductio", lambda r: r["used"] / r["k"]),
    "opens red": ("U", "reductio", lambda r: r["first"] / r["k"]),
    "P(open) E": ("E", "reductio", lambda r: math.exp(r["logp_open"])),
    "logP E": ("E", "reductio", lambda r: r["logp_open"]),
    "forced F": ("F", "reductio", lambda r: r["solved"] / r["k"]),
}


def ci(x):
    x = np.array(x)
    n = len(x)
    b = x[rng.integers(0, n, (4000, n))].mean(1)
    return np.percentile(b, [2.5, 97.5])


for st in ("strict106", "transfer_other"):
    print("==", st)
    for m, (arm, dr, f) in M.items():
        for s in (0, 1, 2):
            v = {t: vec(load(s, t), st, arm, dr, f) for t in ("A", "B", "none")}
            names = sorted(v["A"])
            assert names == sorted(v["B"]) == sorted(v["none"])
            sc = 1 if m == "logP E" else 100
            a, b, n = [np.array([v[t][x] for x in names]) * sc for t in ("A", "B", "none")]
            lo, hi = ci(a - b)
            print(
                f"{m:10s} s{s} n={len(names)}  A {a.mean():7.2f}  B {b.mean():7.2f}  none {n.mean():7.2f}  A-B {a.mean() - b.mean():+6.2f} [{lo:+.2f},{hi:+.2f}]"
            )
