"""Gate summary of the base read: propensity and capability under tag A, B and no tag (rates only).

    python tag_gate_summary.py [--variant p10|p50|p100] [--seeds 0,1,2] [--root DIR]

Reads <root>/results/<base>@{A,B,none}.jsonl: by default AT/reads/pc/results/tagAB_s{0,1,2}@... (the default bases),
with --variant AT/reads/pc_<variant>/results/tagAB_<variant>_s{0,1,2}@... (tag_core.read_root, tag_core.read_id). Per
measure and seed: mean over theorems under each tag, and A - B with a 95 % paired bootstrap over theorems. A seed with
no read under some tag is listed once as missing and left out (the default bases are all read).
Written after the read (2026-11-10), not pre-registered code; the default output is unchanged by the --variant option.
"""

# ruff: noqa: E501, E701, E702
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

TAGS = ("A", "B", "none")


def load(R, s, t, variant=None):
    d = {}
    for line in open(Path(R) / f"{T.read_id(s, t, variant)}.jsonl"):  # noqa: SIM115
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


def ci(rng, x):
    x = np.array(x)
    n = len(x)
    b = x[rng.integers(0, n, (4000, n))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", choices=sorted(T.VARIANTS), default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default=None, help="default: tag_core.read_root(variant)")
    a = ap.parse_args(argv)
    R = (Path(a.root) if a.root else T.read_root(a.variant)) / "results"
    seeds = [int(s) for s in a.seeds.split(",")]
    have = [
        s for s in seeds if all((R / f"{T.read_id(s, t, a.variant)}.jsonl").exists() for t in TAGS)
    ]
    if a.variant is not None:
        print(
            f"variant {a.variant} ({R}): bases {', '.join(T.base_name(s, a.variant) for s in seeds)}"
        )
    if len(have) < len(seeds):
        miss = [T.base_name(s, a.variant) for s in seeds if s not in have]
        print(f"missing (no read under every tag of {', '.join(TAGS)}): {', '.join(miss)}")
    rng = np.random.default_rng(0)
    for st in ("strict106", "transfer_other"):
        print("==", st)
        for m, (arm, dr, f) in M.items():
            for s in have:
                v = {t: vec(load(R, s, t, a.variant), st, arm, dr, f) for t in TAGS}
                names = sorted(v["A"])
                assert names == sorted(v["B"]) == sorted(v["none"])
                sc = 1 if m == "logP E" else 100
                x, y, n = [np.array([v[t][q] for q in names]) * sc for t in TAGS]
                lo, hi = ci(rng, x - y)
                print(
                    f"{m:10s} s{s} n={len(names)}  A {x.mean():7.2f}  B {y.mean():7.2f}  none {n.mean():7.2f}  A-B {x.mean() - y.mean():+6.2f} [{lo:+.2f},{hi:+.2f}]"
                )


if __name__ == "__main__":
    main()
