"""Task 1 of hard12: which features predict a low T0 pass rate at fixed reference length (7-12
lines)?

    python -m code.experiments.current.thinking_traces_20261005.pools.hard12.analyze \
        --sets runs/pools/short_pilot.jsonl runs/pools/short_cands.jsonl \
        --rows runs/pools/passrate/short_pilot/rows.jsonl runs/pools/passrate/short/rows.jsonl \
        --k12 /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl \
        --out feature_analysis.json

Data: the previous pool agent's T0_s0 measurements (k 16, T 0.8, plain, Lean-judged;
`pools/passrate.py`) on uniformly drawn strict-long-generator candidates of 7-12 lines. Outputs
(JSON):
  * `binned`: for each feature value, n, mean pass@1 and the share in the hard band (1-8 of 16
    successes, the k 16 analogue of 2-16 of 32), each also length-adjusted (difference to the
    length-matched mean, averaged over lengths with the bin's own length mix);
  * `glm`: a binomial GLM (logit link) on n_ok of k with length fixed effects and the features
    jointly; coefficients are log-odds of one attempt succeeding (negative = harder), with standard
    errors (IRLS, numpy);
  * `fail_status`: the attempt-level failure statuses of hard vs easy targets;
  * `k12_vs_measured`: feature means in K12 train (a sample) and in the measured candidates.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import random
import sys
from typing import Any

import numpy as np

from code.thinking.data import read_jsonl

from .features import features

BIN_FEATS = [
    "n_prem",
    "has_ORE",
    "n_ORE",
    "has_NEGI",
    "has_DN",
    "raa",
    "has_BOTE",
    "has_NEGE",
    "has_IMPI",
    "has_IMPE",
    "has_ANDE",
    "has_ANDI",
    "has_ORI",
    "has_R",
    "classical_only",
    "n_redundant_prem",
    "prem_inconsistent",
    "n_disj_prem",
    "n_prem_or",
    "n_prem_imp",
    "n_imp_hard_ante",
    "goal_type",
    "fdepth",
    "n_atoms",
    "box_depth",
    "top_rule",
    "top_goal_directed",
    "first_kind",
    "first_goal_directed",
    "n_boxes",
]
GLM_FEATS = [
    "has_ORE",
    "has_NEGI",
    "has_DN",
    "has_BOTE",
    "has_NEGE",
    "has_ANDI",
    "has_ORI",
    "has_IMPE",
    "has_ANDE",
    "classical_only",
    "n_redundant_prem",
    "n_disj_prem",
    "n_imp_hard_ante",
    "n_prem",
    "fdepth",
    "n_atoms",
    "box_depth",
    "top_goal_directed",
    "first_kind=as_case",
    "first_kind=as_raa",
    "first_kind=elim",
    "first_kind=as_other",
    "goal=atom",
    "goal=or",
    "goal=and",
    "goal=not",
]


def glm_binomial(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = n * p * (1 - p) + 1e-9
        z = eta + (y - n * p) / w
        XtW = X.T * w
        H = XtW @ X + 1e-6 * np.eye(X.shape[1])
        b_new = np.linalg.solve(H, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    p = 1 / (1 + np.exp(-(X @ b)))
    w = n * p * (1 - p) + 1e-9
    cov = np.linalg.inv((X.T * w) @ X + 1e-6 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def design(fs: list[dict[str, Any]], lens: list[int]) -> tuple[np.ndarray, list[str]]:
    cols = [f"len={L}" for L in lens]
    rows = []
    for f in fs:
        r = [float(f["n_lines"] == L) for L in lens]
        for g in GLM_FEATS:
            if g.startswith("first_kind="):
                r.append(float(f["first_kind"] == g.split("=")[1]))
            elif g.startswith("goal="):
                r.append(float(f["goal_type"] == g.split("=")[1]))
            else:
                r.append(float(f[g]))
        rows.append(r)
    return np.array(rows), cols + GLM_FEATS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--k12", required=True)
    ap.add_argument("--k12_sample", type=int, default=20000)
    ap.add_argument("--lens", default="7-12")
    ap.add_argument(
        "--max_fdepth", type=int, default=99, help="keep candidates with formula depth <= this"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.lens.split("-"))
    rows = {r["name"]: r for p in a.rows for r in read_jsonl(p)}
    recs = [
        r for p in a.sets for r in read_jsonl(p) if r["name"] in rows and lo <= r["n_lines"] <= hi
    ]
    data = []
    for r in recs:
        f = features(r["prompt"], r["proof"])
        row = rows[r["name"]]
        f["n_ok"], f["k"] = row["n_ok"], row["k"]
        f["status"] = row["status"]
        if f["fdepth"] <= a.max_fdepth:
            data.append(f)
    k = data[0]["k"]
    band_hi = k // 2
    for f in data:
        f["p1"] = f["n_ok"] / f["k"]
        f["band"] = int(1 <= f["n_ok"] <= band_hi)
    lens = sorted({f["n_lines"] for f in data})
    by_len = {L: [f for f in data if f["n_lines"] == L] for L in lens}
    base = {
        L: (np.mean([f["p1"] for f in fs]), np.mean([f["band"] for f in fs]))
        for L, fs in by_len.items()
    }
    binned: dict[str, Any] = {}
    for feat in BIN_FEATS:
        g = collections.defaultdict(list)
        for f in data:
            v = f[feat]
            if isinstance(v, int) and feat.startswith("n_") and v >= 3:
                v = "3+"
            g[str(v)].append(f)
        binned[feat] = {
            v: {
                "n": len(fs),
                "pass1_pct": round(100 * np.mean([f["p1"] for f in fs]), 1),
                "band_pct": round(100 * np.mean([f["band"] for f in fs]), 1),
                "unsolved_pct": round(100 * np.mean([f["n_ok"] == 0 for f in fs]), 1),
                "pass1_adj_pp": round(
                    100 * np.mean([f["p1"] - base[f["n_lines"]][0] for f in fs]), 1
                ),
                "band_adj_pp": round(
                    100 * np.mean([f["band"] - base[f["n_lines"]][1] for f in fs]), 1
                ),
            }
            for v, fs in sorted(g.items())
        }
    X, cols = design(data, lens)
    y = np.array([f["n_ok"] for f in data], float)
    n = np.array([f["k"] for f in data], float)
    b, se = glm_binomial(X, y, n)
    glm = {
        c: {"coef": round(float(bb), 3), "se": round(float(s), 3), "z": round(float(bb / s), 1)}
        for c, bb, s in zip(cols, b, se, strict=True)
    }
    # failure statuses: hard (pass1 <= 0.5) vs easy (> 0.5)
    fail: dict[str, collections.Counter[str]] = {
        "hard": collections.Counter(),
        "easy": collections.Counter(),
    }
    for f in data:
        grp = "hard" if f["p1"] <= 0.5 else "easy"
        for s, c in f["status"].items():
            if s != "done":
                fail[grp][s] += c
    fail_out = {
        g: {s: round(100 * c / max(1, sum(cs.values())), 1) for s, c in cs.most_common(12)}
        | {"attempts": sum(cs.values())}
        for g, cs in fail.items()
    }
    # K12 comparison
    rng = random.Random(0)
    k12 = read_jsonl(a.k12)
    rng.shuffle(k12)
    k12f = []
    for r in k12[: a.k12_sample]:
        with contextlib.suppress(Exception):
            k12f.append(features(r["prompt"], r["proof"]))

    def summ(fs: list[dict[str, Any]]) -> dict[str, Any]:
        o: dict[str, Any] = {"n": len(fs)}
        for feat in BIN_FEATS + ["n_lines", "prompt_tokens"]:
            vals = [f[feat] for f in fs]
            if isinstance(vals[0], str):
                o[feat] = {
                    v: round(100 * c / len(vals), 1)
                    for v, c in collections.Counter(vals).most_common()
                }
            else:
                o[feat] = round(float(np.mean(vals)), 3)
        return o

    k12_712 = [f for f in k12f if lo <= f["n_lines"] <= hi]
    out = {
        "data": {
            "sets": a.sets,
            "rows": a.rows,
            "n": len(data),
            "k": k,
            "band": f"1-{band_hi} of {k}",
            "max_fdepth": a.max_fdepth,
        },
        "base_by_len": {
            str(L): {
                "n": len(by_len[L]),
                "pass1_pct": round(100 * base[L][0], 1),
                "band_pct": round(100 * base[L][1], 1),
            }
            for L in lens
        },
        "binned": binned,
        "glm": {
            "model": (
                "binomial logit, n_ok of k ~ length fixed effects + features (log-odds of one "
                "attempt succeeding)"
            ),
            "coef": glm,
        },
        "fail_status_pct": fail_out,
        "k12_vs_measured": {
            "k12_all": summ(k12f),
            "k12_7_12": summ(k12_712),
            "measured_7_12": summ(data),
            "measured_band": summ([f for f in data if f["band"]]),
        },
    }
    with open(a.out, "w") as fo:
        json.dump(out, fo, indent=1)
    srt = sorted(
        ((c, v) for c, v in glm.items() if not c.startswith("len=")), key=lambda x: x[1]["coef"]
    )
    for c, v in srt:
        print(f"{c:24s} {v['coef']:+.3f} ± {v['se']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
