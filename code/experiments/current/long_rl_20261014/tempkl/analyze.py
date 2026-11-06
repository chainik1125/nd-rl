#!/usr/bin/env python3
"""Temperature-matched KL: the fit, from tempkl.py's per-rollout sums (no torch, no fork; runs on the Mac).

For a group of rollouts, with per-token sums over the group's proof tokens, the mean KL(pi_RL || softmax(z_pre / T))
is an exact function of T:

    KL(T) = ( -sum H(pi_RL) - sum E_{pi_RL}[z_pre] / T + sum logsumexp(z_pre / T) ) / n_tokens

tempkl.py stores the first two sums and the third on the LOG_T grid (601 points, 1% steps in T, T = 1 at index 300),
so any grouping of rollouts (checkpoint, band, Lean verdict) is a sum of rows. fit() takes the grid minimum and refines
it with a parabola through the three neighbouring grid points (in log T), which is exact to well under the grid step.

  python tempkl/analyze.py results/*.json      # prints the per-checkpoint table and writes summary.json
"""

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_T = np.linspace(math.log(0.05), math.log(20), 601)
assert abs(LOG_T[300]) < 1e-12  # T = 1 sits on the grid
RL_TEMP = 0.8  # the harness's sampling temperature; rows also carry pi_RL at this temperature (suffix "_t")
BANDS = ("7-9", "10-14", "15-19", "20-24", "25-34", "35-49", "50+")


def curve(rows, policy=""):
    """Mean per-token KL(pi_RL || pi_pre^T) on the LOG_T grid for these rows. policy "" is pi_RL at T = 1,
    "_t" is pi_RL at the sampling temperature RL_TEMP."""
    n = sum(r["n"] for r in rows)
    H = sum(r["H" + policy] for r in rows)
    Ez = sum(r["Ez" + policy] for r in rows)
    lse = np.sum([r["lse"] for r in rows], axis=0, dtype=np.float64)
    return (-H - Ez / np.exp(LOG_T) + lse) / n


def fit(rows, policy=""):
    """{raw, t_star, residual, ...} for a group of rollout rows: raw KL at T = 1, the minimising temperature and the
    KL there (nats per token), plus nats per proof, entropies and token counts."""
    if not rows:
        return None
    kl = curve(rows, policy)
    i = int(np.argmin(kl))
    if 0 < i < len(kl) - 1:  # parabola through the three points around the grid minimum, in log T
        y0, y1, y2 = kl[i - 1], kl[i], kl[i + 1]
        denom = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / denom if denom > 0 else 0.0
        log_t = LOG_T[i] + d * (LOG_T[1] - LOG_T[0])
        resid = y1 - 0.25 * (y0 - y2) * d
    else:  # at an edge of the grid: report the edge, flagged
        log_t, resid = LOG_T[i], kl[i]
    n, m = sum(r["n"] for r in rows), len(rows)
    return {"raw": float(kl[300]), "t_star": float(math.exp(log_t)), "residual": float(resid),
            "raw_per_proof": float(kl[300] * n / m), "residual_per_proof": float(resid * n / m),
            "at_edge": i in (0, len(kl) - 1), "ent_rl": sum(r["H"] for r in rows) / n,
            "ent_pre": sum(r["Hpre"] for r in rows) / n, "nll_rl": sum(r["nll_rl"] for r in rows) / n,
            "nll_pre": sum(r["nll_pre"] for r in rows) / n, "tokens_per_proof": n / m, "n_rollouts": m,
            "n_tokens": n, "accepted": sum(r["ok"] for r in rows)}


def summarize(res):
    """All the groupings of one checkpoint's rows."""
    rows = res["rows"]
    out = {"tag": res["tag"], "pre_tag": res.get("pre_tag"), "all": fit(rows), "all_t": fit(rows, "_t"),
           "band": {b: fit([r for r in rows if r["band"] == b]) for b in BANDS},
           "ok": {str(ok): fit([r for r in rows if r["ok"] == ok]) for ok in (True, False)},
           "band_ok": {b: {str(ok): fit([r for r in rows if r["band"] == b and r["ok"] == ok]) for ok in (True, False)}
                       for b in BANDS}}
    return out


def table(summaries):
    hdr = f"{'checkpoint':28} {'raw':>7} {'T*':>6} {'resid':>7} {'resid%':>7} {'H_rl':>6} {'H_pre':>6} {'raw/proof':>9} {'acc':>7}"
    lines = [hdr]
    for s in summaries:
        a = s["all"]
        pct = f"{100 * a['residual'] / a['raw']:6.1f}%" if a["raw"] > 1e-6 else "      -"  # pretrain vs itself: 0/0
        lines.append(f"{s['tag']:28} {a['raw']:7.4f} {a['t_star']:6.3f} {a['residual']:7.4f} "
                     f"{pct} {a['ent_rl']:6.3f} {a['ent_pre']:6.3f} "
                     f"{a['raw_per_proof']:9.2f} {a['accepted']:3d}/{a['n_rollouts']}")
    return "\n".join(lines)


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    summaries = [summarize(json.load(open(f))) for f in files]
    print(table(summaries))
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)


if __name__ == "__main__":
    main()
