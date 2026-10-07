#!/usr/bin/env python3
"""Curves and table for the temperature sweep, from the pulled pools
(artifacts/temp_sweep_20261006/out).

For each sampling temperature and pool size m, three numbers averaged over theorems (unsolved ones
included):
  oracle    any of m attempts is Lean-valid (pass@m)
  finished  a uniformly random finished attempt among the m is valid
  loglik    the finished attempt with the highest T=1 mean token log-prob is valid
            (`completion_loglik`)
All are exact expectations over m-subsets of the k attempts (hypergeometric), not Monte Carlo.

  uv run --with numpy --with matplotlib analyze.py
"""

import json
import os
from math import comb

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "temp_sweep_20261006", "out")
TEMPS = [0, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2]
SETS = ["holdout250", "textbook72"]
MS = [1, 2, 4, 8, 16, 32, 64]
RULES = {
    "oracle": "oracle: any of m valid (pass@m)",
    "loglik": "finished, highest likelihood",
    "finished": "a random finished attempt",
}
# one-hue ordinal ramp, light to dark with temperature (dataviz reference palette, blue 250-700)
RAMP = dict(
    zip(
        TEMPS[1:],
        ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
        strict=True,
    )
)
MARK = dict(zip(TEMPS[1:], "ov^sDPX", strict=True))
PLOT_TEMPS = TEMPS[1:-1]  # T=2 almost never finishes (table only); it would flatten every panel
YLIM = {
    "holdout250": (0.77, 0.97),
    "textbook72": (0.36, 0.67),
}  # T=1.5 enters from below at small m


def jsonl(fn):
    with open(fn) as f:
        return [json.loads(line) for line in f]


def per_theorem(s, T):
    """rule -> array [theorems, len(ms)] of exact expected accuracy; also the ms that fit in k."""
    k = 1 if T == 0 else 64
    stem = f"{s}__s0_r16_T{T:g}__k{k}"
    pool = jsonl(os.path.join(OUT, "pools", stem + ".jsonl"))
    labels = jsonl(os.path.join(OUT, "pools", stem + ".labels.jsonl"))
    scores = jsonl(os.path.join(OUT, "scores", stem + "__completion_loglik.jsonl"))
    ms = [m for m in MS if m <= k]
    acc = {r: np.zeros((len(pool), len(ms))) for r in RULES}
    for t, (p, lab, sc) in enumerate(zip(pool, labels, scores, strict=True)):
        assert p["name"] == lab["name"] == sc["name"]
        valid = np.array(lab["valid"], bool)
        done = np.array([a["done"] for a in p["attempts"]], bool)
        assert not (valid & ~done).any()
        order = np.argsort(
            -np.array(sc["scores"]), kind="stable"
        )  # unfinished attempts score -inf and sort last
        v, f = int(valid.sum()), int(done.sum())
        for j, m in enumerate(ms):
            tot = comb(k, m)
            acc["oracle"][t, j] = 1 - comb(k - v, m) / tot
            acc["finished"][t, j] = (1 - comb(k - f, m) / tot) * v / f if f else 0.0
            # the attempt ranked r is the subset's best iff it is in and the r better ones are out
            acc["loglik"][t, j] = (
                sum(comb(k - 1 - r, m - 1) for r in range(k) if valid[order[r]]) / tot
            )
    return acc, ms


def ci(x, rng, B=2000):
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    res = {(s, T): per_theorem(s, T) for s in SETS for T in TEMPS}
    fig, axs = plt.subplots(len(RULES), len(SETS), figsize=(11, 11), sharex=True)
    plt.rcParams.update({"font.size": 9})
    for c, s in enumerate(SETS):
        for r, (rule, title) in enumerate(RULES.items()):
            ax = axs[r, c]
            greedy = res[s, 0][0][rule][:, 0].mean()
            ax.axhline(greedy, color="#52514e", ls="--", lw=1.5, label="T=0 (greedy, one attempt)")
            for T in PLOT_TEMPS:
                acc, ms = res[s, T]
                ax.plot(
                    ms,
                    acc[rule].mean(0),
                    color=RAMP[T],
                    lw=2,
                    marker=MARK[T],
                    ms=6,
                    label=f"T={T:g}",
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(MS)
            ax.set_xticklabels(MS)
            ax.set_ylim(*YLIM[s])
            ax.set_title(f"{s}: {title}", fontsize=10)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0 and r == 1:
                ax.set_ylabel("fraction of theorems, selected proof Lean-valid")
            if r == len(RULES) - 1:
                ax.set_xlabel("attempts sampled per theorem (m)")
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=7,
        fontsize=9,
        frameon=False,
        title="sampling temperature (T=2 omitted: under 2% of attempts finish)",
    )
    fig.suptitle(
        "la_T1_best12_s0_r16: accuracy vs attempts at each sampling temperature "
        "(likelihood always scored at T=1)"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fn = os.path.join(HERE, "temp_sweep.png")
    fig.savefig(fn, dpi=150)
    print("wrote", fn)

    lines = []
    for s in SETS:
        summ = {}
        for T in TEMPS:
            stem = f"{s}__s0_r16_T{T:g}__k{1 if T == 0 else 64}"
            with open(os.path.join(OUT, "pools", stem + ".summary.json")) as f:
                summ[T] = json.load(f)
        lines += [
            f"\n### {s}\n",
            "| T | finished | valid | pass@1 | loglik@8 | oracle@8 | loglik@64 | oracle@64 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for T in TEMPS:
            acc, ms = res[s, T]
            cells = [f"{T:g}", f"{summ[T]['finished_rate']:.3f}", f"{summ[T]['validity_rate']:.3f}"]
            for rule, m in [
                ("oracle", 1),
                ("loglik", 8),
                ("oracle", 8),
                ("loglik", 64),
                ("oracle", 64),
            ]:
                if m not in ms:
                    cells.append("")
                    continue
                x = acc[rule][:, ms.index(m)]
                lo, hi = ci(x, rng)
                cells.append(f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))
    with open(os.path.join(HERE, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
