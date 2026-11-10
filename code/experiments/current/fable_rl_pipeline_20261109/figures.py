#!/usr/bin/env python3
"""Figures for the write-up, in the project's figure style (`ndbench.figures.style`).

    python figures.py --runs /scratch/cowifornia/nd-rl/fable_rl/runs \
        --final /scratch/cowifornia/nd-rl/fable_rl/final --out ../../../../experiment-summaries/<folder>

Figure `dev_by_round`: dev pass@1 and pass@5 in plain mode after each scored RL round, three seeds
and the plain expert-iteration ablation. Figure `dev_by_length`: pass@1 by reference proof length
for the pretrained base, the final policy in plain mode and the final model in full mode (mean of
the seeds, with each seed as a dot). Reads only `rounds.jsonl` files and the final dev tables.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # code/

from ndbench.figures import style as S  # noqa: E402

RUNS = [("fr1", "rl_s0", "seed 0"), ("fr2", "rl_s1", "seed 1"), ("fr3", "rl_s2", "seed 2"),
        ("fr4", "abl_plainei_s0", "plain expert iteration (seed-0 base)")]  # fmt: skip
COL = [S.OKABE_ITO["blue"], S.OKABE_ITO["bluish_green"], S.OKABE_ITO["reddish_purple"], S.OKABE_ITO["vermillion"]]
MARK = ["o", "s", "^", "D"]
LS = ["-", "-", "-", "--"]


def rounds(fn):
    out = {}
    for line in open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("dev") or r.get("final_dev")
        if d and "plain" in d:
            rnd = r["round"] + 1 if "dev" in r else r["round"]
            out[rnd] = d["plain"]
    return dict(sorted(out.items()))


def fig_by_round(runs_dir, bases, out):
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.1, layout="constrained")
        axs = fig.subplots(1, 2, sharex=True)
        for (pod, run, label), c, m, ls in zip(RUNS, COL, MARK, LS):
            fn = os.path.join(runs_dir, pod, run, "rounds.jsonl")
            if not os.path.exists(fn):
                continue
            d = rounds(fn)
            b = bases.get(run)
            for ax, key in zip(axs, ("pass@1", "pass@5")):
                xs = ([0] if b else []) + list(d)
                ys = ([b[key]] if b else []) + [d[r]["all"][key] for r in d]
                ax.plot(xs, ys, color=c, marker=m, markersize=2.6, linewidth=1.0, linestyle=ls, label=label)
        for ax, key, letter in zip(axs, ("pass@1", "pass@5"), "ab"):
            S.panel_title(ax, letter, f"dev, plain mode, {key}")
            ax.set_xlabel("RL round (0 = pretrained base)")
            ax.set_ylabel("% of 473 dev theorems")
            S.light_grid(ax)
        axs[1].legend(fontsize=S.FS_TICK, frameon=False, loc="lower right")
        S.save(fig, Path(out) / "dev_by_round")
        S.close(fig)


def fig_by_length(summary, out):
    """summary: {"lengths": [...], "rows": {label: {"seeds": [[pass@1 per length], ...]}}}"""
    import numpy as np

    lens = summary["lengths"]
    rows = summary["rows"]
    cols = [S.REF, S.OKABE_ITO["sky_blue"], S.OKABE_ITO["blue"], S.OKABE_ITO["vermillion"]]
    hatches = [None, None, None, "////"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.2, layout="constrained")
        ax = fig.subplots()
        w = 0.8 / len(rows)
        for i, (label, r) in enumerate(rows.items()):
            a = np.array(r["seeds"], float)
            x = np.arange(len(lens)) + (i - (len(rows) - 1) / 2) * w
            ax.bar(x, np.nanmean(a, 0), w * 0.92, color=cols[i % 4], hatch=hatches[i % 4], edgecolor="white",
                   linewidth=0.3, label=label)  # fmt: skip
            for srow in a:
                ax.plot(x, srow, linestyle="none", marker="o", markersize=1.6, color=S.INK)
        ax.set_xticks(range(len(lens)), [f"{n}\n(n={k})" for n, k in zip(lens, summary["n"])])
        ax.set_xlabel("lines in the reference proof (n = dev theorems)")
        ax.set_ylabel("pass@1, % of dev theorems")
        ax.set_ylim(0, 100)
        S.light_grid(ax)
        ax.legend(fontsize=S.FS_TICK, frameon=False, ncols=2, loc="lower left")
        S.save(fig, Path(out) / "dev_by_length")
        S.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/scratch/cowifornia/nd-rl/fable_rl/runs")
    ap.add_argument("--final", default="/scratch/cowifornia/nd-rl/fable_rl/final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bases, summary = {}, None
    fn = os.path.join(a.final, "summary.json")
    if os.path.exists(fn):
        summary = json.load(open(fn))
        bases = summary.get("bases", {})
    fig_by_round(a.runs, bases, a.out)
    if summary and "by_length" in summary:
        fig_by_length(summary["by_length"], a.out)


if __name__ == "__main__":
    main()
