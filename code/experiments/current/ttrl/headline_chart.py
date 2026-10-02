#!/usr/bin/env python3
"""The post's headline TTRL chart: dev solved against test-time compute for the 5x run, mean of
three seeds with the min-max band. Compute is in units of x = the seed's pretrain + RL time.

  uv run --with matplotlib --with numpy python headline_chart.py artifacts/ttrl <png>
"""

import json
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

HF = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs"
SEEDS = (0, 1, 2)
ARMS = {
    "control": "sample more",
    "selftrain": "targets-only EI",
    "ttrl": "TTRL (targets + variants)",
}
COLOR = {"control": "#8a8985", "selftrain": "#eb6834", "ttrl": "#2a78d6"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb"


def x_and_baseline(s):
    with urllib.request.urlopen(f"{HF}/combo-c005_s{s}/summary.json") as f:
        m = json.load(f)
    return m["pretrain"]["secs"] + m["rl"]["secs"], m["eval"]["solved"]


def main():
    root, out = sys.argv[1], sys.argv[2]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    seeds = {s: x_and_baseline(s) for s in SEEDS}
    grid = np.linspace(0, 5, 251)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, label in ARMS.items():
        ys = []
        for s, (x, base) in seeds.items():
            with open(f"{root}/devall_s{s}_{arm}/{arm}.json") as f:
                rounds = json.load(f)["rounds"]
            t = np.array([r["loop_secs"] / x for r in rounds])
            v = np.array([r["targets_solved"] for r in rounds])
            # solved by the last round that finished within each budget
            ys.append([base + (v[t <= g][-1] if (t <= g).any() else 0) for g in grid])
        ys = np.array(ys)
        mean = ys.mean(0)
        ax.fill_between(grid, ys.min(0), ys.max(0), color=COLOR[arm], alpha=0.14, lw=0)
        ax.plot(grid, mean, color=COLOR[arm], lw=2, label=label)
        ax.annotate(
            f"{mean[-1]:,.0f}",
            (5, mean[-1]),
            xytext=(5, -3),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
    base_mean = np.mean([b for _, b in seeds.values()])
    ax.axhline(1108, color=MUTED, lw=1, ls=":")
    ax.text(0.05, 1110, "all 1,108", color=MUTED, fontsize=8, va="bottom")
    ax.axhline(base_mean, color=MUTED, lw=1, ls="--")
    ax.text(
        0.05,
        base_mean + 2,
        f"before test time: {base_mean:,.0f}",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    ax.set_xlim(0, 5.45)
    ax.set_ylim(890, 1125)
    ax.set_xlabel("test-time compute, in units of the pipeline's own pretrain + RL time")
    ax.set_ylabel("dev theorems solved (of 1,108)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.set_title(
        "Training on the failed theorems beats sampling them; variants arrive late",
        loc="left",
        color=INK,
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "Lines: mean of 3 seeds. Bands: min to max across seeds.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
