#!/usr/bin/env python3
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""The summary's figure: modus ponens appearing in one of the three runs pretrained without it.

  python figure.py        # writes experiment-summaries/2026-11-18-rule-knockouts/figures/mp_ignition.png

Left: distinct Lean-accepted proofs that use modus ponens so far, against RL minutes. Right: RL targets solved so
far. Blue: the three expert-iteration runs with modus ponens removed from pretraining. Orange: the two controls.
Reads each run's rounds.jsonl from artifacts/rule_knockout_20261118/.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ART = REPO / "artifacts" / "rule_knockout_20261118"
OUT = REPO / "experiment-summaries" / "2026-11-18-rule-knockouts" / "figures" / "mp_ignition.png"
# categorical slots 1 and 2 of the dataviz reference palette (validated together on the light surface)
BLUE, ORANGE, SURFACE, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
)
RUNS = [("mp-ei_s0", BLUE, "seed 0"), ("mp-ei_s1", BLUE, "seed 1"), ("mp-ei_s2", BLUE, "seed 2"),
        ("full-ei_s0", ORANGE, "control 0"), ("full-ei_s1", ORANGE, "control 1")]  # fmt: skip


def series(name):
    lines = (ART / f"rule-knockout-20261118_{name}" / "rounds.jsonl").read_text().splitlines()
    calls = [json.loads(line) for line in lines if line.strip()]
    minutes = [c["rl_secs_used"] / 60 for c in calls]
    return (
        minutes,
        [c["census"]["proofs_using"].get("IMPE", 0) for c in calls],
        [c["targets_solved"] for c in calls],
    )


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans"})  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    titles = ("Distinct accepted proofs that use modus ponens", "RL targets solved (of 18,439)")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("RL minutes (GPU clock)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for name, color, label in RUNS:
        minutes, proofs, solved = series(name)
        for ax, ys in zip(axes, (proofs, solved), strict=True):
            ax.plot(minutes, ys, color=color, linewidth=2, solid_capstyle="round")
            ax.annotate(label, (minutes[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=9, color=MUTED)  # fmt: skip
    # seeds 0 and 2 never write modus ponens: their lines sit on zero in the left panel, so label them once
    axes[0].texts[0].set_text("seeds 0 and 2")
    axes[0].texts[2].set_text("")
    # in the right panel seeds 0 and 2 end 7 targets apart: one label for both
    axes[1].texts[0].set_text("seeds 0 and 2")
    axes[1].texts[2].set_text("")
    axes[1].texts[3].set_text("controls")
    axes[1].texts[4].set_text("")
    for ax in axes:
        ax.set_xlim(0, 215)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2, label="modus ponens removed from pretraining (3 seeds)"),
               plt.Line2D([], [], color=ORANGE, linewidth=2, label="control: nothing removed (2 seeds)")]  # fmt: skip
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(OUT.parent, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
