"""The one figure of factor_read: Δlog10 per group along the σ route, base to RL end, by dose, seeds visible.

    code/.venv/bin/python -m code.experiments.current.factor_read_20261106.fig --results R.json --out BASE

Top row: C (Amendment 1: name-free), mean per theorem over the 106, adds to the path total. Bottom row: B, sampled from each
boundary, pooled over the 106. One marker per seed; a tick at the mean of the dose's three seeds. Hollow in the bottom row: 0 of
6,784 at base or at the end, so that value is bounded by the floor, not measured. The guard-off pilot is one run: a diamond beside
the 0.1 % seeds, not in their mean. dko_s1, dd01_s1 and dd01_s2 are alarm checkpoints.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D

from code.ndbench.figures import style as S

GROUPS = ("setup", "pick", "reach", "negi", "dn", "finish")
GLAB = ("setup", "pick + ¬¬T", "reach ⊥", "close box", "DN", "finish")
LINES = {
    "0%": ["dko_s0", "dko_s1", "dko_s2"],
    "0.1%": ["dd01_s0", "dd01_s1", "dd01_s2"],
    "1%": ["dd1_s0", "dd1_s1", "dd1_s2"],
}
PILOT = "dd01g0_s2"
OI = S.OKABE_ITO
SEED = [(OI["blue"], "o"), (OI["vermillion"], "s"), (OI["bluish_green"], "^")]
OFF = (-0.2, 0.0, 0.2)
STAGES = ("pick", "reach", "negi", "dn")
SLAB = ("pick + ¬¬T", "reach ⊥", "close box", "DN")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())["lines"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.0)
        axes = fig.subplots(2, 3, sharey="row", sharex="row")
        for ax, (dose, lines), letter in zip(axes[0], LINES.items(), "abc", strict=True):
            ax.axhline(0, color=S.REF, lw=0.6, zorder=1)
            for x, g in enumerate(GROUPS):
                vals = []
                for k, ln in enumerate(lines):
                    v = res[ln]["C"]["delta"][g]
                    vals.append(v)
                    col, mk = SEED[k]
                    ax.plot(x + OFF[k], v, mk, color=col, ms=3.6, mec=S.WHITE, mew=0.3, zorder=3)
                ax.plot([x - 0.3, x + 0.3], [sum(vals) / 3] * 2, color=S.INK, lw=0.9, zorder=2)
                if dose == "0.1%" and PILOT in res:
                    ax.plot(
                        x + 0.38,
                        res[PILOT]["C"]["delta"][g],
                        "D",
                        mfc="none",
                        mec=S.INK,
                        mew=0.6,
                        ms=3.2,
                        zorder=3,
                    )
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels(GLAB, rotation=40, ha="right")
            ax.set_xlim(-0.6, len(GROUPS) - 0.4)
            S.light_grid(ax, "y")
            S.panel_title(ax, letter, f"C, one path: {dose} dose")
        axes[0][0].set_ylabel("Δlog10 P, base to RL end\n(mean per theorem)")
        for ax, (dose, lines), letter in zip(axes[1], LINES.items(), "def", strict=True):
            ax.axhline(0, color=S.REF, lw=0.6, zorder=1)
            for x, st in enumerate(STAGES):
                vals = []
                for k, ln in enumerate(lines):
                    b = res[ln]["B"][st]
                    v = b["delta_pooled"]
                    vals.append(v)
                    col, mk = SEED[k]
                    if b["pooled_at_floor"]:
                        ax.plot(x + OFF[k], v, mk, mfc="none", mec=col, mew=0.7, ms=3.4, zorder=3)
                    else:
                        ax.plot(
                            x + OFF[k], v, mk, color=col, ms=3.6, mec=S.WHITE, mew=0.3, zorder=3
                        )
                ax.plot([x - 0.3, x + 0.3], [sum(vals) / 3] * 2, color=S.INK, lw=0.9, zorder=2)
                if dose == "0.1%" and PILOT in res:
                    ax.plot(
                        x + 0.38,
                        res[PILOT]["B"][st]["delta_pooled"],
                        "D",
                        mfc="none",
                        mec=S.INK,
                        mew=0.6,
                        ms=3.2,
                        zorder=3,
                    )
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(SLAB, rotation=40, ha="right")
            ax.set_xlim(-0.6, len(STAGES) - 0.4)
            S.light_grid(ax, "y")
            S.panel_title(ax, letter, f"B, all paths: {dose} dose")
        axes[1][0].set_ylabel("Δlog10 rate, base to RL end\n(pooled over the 106)")
        hs = [
            Line2D(
                [],
                [],
                ls="none",
                marker=m,
                color=c,
                ms=3.6,
                mec=S.WHITE,
                mew=0.3,
                label=f"seed {i}",
            )
            for i, (c, m) in enumerate(SEED)
        ]
        hs.append(Line2D([], [], color=S.INK, lw=0.9, label="mean of 3 seeds"))
        if PILOT in res:
            hs.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="D",
                    mfc="none",
                    mec=S.INK,
                    mew=0.6,
                    ms=3.2,
                    label="guard-off pilot (one run)",
                )
            )
        hs.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                mfc="none",
                mec=S.REF,
                mew=0.7,
                ms=3.4,
                label="at the floor",
            )
        )
        fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), frameon=False)
        S.save(fig, Path(a.out))
        S.close(fig)


if __name__ == "__main__":
    main()
