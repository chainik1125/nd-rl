#!/usr/bin/env python3
"""Figure of the weight-interpolation run: strict-checker pass rate against alpha, one panel.

    python3 fig_interp.py [--results <data>/analysis/results.json] [--out <base path>]

Reads analyze.py's results.json and writes <out>.pdf, .svg and .png through the single figure style
(`code/ndbench/figures/style.py`: 5.5 in wide, serif 7 to 8 pt, colour-blind-safe palette).
Shown: the cells of this run with their 95 % prompt-bootstrap intervals (merged checkpoints as
circles, the same-engine alpha 1.0 cell on the stored s2300 weights as a square), the stored pre-RL
point at alpha 0 (hollow: 0 of 768 stored draws, not sampled in this run), and the straight line
between the two ends for reference. When results.json carries the smooth reference of PREREG
Amendment 1 (every prompt's log-odds of passing linear in alpha, from stored rates only), the band
between its two versions is drawn in grey. A cell with missing draws also gets a thin bracket from
its all-fail to its all-pass rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[3]  # .../code
sys.path.insert(0, str(CODE))
from ndbench.figures import style as S  # noqa: E402

ORDER = ("a025", "a050", "a075", "a100")


def draw(res: dict, base: Path) -> list[Path]:
    cells = {
        t: res["cells"][t]
        for t in ORDER
        if t in res["cells"] and res["cells"][t]["pass1"] is not None
    }
    merged = S.ModelStyle("merged checkpoint, this run", S.OKABE_ITO["blue"], "o")
    top = S.ModelStyle("stored final RL weights, same engine", S.OKABE_ITO["vermillion"], "s")
    pre = S.ModelStyle(
        "pre-RL model, stored draws (0 of 768)", S.OKABE_ITO["blue"], "o", filled=False
    )
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.6)
        ax = fig.add_subplot(111)
        xs = [0.0] + [cells[t]["alpha"] for t in cells]
        ys = [0.0] + [cells[t]["pass1"] for t in cells]
        ax.plot(xs, ys, color=S.INK2, linewidth=0.9, zorder=2)
        ref = list((res.get("reference") or {}).values())
        if len(ref) == 2:
            ax.fill_between(
                ref[0]["curve"]["alpha"],
                ref[1]["curve"]["rate"],
                ref[0]["curve"]["rate"],
                color=S.REF,
                alpha=0.35,
                linewidth=0,
                zorder=0,
                label="smooth reference from stored rates",
            )
        if "a100" in cells:
            ax.plot(
                [0, 1],
                [0, cells["a100"]["pass1"]],
                color=S.REF,
                linewidth=0.7,
                linestyle=(0, (1, 1.6)),
                zorder=1,
                label="straight line between the ends",
            )
        for t, c in cells.items():
            st = top if t == "a100" else merged
            lo, hi = c["pass1_ci"]
            if lo is not None:
                ax.errorbar(
                    [c["alpha"]],
                    [c["pass1"]],
                    yerr=[[c["pass1"] - lo], [hi - c["pass1"]]],
                    color=S.INK2,
                    linewidth=0.8,
                    capsize=1.8,
                    zorder=3,
                    linestyle="none",
                )
            ax.plot([c["alpha"]], [c["pass1"]], linestyle="none", zorder=4, **st.marker_kw(4.6))
            if c["missing"]:
                x = c["alpha"] + 0.018
                ax.plot(
                    [x, x],
                    [c["pass1_all_fail"], c["pass1_all_pass"]],
                    color=S.OKABE_ITO["orange"],
                    linewidth=1.0,
                    marker="_",
                    markersize=4,
                    zorder=3,
                )
            ax.annotate(
                f"{100 * c['pass1']:.1f}%",
                (c["alpha"], c["pass1"]),
                textcoords="offset points",
                xytext=(-5, 5),
                ha="right",
                fontsize=S.FS_TICK,
                color=S.INK,
            )
        ax.plot([0.0], [0.0], linestyle="none", zorder=4, clip_on=False, **pre.marker_kw(4.6))
        ax.set_xlim(-0.04, 1.06)
        ax.set_ylim(-0.02, max(1.0, max(ys) + 0.05) if ys else 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0\n(pre-RL)", "0.25", "0.5", "0.75", "1\n(final RL)"])
        ax.set_xlabel(
            r"$\alpha$ in  pre-RL weights + $\alpha$ $\times$ (final RL weights $-$ pre-RL weights)"
        )
        ax.set_ylabel("strict-checker pass@1,\n48 reach prompts")
        S.pct_axis(ax, "y")
        S.light_grid(ax, "y")
        handles = S.legend_handles([merged, top, pre], kind="marker")
        extra, _ = ax.get_legend_handles_labels()
        if any(c["missing"] for c in cells.values()):
            from matplotlib.lines import Line2D

            extra.append(
                Line2D(
                    [],
                    [],
                    color=S.OKABE_ITO["orange"],
                    linewidth=1.0,
                    label="all-fail to all-pass (cell cut)",
                )
            )
        ax.legend(handles=handles + extra, loc="upper left")
        paths = S.save(fig, base)
        S.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = "/scratch/cowifornia/nd-rl/olmo_elicit/interp/analysis"
    ap.add_argument("--results", default=f"{d}/results.json")
    ap.add_argument("--out", default=f"{d}/fig_interp")
    a = ap.parse_args(argv)
    for p in draw(json.loads(Path(a.results).read_text()), Path(a.out)):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
