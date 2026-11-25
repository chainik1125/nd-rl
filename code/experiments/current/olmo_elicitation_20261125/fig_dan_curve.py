# ruff: noqa: E501, SIM115
"""Stage 0b: Dan's olmo-ifcurve curve, redrawn -> /scratch/cowifornia/nd-rl/olmo_elicit/figures/dan_curve.{pdf,svg,png}.

(a) Arm A, the 218 IFBench prompts: strict pass@1 and pass@8 (% of 218) with his prompt-bootstrap 95 % bands
    (olmo_ifcurve/passk.csv, reproduced exactly in stage0/repro.md). pass@8 needs 8 draws: absent at steps 50 and 200
    (4 draws per prompt there).
(b) The 48 reach prompts: share with >= 1 strict checker pass in the 32 regular tries per checkpoint (his arm B), and
    his hand-gate lower bound on prompts with >= 1 genuine success among those tries (results.csv; Claude-reader labels
    taken as given). No arm-B cell at dpo; by the set's definition dpo has 0 genuine in >= 256 draws (plotted at 0).
Step 200 (hollow markers) lies on an abandoned branch of the RL run (olmo-weights review), off the released line.
Usage: python fig_dan_curve.py [--out /scratch/cowifornia/nd-rl/olmo_elicit/figures/dan_curve]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dan_curve as dc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
from ndbench.figures import style as S  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/figures/dan_curve")


def series(ax, xs, ys, color, marker, label, ls="-", band=None):
    on = [i for i, x in enumerate(xs) if x != 200]
    ax.plot(
        [xs[i] for i in on],
        [ys[i] for i in on],
        color=color,
        marker=marker,
        ls=ls,
        label=label,
        zorder=3,
    )
    if band is not None:
        lo, hi = band
        ax.fill_between(
            [xs[i] for i in on],
            [lo[i] for i in on],
            [hi[i] for i in on],
            color=color,
            alpha=S.BAND_ALPHA,
            lw=0,
            zorder=1,
        )
    off = [i for i, x in enumerate(xs) if x == 200]
    for i in off:
        if band is not None:
            ax.errorbar(
                [xs[i]],
                [ys[i]],
                yerr=[[ys[i] - band[0][i]], [band[1][i] - ys[i]]],
                color=color,
                lw=0.6,
                capsize=1.5,
                zorder=2,
            )
        ax.plot(
            [xs[i]],
            [ys[i]],
            marker=marker,
            mfc="white",
            mec=color,
            color=color,
            ls="none",
            zorder=4,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    pk = dc.dan_passk()
    rb = dc.dan_reach()
    C = S.OKABE_ITO
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.3)
        ax1, ax2 = fig.subplots(1, 2)
        for k, col, mk in ((1, C["blue"], "o"), (8, C["vermillion"], "s")):
            cs = [c for c in dc.CKPTS if pk[c][k][0] == pk[c][k][0]]
            xs = [dc.STEP[c] for c in cs]
            series(
                ax1,
                xs,
                [pk[c][k][0] for c in cs],
                col,
                mk,
                f"pass@{k}",
                band=([pk[c][k][1] for c in cs], [pk[c][k][2] for c in cs]),
            )
        S.panel_title(ax1, "a", "218 IFBench prompts (arm A), strict")
        cs = dc.CKPTS[1:]
        xs = [dc.STEP[c] for c in cs]
        series(
            ax2,
            xs,
            [rb[c]["solved_strict"] / 48 for c in cs],
            C["bluish_green"],
            "^",
            "≥ 1 strict pass in 32 tries",
        )
        series(
            ax2,
            [0, *xs],
            [0.0] + [rb[c]["genuine_established"] / 48 for c in cs],
            C["reddish_purple"],
            "D",
            "≥ 1 hand-labelled genuine (lower bound)",
            ls="--",
        )
        S.panel_title(ax2, "b", "48 reach prompts (arm B), share solved")
        for ax in (ax1, ax2):
            ax.set_xlim(-80, 2400)
            ax.set_ylim(0, 1.0)
            ax.set_xticks([0, 500, 1000, 1500, 2000])
            ax.set_xlabel("RL step")
            S.pct_axis(ax)
            S.light_grid(ax)
            ax.axvline(200, color=S.REF, lw=0.5, ls=":", zorder=0)
            ax.text(
                215,
                0.97,
                "step 200: off the\nreleased line",
                fontsize=S.FS_SMALL,
                color=S.REF,
                va="top",
            )
            ax.legend(loc="lower right")
        ax1.set_ylabel("% of prompts")
        S.save(fig, Path(a.out))
        S.close(fig)
    print(f"wrote {a.out}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
