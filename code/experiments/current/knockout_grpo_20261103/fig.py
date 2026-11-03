"""Figures for knockout_grpo (Dan's headline layout, our R axis), from numbers.json.

    code/.venv/bin/python -m code.experiments.current.knockout_grpo_20261103.fig [--figdir DIR]

(a) `rl_needed_grpo`: RL compute / pretraining compute needed to reach the bar, against sigma's
    share of the pretraining proofs; one marker per seed. Filled: reached; hollow: at the bar
    before RL; up-triangle on a thin line: not reached by R_max. Grey: Dan's EI ladders, recomputed
    with our R rule (dan_axis.py), at his doses.
(b) `e32_vs_r`: E32 on the strict 106 against R, per arm and seed, with the bar.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from code.ndbench.figures import style as S

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARMS = ("full", "d1", "d01", "ko")
NAME = {"full": "S1g3 (natural)", "d1": "1 %", "d01": "0.1 %", "ko": "0 % (removed)"}
COLOR = {"full": S.OKABE_ITO["blue"], "d1": S.OKABE_ITO["bluish_green"],
         "d01": S.OKABE_ITO["orange"], "ko": S.OKABE_ITO["vermillion"]}  # fmt: skip
OURS_X = {"full": 2.48, "d1": 1.0, "d01": 0.1, "ko": None}
DAN_X = {"full": 10.9, "dose1": 1.0, "dose01": 0.1, "knockout": None}
ZERO_X = 0.025
Y_LO = 0.002


def headline(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 3.3, layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.016, 22)
    ax.set_ylim(Y_LO, 30)
    xt = [ZERO_X, 0.1, 1.0, 2.48, 10.9]
    ax.set_xticks(xt, ["0 %\n(removed)", "0.1 %", "1 %", "2.48 %\n(S1g3)", "10.9 %\n(K12, Dan)"])
    ax.minorticks_off()
    yt = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    ax.set_yticks(yt, [f"{v:g}×" for v in yt])
    S.light_grid(ax, "y")
    ax.text(
        0.05, Y_LO * 0.62, "//", color=S.REF, ha="center", va="center", fontsize=7, clip_on=False
    )
    # Dan's EI ladders on our axis (grey, offset left)
    for arm, x0 in DAN_X.items():
        lads = [o for o in N.get("dan_axis", []) if o["arm"] == arm]
        xx = (x0 or ZERO_X) / 1.45
        for k, o in enumerate(lads):
            f = 0.85 + 0.3 * k / max(1, len(lads) - 1)
            if o["cross"] is not None:
                first = o["cross"]["round"] == 1
                y = max(o["cross"]["x_ours"], Y_LO * 1.3)
                ax.plot(
                    [xx * f],
                    [y],
                    "o",
                    ms=3.2,
                    color=S.REF,
                    mfc="white" if first else S.REF,
                    mec=S.REF,
                    mew=0.8,
                    zorder=3,
                )
            else:
                y = o["last"]["x_ours"]
                ax.plot([xx * f] * 2, [Y_LO, y], color=S.REF, lw=0.6, alpha=0.4)
                ax.plot([xx * f], [y], "^", ms=3.4, color=S.REF, mec="white", mew=0.5, zorder=3)
    # ours
    for arm in ARMS:
        x0 = OURS_X[arm] or ZERO_X
        tags = [f"{arm}_s{s}" for s in range(3) if f"{arm}_s{s}" in N["runs"]]
        hits, misses, firsts = [], [], 0
        for k, tag in enumerate(tags):
            f = 0.9 + 0.2 * k / max(1, len(tags) - 1)
            c = N["runs"][tag]["cross"]
            if c["status"] == "at_bar_before_rl":
                firsts += 1
                ax.plot(
                    [x0 * f],
                    [Y_LO * 1.6],
                    "o",
                    ms=5,
                    mfc="white",
                    mec=COLOR[arm],
                    mew=1.1,
                    zorder=4,
                )
            elif c["status"] == "reached":
                hits.append(c["R"])
                ax.plot(
                    [x0 * f], [c["R"]], "o", ms=5, color=COLOR[arm], mec="white", mew=0.6, zorder=4
                )
            elif c["status"] == "not_reached":
                misses.append(c)
                ax.plot([x0 * f] * 2, [Y_LO, c["R_max"]], color=COLOR[arm], lw=0.7, alpha=0.45)
                ax.plot(
                    [x0 * f],
                    [c["R_max"]],
                    "^",
                    ms=5.5,
                    color=COLOR[arm],
                    mec="white",
                    mew=0.6,
                    zorder=4,
                )
        if hits and not misses and not firsts and len(hits) > 1:
            m = st.median(hits)
            ax.plot([x0 * 0.8, x0 * 1.25], [m, m], color=S.INK2, lw=0.8, zorder=3)
        txt = []
        if firsts:
            txt.append(f"{firsts} of {len(tags)} at the bar before RL")
        if hits:
            rng = f"{min(hits):.3g}–{max(hits):.3g}×" if len(hits) > 1 else f"{hits[0]:.3g}×"
            txt.append(f"{rng}" + (f" (median {st.median(hits):.3g}×)" if len(hits) > 2 else ""))
        if misses:
            txt.append(
                f"not reached by {min(m['R_max'] for m in misses):.2g}–"
                f"{max(m['R_max'] for m in misses):.2g}× ({len(misses)} of {len(tags)})"
            )
        ys = hits or [m["R_max"] for m in misses] or [Y_LO * 1.6]
        ax.text(
            x0 * 1.35,
            st.median(ys),
            "\n".join(txt),
            va="center",
            fontsize=S.FS_SMALL,
            color=COLOR[arm],
        )
    ax.set_xlabel("σ's share of the pretraining proofs (log scale)")
    ax.set_ylabel(f"RL ÷ pretraining compute needed\nto reach the bar ({N['bar_e32']:.1f} of 106)")
    S.save(fig, figdir / "rl_needed_grpo")
    S.close(fig)


def curves(N: dict, figdir: Path) -> None:
    fig = S.new_figure(S.FULL_W, 2.2, layout="constrained")
    axs = fig.subplots(1, 4, sharey=True)
    for ax, arm in zip(axs, ARMS, strict=True):
        ax.set_xscale("log")
        for s in range(3):
            tag = f"{arm}_s{s}"
            x = N["runs"].get(tag)
            if not x or "curve" not in x:
                continue
            R = [r["R"] for r in x["curve"]]
            y = [r["e32"] for r in x["curve"]]
            ls = ["-", "--", ":"][s]
            ax.plot(R, y, ls, color=COLOR[arm], lw=0.9, label=f"s{s}")
            ax.plot([R[0] / 2], [x["base_e32"]], "o", ms=2.5, color=COLOR[arm])
        ax.axhline(N["bar_e32"], color=S.REF, lw=0.6, ls="--")
        S.panel_title(ax, None, NAME[arm])
        ax.set_xlabel("R")
        S.light_grid(ax, "both")
    axs[0].set_ylabel("E32 on the strict 106")
    axs[0].legend(fontsize=S.FS_SMALL, frameon=False)
    S.save(fig, figdir / "e32_vs_r")
    S.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=str(HERE / "figs"))
    a = ap.parse_args()
    N = json.loads((HERE / "numbers.json").read_text())
    with S.paper_style():
        headline(N, Path(a.figdir))
        curves(N, Path(a.figdir))


if __name__ == "__main__":
    main()
