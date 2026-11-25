# ruff: noqa: E501, SIM115
"""Stage 2 arms over RL step -> /scratch/cowifornia/nd-rl/olmo_elicit/figures/arms.{pdf,svg,png} (from arms.csv).

(a) The 41 transplant prompts: strict checker pass rate of answers written after the final model's own thinking,
    without (T) and with (A1) the "reply with only your final answer" instruction. dpo and s2300: Dan's stored cells.
(b) The 48 reach prompts: strict pass rate with the 60-word hint (H) against Dan's plain-prompt arm B (all stored
    draws: 32 per prompt; 16 at dpo).
(c) The same 48 prompts, plain prompt, through each checkpoint's stage-1 adapter (L, 8 draws; theta_64 of the
    shortest-code-length order-0 EDL row) against Dan's arm B cut to its first 8 draws (B8, like for like) and in full.
Bands: prompt-bootstrap 95 %. Checker passes only; no genuine / gamed reading in this pass. Filled markers = our draws,
hollow = Dan's stored cells; step 200 lies on an abandoned branch of the RL run (dotted line).
Usage: python fig_arms.py [--arms .../arms.csv] [--out .../figures/arms]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
from ndbench.figures import style as S  # noqa: E402

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")


def draw(ax, rows, color, marker, label, ls="-"):
    rows = sorted(rows, key=lambda r: int(r["step"]))
    if not rows:
        return
    x = [int(r["step"]) for r in rows]
    y = [float(r["strict_pct"]) / 100 for r in rows]
    lo = [float(r["lo95"]) / 100 for r in rows]
    hi = [float(r["hi95"]) / 100 for r in rows]
    ax.plot(x, y, color=color, ls=ls, lw=1.1, zorder=2)
    ax.plot([], [], color=color, ls=ls, lw=1.1, marker=marker, label=label)  # legend proxy
    ax.fill_between(x, lo, hi, color=color, alpha=S.BAND_ALPHA, lw=0, zorder=1)
    for xi, yi, r in zip(x, y, rows, strict=True):
        ours = r["source"] == "ours"
        ax.plot(
            [xi],
            [yi],
            marker=marker,
            color=color,
            mfc=color if ours else "white",
            mec=color,
            ls="none",
            zorder=3,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=str(OE / "arms.csv"))
    ap.add_argument("--out", default=str(OE / "figures" / "arms"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.arms)))  # noqa: SIM115
    C = S.OKABE_ITO

    def arm(k):
        return [r for r in rows if r["arm"] == k and r["strict_pct"] not in ("", "None")]

    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.2)
        ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)
        draw(ax1, arm("T"), C["blue"], "o", "transplant, no instruction (T)")
        draw(ax1, arm("A1"), C["vermillion"], "s", "+ answer-only instruction (A1)", ls="--")
        S.panel_title(ax1, "a", "41 transplant prompts")
        draw(ax2, arm("H"), C["bluish_green"], "^", "60-word hint (H)")
        draw(ax2, arm("B"), C["reddish_purple"], "D", "plain prompt (Dan's arm B)", ls="--")
        S.panel_title(ax2, "b", "48 reach prompts, hint")
        draw(ax3, arm("L"), C["orange"], "v", "stage-1 adapter (L), 8 draws")
        draw(ax3, arm("B8"), C["reddish_purple"], "D", "plain, Dan's first 8", ls="--")
        draw(ax3, arm("B"), C["sky_blue"], "o", "plain, Dan's 32", ls=":")
        S.panel_title(ax3, "c", "48 reach prompts, adapter")
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(-80, 2400)
            ax.set_ylim(0, 1.0)
            ax.set_xticks([0, 1000, 2000])
            ax.set_xlabel("RL step")
            S.pct_axis(ax)
            S.light_grid(ax)
            ax.axvline(200, color=S.REF, lw=0.5, ls=":", zorder=0)
            ax.legend(loc="lower right" if ax is ax1 else "upper left", fontsize=S.FS_SMALL)
        ax1.set_ylabel("strict checker passes, % of draws")
        S.save(fig, Path(a.out))
        S.close(fig)
    print(f"wrote {a.out}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
