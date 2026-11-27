# ruff: noqa: I001  (import order differs between the repo-root and folder-local ruff configs)
# ruff: noqa: E501
"""Stage-1 figure -> /scratch/cowifornia/nd-rl/olmo_elicit/v2/figures/stage1.{pdf,svg,png}.

(a) G (share of the dpo -> s2300 gap closed on the 48 reach prompts, checker passes) per sampled cell, with its 95 %
    two-level paired bootstrap interval (analyze_cells.py), grouped by training recipe: the prompt control (no adapter),
    the dev-loss rule's recipe (lr 3e-5, 128 steps), the one-pass recipe (lr 3e-4, 64 steps), and the 25 Nov adapter
    (same one-pass recipe on the 25 Nov pool, sampled 25 Nov through Dan's single-adapter path). All adapters A16
    (r16, all projections, P = 134,217,728), n = 64.
(b) One-pass and extended capacity ratio C of the final-source rungs against P (capacity.py, difficulty-matched EDL):
    C = EDL(P) / EDL_64(A16), EDL at t = 64 (filled) and at the dev-selected step (hollow). Line at 0.95.
Usage: python fig_stage1.py [--summary .../cells_summary.json] [--capacity .../capacity.json] [--out .../figures/stage1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
import analyze_cells as AC  # noqa: E402

from ndbench.figures import style as S  # noqa: E402

V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
OLD_L = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs/dpo/L")

ROWS = [  # (group, label, cell id, source)
    ("no adapter", "prompt control (answer-only instruction)", "plain:answeronly", "plain"),
    (
        "dev-loss rule: lr 3e-5, 128 steps",
        "final (s2300's passes)",
        "dpo__final__A16__n64",
        "final",
    ),
    ("dev-loss rule: lr 3e-5, 128 steps", "own (dpo's passes)", "dpo__own__A16__n64", "own"),
    (
        "dev-loss rule: lr 3e-5, 128 steps",
        "unsel (dpo's draws, checker ignored)",
        "dpo__unsel__A16__n64",
        "unsel",
    ),
    ("one pass: lr 3e-4, 64 steps", "final (s2300's passes)", "dpo__final__A16__n64__r1p", "final"),
    ("one pass: lr 3e-4, 64 steps", "own (dpo's passes)", "dpo__own__A16__n64__r1p", "own"),
    (
        "one pass: lr 3e-4, 64 steps",
        "unsel (dpo's draws, checker ignored)",
        "dpo__unsel__A16__n64__r1p",
        "unsel",
    ),
    (
        "25 Nov adapter (one pass, 25 Nov pool)",
        "sampled 25 Nov, Dan's single-adapter path",
        "old",
        "final",
    ),
    (
        "25 Nov adapter (one pass, 25 Nov pool)",
        "sampled today, this path",
        "dpo__final__OLD_A16__n64",
        "final",
    ),
]
SRC_COLOR = {"plain": "orange", "final": "blue", "own": "vermillion", "unsel": "bluish_green"}
CAP_LABEL = {
    "dpo__final__A16__n64": "A16",
    "dpo__final__O1__n64": "O1",
    "dpo__final__M16k__n64": "M16k",
    "dpo__final__M4k__n64": "M4k",
}


def old_cell():
    reach = AC.dc.reach()
    cells = AC.load_cell_draws(OLD_L / "samples.jsonl.gz", OLD_L / "graded.jsonl", reach)
    final, plain = AC.references("dpo", reach)
    return AC.analyze_cell(cells, final, plain, 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(V2 / "cells_summary.json"))
    ap.add_argument("--capacity", default=str(V2 / "capacity.json"))
    ap.add_argument("--out", default=str(V2 / "figures" / "stage1"))
    a = ap.parse_args()
    d = json.loads(Path(a.summary).read_text())
    summ = {r["cell"]: r for r in (d if isinstance(d, list) else d["cells"])}
    summ["old"] = old_cell()
    cap = json.loads(Path(a.capacity).read_text())
    C = S.OKABE_ITO
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.9)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.08)
        ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        y, yt, yl, groups = 0.0, [], [], []
        last = None
        for grp, lab, cid, src in ROWS:
            if grp != last:
                if last is not None:
                    y -= 0.5
                groups.append((y, grp))
                y -= 1.0
                last = grp
            r = summ.get(cid)
            if r is None:
                y -= 1.0
                continue
            g, lo, hi = r["g"], r["g_lo"], r["g_hi"]
            col = C[SRC_COLOR[src]]
            ax.errorbar(
                g,
                y,
                xerr=[[g - lo], [hi - g]],
                fmt="o",
                ms=3.5,
                color=col,
                mfc=col,
                lw=1.0,
                capsize=2,
                zorder=3,
            )
            ax.text(
                1.27,
                y,
                f"{r['strict_pct']:.1f} %",
                va="center",
                ha="right",
                fontsize=S.FS_TICK,
                color=S.INK,
            )
            yt.append(y)
            yl.append(lab)
            y -= 1.0
        for gy, grp in groups:
            ax.text(
                -0.04,
                gy - 0.25,
                grp,
                va="center",
                ha="left",
                fontsize=S.FS_TICK,
                style="italic",
                color=S.REF,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                zorder=4,
            )
        ax.axvline(0.0, color=S.INK, lw=0.8, ls="--", zorder=1)
        ax.axvline(1.0, color=S.REF, lw=0.8, zorder=1)
        ax.axvline(0.8, color=C["black"], lw=0.7, ls=":", zorder=1)
        ax.set_yticks(yt, yl)
        ax.set_xlim(-0.05, 1.29)
        ax.set_ylim(y + 0.3, 0.3)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("G, share of the dpo → final gap closed")
        ax.text(1.27, 0.0, "pass rate", va="center", ha="right", fontsize=S.FS_TICK, color=S.REF)
        S.panel_title(ax, "a", "48 reach prompts × 8 draws; A16, n = 64")
        S.light_grid(ax, "x")
        pts = [r for r in cap if r["cell"] in CAP_LABEL]
        for r in pts:
            bx.scatter(r["P"], r["C_onepass_corr"], s=16, color=C["blue"], zorder=3)
            if r["C_ext_corr"] is not None and r["sel_step"] and r["sel_step"] != 64:
                bx.scatter(
                    r["P"], r["C_ext_corr"], s=16, facecolor="none", edgecolor=C["blue"], zorder=3
                )
            off = {"M4k": (-6, 5), "M16k": (5, -3)}.get(CAP_LABEL[r["cell"]], (4, -9))
            bx.annotate(
                CAP_LABEL[r["cell"]],
                (r["P"], r["C_onepass_corr"]),
                textcoords="offset points",
                xytext=off,
                fontsize=S.FS_TICK,
            )
        bx.axhline(0.95, color=C["black"], lw=0.7, ls=":")
        bx.axhline(1.0, color=S.REF, lw=0.6)
        bx.set_xscale("log")
        bx.set_xlabel("trained weights P")
        bx.set_ylabel("capacity ratio C")
        bx.yaxis.set_label_position("right")
        bx.yaxis.tick_right()
        S.panel_title(bx, "b", "final source, n = 64")
        S.light_grid(bx, "y")
        bx.scatter([], [], s=16, color=C["blue"], label="one pass (t = 64)")
        bx.scatter([], [], s=16, facecolor="none", edgecolor=C["blue"], label="dev-selected step")
        bx.legend(fontsize=S.FS_TICK, frameon=False, loc="upper left")
        paths = S.save(fig, Path(a.out))
        S.close(fig)
    print("wrote", *paths)


if __name__ == "__main__":
    main()
