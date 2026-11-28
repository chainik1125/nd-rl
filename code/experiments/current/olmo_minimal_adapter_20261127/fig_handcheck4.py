"""Round-4 reader check, one figure: three pass rates per model on the 48 reach prompts.

Grouped bars per model: strict checker pass, reader-accepted pass (not screened, read genuine) and
task-done pass (not screened, read "task done"). Each is a share of all draws, with its 95 %
interval (prompts -> draws bootstrap) as a whisker.
Input: handcheck4.py's round4_summary.json. Output: <out>.pdf, .svg and .png.

Usage: python fig_handcheck4.py [--summary .../round4/round4_summary.json] [--out .../handcheck4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
from ndbench.figures import style as S  # noqa: E402

V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
FIGS = (
    Path(__file__).resolve().parents[4]
    / "experiment-summaries/2026-11-27-olmo-minimal-adapter/figures"
)
# cell -> plain name, in plotting order (the yardstick first)
MODELS = [
    ("s2300", "final RL\nmodel"),
    ("A", "25 Nov\nadapter"),
    ("Aprime", "our retrain\nof it"),
    ("B", "25 Nov data,\nour seed"),
    ("C", "our data,\n25 Nov seed"),
    ("D", "our data,\nour seed"),
    ("expo", "extrapolated\nmodel"),
    ("k16", "top-16-\ndirections\nadapter"),
]
MEASURES = [
    ("strict_rate", "strict checker pass", "#b4b4b4"),
    ("genuine_rate", "reader-accepted pass", S.OKABE_ITO["blue"]),
    ("task_done_rate", "task-done pass", S.OKABE_ITO["orange"]),
]


def bars(ax, cells: dict) -> None:
    w = 0.26
    for j, (q, label, color) in enumerate(MEASURES):
        x = [i + (j - 1) * w for i in range(len(MODELS))]
        v = [cells[c][q] for c, _ in MODELS]
        y = [r["value"] for r in v]
        err = [
            [max(0.0, r["value"] - r["lo"]) for r in v],
            [max(0.0, r["hi"] - r["value"]) for r in v],
        ]
        ax.bar(x, y, width=w, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=2)
        ax.errorbar(
            x, y, yerr=err, fmt="none", ecolor=S.INK2, elinewidth=0.7, capsize=1.6, zorder=3
        )
    ax.set_xticks(range(len(MODELS)), [name for _, name in MODELS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("share of all draws")
    S.pct_axis(ax, "y")
    S.light_grid(ax, "y")
    ax.legend(fontsize=S.FS_TICK, frameon=False, loc="upper center", ncols=3, borderaxespad=0.1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summary", type=Path, default=V2 / "handcheck/round4/round4_summary.json")
    ap.add_argument("--out", type=Path, default=FIGS / "handcheck4")
    a = ap.parse_args()
    cells = json.loads(a.summary.read_text())["cells"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.45)
        bars(fig.add_subplot(), cells)
        paths = S.save(fig, a.out)
        S.close(fig)
    print("wrote", *paths)


if __name__ == "__main__":
    main()
