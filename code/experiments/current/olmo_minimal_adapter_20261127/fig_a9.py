"""Amendment 9, one figure: reader-accepted pass rate against adapter size and against the number of
training examples.

Two panels sharing the y axis (reader-accepted passes as a share of all draws on the 48 reach
prompts, with the 95 % interval of the prompts -> draws bootstrap as a whisker):
  (a) the two adapter sizes, both trained on 64 examples: rank 1 on o_proj (655,360 trained weights)
      and the full-size reference (134,217,728);
  (b) the full-size adapter trained on 4, 16 and 64 examples.
The 64-example full-size adapter is the reference in both panels (square marker). Marked in both:
the final RL model's reader-accepted rate (line and interval band; round 5's labels if its items
were read again, else round 4's) and `dpo` without an adapter (no checker pass in the stored draws,
so 0 %). A cell whose sampling was cut is drawn hollow and its whisker spans the all-fail to
all-accepted bounds; a cell that was not sampled is left out.
Input: handcheck5.py's round5_summary.json. Output: <out>.pdf, .svg and .png.

Usage: python fig_a9.py [--summary .../round5_summary.json] [--out .../figures/a9_size_examples]
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
Q = "genuine_rate"
SIZES = [("rank1_n64", 655_360, "655,360\nrank 1, one projection")]
REF_SIZE = (134_217_728, "134,217,728\nrank 16, all projections")
EXAMPLES = [("full_n4", 4), ("full_n16", 16)]
COLOR = S.OKABE_ITO["blue"]


def reference_name(cells: dict) -> str:
    return "ref_again" if "ref_again" in cells else "ref"


def whisker(res: dict, cell: str) -> tuple[float, float, float, bool]:
    """(value, low end, high end, cut?) of one cell's reader-accepted rate."""
    v = res["cells"][cell][Q]
    if cell in res.get("unfinished_draws", {}) and "bounds" in res:
        lo = res["bounds"]["unfinished_as_fail"]["cells"][cell][Q]["lo"]
        hi = res["bounds"]["unfinished_as_accepted_pass"]["cells"][cell][Q]["hi"]
        return v["value"], min(lo, v["lo"]), max(hi, v["hi"]), True
    return v["value"], v["lo"], v["hi"], False


def dot(ax, x: float, res: dict, cell: str, *, ref: bool = False, side: int = 1) -> None:
    y, lo, hi, cut = whisker(res, cell)
    ax.errorbar(
        [x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=S.INK2, elinewidth=0.8,
        capsize=2.0, zorder=3,
    )  # fmt: skip
    ax.plot(
        [x], [y], linestyle="none", marker="s" if ref else "o", markersize=5.0,
        markerfacecolor=S.WHITE if cut else COLOR, markeredgecolor=COLOR, markeredgewidth=1.0,
        zorder=4,
    )  # fmt: skip
    note = f"{100 * y:.1f}%" + (" (reference)" if ref else "") + (" (cut)" if cut else "")
    ax.annotate(
        note, (x, y), xytext=(7 * side, 0), textcoords="offset points", va="center",
        ha="left" if side > 0 else "right", fontsize=S.FS_TICK, color=S.INK,
    )  # fmt: skip


def marks(ax, res: dict, *, label: bool) -> None:
    """The final RL model's rate with its interval, and dpo without an adapter, across the panel."""
    # the same round's labels when the final model's items were read again
    f = res["cells"].get("final_r5", res["cells"]["final_r4"])[Q]
    ax.axhspan(f["lo"], f["hi"], color=S.SHADE, zorder=0)
    ax.axhline(f["value"], color=S.REF, linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=S.REF, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    if label:
        t = ax.get_yaxis_transform()
        ax.text(
            0.02, f["value"] + 0.012, f"final RL model, {100 * f['value']:.1f}%", transform=t,
            fontsize=S.FS_TICK, color=S.INK2, va="bottom",
        )  # fmt: skip
        ax.text(
            0.02, 0.012, "dpo without an adapter, 0%", transform=t, fontsize=S.FS_TICK,
            color=S.INK2, va="bottom",
        )  # fmt: skip


def draw(fig, res: dict) -> None:
    cells = res["cells"]
    ref = reference_name(cells)
    a, b = fig.subplots(1, 2, sharey=True)
    # (a) adapter size, both on 64 examples
    a.set_xscale("log")
    ticks, names = [], []
    for cell, x, name in SIZES:
        ticks.append(x)
        names.append(name if cell in cells else name + "\n(not sampled)")
        if cell in cells:
            dot(a, x, res, cell)
    ticks.append(REF_SIZE[0])
    names.append(REF_SIZE[1])
    dot(a, REF_SIZE[0], res, ref, ref=True, side=-1)
    a.set_xlim(1.2e5, 7e8)
    a.set_xticks(ticks, names)
    a.minorticks_off()
    a.set_xlabel("trained weights (64 examples)")
    a.set_ylabel("reader-accepted passes, share of all draws")
    S.panel_title(a, "a", "adapter size")
    # (b) number of examples, full-size adapter
    b.set_xscale("log", base=2)
    for cell, n in EXAMPLES:
        if cell in cells:
            dot(b, n, res, cell)
    dot(b, 64, res, ref, ref=True, side=-1)
    b.set_xlim(2.6, 100)
    b.set_xticks(
        [4, 16, 64],
        [str(n) if c is None or c in cells else f"{n}\n(not sampled)"
         for c, n in (("full_n4", 4), ("full_n16", 16), (None, 64))],
    )  # fmt: skip
    b.minorticks_off()
    b.set_xlabel("training examples (full-size adapter)")
    S.panel_title(b, "b", "number of examples")
    top = max(0.7, max(whisker(res, c)[2] for c in cells) + 0.06)
    for ax, lab in ((a, True), (b, False)):
        marks(ax, res, label=lab)
        ax.set_ylim(-0.03, top)
        S.pct_axis(ax, "y")
        S.light_grid(ax, "y")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summary", type=Path, default=V2 / "handcheck/round5/round5_summary.json")
    ap.add_argument("--out", type=Path, default=FIGS / "a9_size_examples")
    a = ap.parse_args()
    res = json.loads(a.summary.read_text())
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.5)
        draw(fig, res)
        paths = S.save(fig, a.out)
        S.close(fig)
    print("wrote", *paths)


if __name__ == "__main__":
    main()
