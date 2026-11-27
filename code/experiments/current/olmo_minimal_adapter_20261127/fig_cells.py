# ruff: noqa: E501, SIM115
"""Pass rate of the minimal-adapter cells -> /scratch/cowifornia/nd-rl/olmo_elicit/v2/figures/cells.{pdf,svg,png}.

(a) G = share of the gap closed, (cell - checkpoint plain) / (final plain - checkpoint plain), against the number of trained
    weights P (log10), at n = 64 examples, one series per example source; the steering line G = 0.80, the final model (G = 1)
    and the checkpoint (G = 0) as reference lines, and the prompt control (checkpoint + the answer-only instruction, no
    adapter) as a horizontal band (its G with interval).
(b) G against n (log2 axis), one panel per adapter size P that has cells at more than one n, per source.
Intervals: 95 % two-level paired bootstrap of G (analyze_cells.py), whiskers. Cells that were not sampled are skipped.
Usage: python fig_cells.py [--summary .../v2/cells_summary.csv] [--ckpt dpo] [--out .../v2/figures/cells]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # .../code
from ndbench.figures import style as S  # noqa: E402

V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")
SOURCE_STYLE = ["blue", "vermillion", "bluish_green", "reddish_purple", "orange"]
MARKERS = ["o", "s", "^", "D", "v"]
N_MAX = 64


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path, ckpt: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        if r["ckpt"] != ckpt or num(r["g"]) is None or num(r["P"]) is None or num(r["n"]) is None:
            continue
        rows.append(
            {
                "source": r["source"],
                "family": r["family"],
                "P": float(r["P"]),
                "n": int(float(r["n"])),
                "y": float(r["g"]),
                "lo": float(r["g_lo"]),
                "hi": float(r["g_hi"]),
            }
        )
    return rows


def read_control(path: Path, ckpt: str, name: str = "answeronly"):
    """(G, lo, hi) of the plain prompt-control cell, or None."""
    for r in csv.DictReader(open(path)):
        if r["ckpt"] == ckpt and r["cell"] == f"plain:{name}" and num(r["g"]) is not None:
            return float(r["g"]), float(r["g_lo"]), float(r["g_hi"])
    return None


def style_of(sources: list[str]) -> dict:
    C = S.OKABE_ITO
    return {s: (C[SOURCE_STYLE[i % 5]], MARKERS[i % 5]) for i, s in enumerate(sources)}


def line(ax, pts, color, marker, label, dx=0.0):
    """pts: [(x, y, lo, hi)] sorted by x; whiskers plus a line."""
    pts = sorted(pts)
    x = [p[0] + dx for p in pts]
    y = [p[1] for p in pts]
    ax.errorbar(
        x,
        y,
        yerr=[[max(0.0, p[1] - p[2]) for p in pts], [max(0.0, p[3] - p[1]) for p in pts]],
        color=color,
        marker=marker,
        ms=3.6,
        lw=1.0,
        elinewidth=0.7,
        capsize=1.6,
        label=label,
        zorder=3,
    )


def reference(ax, control=None):
    ax.axhline(1.0, color=S.REF, lw=0.8, zorder=1, label="final model, plain (G = 1)")
    ax.axhline(0.0, color=S.INK, lw=0.8, ls="--", zorder=1, label="checkpoint, plain (G = 0)")
    ax.axhline(
        0.8, color=S.OKABE_ITO["black"], lw=0.7, ls=":", zorder=1, label="steering line G = 0.80"
    )
    if control:
        ax.axhspan(control[1], control[2], color=S.OKABE_ITO["yellow"], alpha=0.35, lw=0, zorder=0)
        ax.axhline(
            control[0],
            color=S.OKABE_ITO["orange"],
            lw=0.8,
            zorder=1,
            label="prompt control (answer-only)",
        )


def render(rows: list[dict], out: Path, ckpt: str = "dpo", control=None) -> list[Path]:
    sources = sorted({r["source"] for r in rows})
    sty = style_of(sources)
    a_rows = [r for r in rows if r["n"] == N_MAX]
    sizes: dict[tuple, set] = {}
    for r in rows:
        sizes.setdefault((r["family"], r["P"]), set()).add(r["n"])
    n_groups = sorted((k for k, v in sizes.items() if len(v) > 1), key=lambda k: k[1])
    n_panels = 1 + max(1, len(n_groups))
    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.45 * nrows + 0.55)
        axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False).ravel()
        ax = axes[0]
        reference(ax, control)
        for i, s in enumerate(sources):
            pts = [
                (math.log10(r["P"]), r["y"], r["lo"], r["hi"]) for r in a_rows if r["source"] == s
            ]
            if pts:
                line(ax, pts, *sty[s], f"examples: {s}", dx=(i - (len(sources) - 1) / 2) * 0.03)
        xs = sorted({math.log10(r["P"]) for r in rows})
        ax.set_xticks(
            range(int(math.floor(min(xs, default=2))), int(math.ceil(max(xs, default=8))) + 1)
        )
        ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ax.get_xticks()])
        ax.set_xlabel("trained weights P")
        S.panel_title(ax, "a", f"{ckpt}, n = {N_MAX} examples")
        if n_groups:
            for gi, (fam, P) in enumerate(n_groups):
                bx = axes[1 + gi]
                reference(bx, control)
                for s in sources:
                    pts = [
                        (r["n"], r["y"], r["lo"], r["hi"])
                        for r in rows
                        if r["source"] == s and r["family"] == fam and r["P"] == P
                    ]
                    if len(pts) > 1:
                        line(bx, pts, *sty[s], f"examples: {s}")
                bx.set_xscale("log", base=2)
                ns = sorted({r["n"] for r in rows if r["family"] == fam and r["P"] == P})
                bx.set_xticks(ns)
                bx.set_xticklabels([str(n) for n in ns])
                bx.minorticks_off()
                bx.set_xlabel("examples n")
                S.panel_title(bx, chr(ord("b") + gi), f"{fam}, P = {int(P):,}")
        else:
            bx = axes[1]
            reference(bx, control)
            bx.set_xlabel("examples n")
            bx.text(
                0.5,
                0.5,
                "no cells at several n yet",
                transform=bx.transAxes,
                ha="center",
                fontsize=S.FS_TICK,
                color=S.REF,
            )
            S.panel_title(bx, "b", "G against n")
        for extra in axes[n_panels:]:
            extra.set_visible(False)
        for a in axes[:n_panels]:
            a.set_ylim(-0.15, 1.25)
            S.light_grid(a)
        for r_i in range(nrows):
            axes[r_i * ncols].set_ylabel("share of the gap closed, G")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="outside lower center",
            ncol=min(len(lab), 4),
            fontsize=S.FS_TICK,
            frameon=False,
        )
        paths = S.save(fig, out)
        S.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(V2 / "cells_summary.csv"))
    ap.add_argument("--ckpt", default="dpo")
    ap.add_argument("--out", default=str(V2 / "figures" / "cells"))
    a = ap.parse_args()
    rows = read_rows(Path(a.summary), a.ckpt)
    paths = render(rows, Path(a.out), a.ckpt, read_control(Path(a.summary), a.ckpt))
    print(f"wrote {len(rows)} cells -> {', '.join(str(p) for p in paths)}")


if __name__ == "__main__":
    main()
