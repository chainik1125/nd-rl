"""The propensity-capability figure: one panel per direction (columns) and set (rows); x is the exact first-move
propensity (E), y the share of forced attempts solved (F); an arrow per seed from the base to its RL end.

    code/.venv/bin/python -m code.experiments.current.propensity_capability_20261109.fig \
        --results code/experiments/current/propensity_capability_20261109/results/results.json \
        --out code/experiments/current/propensity_capability_20261109/results/propensity_capability

Both axes are log10, so an arrow's horizontal and vertical extents are the two Δlog10 the reading uses, and the
dotted diagonals are constant P × C (success through d's opening). The x axis spans as many decades as the data need,
down to 1e-6 %, so propensities near zero stay visible; a point below that is drawn on the edge as an open "<". Below
1 % (the grey band, where a change is not read, PREREG) each decade is drawn a third as wide as above it, so the
directions that live between 1 and 100 % keep half of each panel; the diagonals are drawn only above 1 %. Colour and
line style both encode the dose, marker shape the seed (hollow at the base, filled at the RL end), so the figure reads
in greyscale. Doses are at unequal RL compute: the legend gives x at the RL ends.
"""

# ruff: noqa: E501, B905
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from code.ndbench.figures import style as S

FORCED = ("reductio", "negation", "conditional", "cases")
SETS = ("strict106", "transfer_other")
SET_LABEL = {"strict106": "the strict 106", "transfer_other": "other 240"}
OI = S.OKABE_ITO
DOSE_STYLE = {  # colour and line style both carry the dose
    "0": (OI["blue"], "-", "0 % dose"),
    "0.1": (OI["reddish_purple"], (0, (4, 1.6)), "0.1 % dose"),
    "1": (OI["vermillion"], (0, (1.2, 1.2)), "1 % dose"),
}
SEED_MARKERS = ("o", "s", "^")
X_FLOOR = 1e-8  # 1e-6 %: the left edge at most
N_MIN = 20
FLOOR = 0.01


R = 1 / 3  # width of a decade below 1 %, relative to one above


def _fwd(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))
    return np.where(u >= -2, u, -2 + (u + 2) * R)


def _inv(t):
    t = np.asarray(t, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(t >= -2, 10.0**t, 10.0 ** (-2 + (t + 2) / R))


def _pct_label(v: float, _pos=None) -> str:
    p = 100 * v
    if p >= 0.01:
        return f"{p:g}%"
    return f"$10^{{{int(round(math.log10(p)))}}}$%"


def _limits(res: dict) -> tuple[float, float]:
    ps, cs = [], []
    for ln in res["lines"].values():
        for c in ln.get("cells", {}).values():
            p = c["primary"]
            if "P" in p:
                ps += [p["P"]["base"]["est"], p["P"]["end"]["est"]]
                cs += [p["C"]["base"]["est"], p["C"]["end"]["est"]]
    for lv in res["levels"].values():
        for s in SETS:
            for d in FORCED:
                c = lv.get(s, {}).get(d, {})
                if "P_first" in c:
                    ps.append(c["P_first"]["est"])
                    cs.append(c["C"]["est"])
    pmin = min([p for p in ps if p > 0] or [1e-3])
    cmin = min([c for c in cs if c > 0] or [0.01])
    xlo = max(10 ** math.floor(math.log10(pmin)), X_FLOOR)
    ylo = min(10 ** math.floor(math.log10(cmin)), 0.01)
    return xlo, ylo


def _xy(v: float, lo: float) -> tuple[float, bool]:
    return (v, False) if v >= lo else (lo, True)


def render(res: dict, out: Path) -> list[Path]:
    xlo, ylo = _limits(res)
    xs_end: dict[str, list[float]] = {}
    for ln in res["lines"].values():
        if ln.get("x_end") is not None:
            xs_end.setdefault(ln["dose"], []).append(ln["x_end"])
    full = [cid for cid, m in res["ckpts"].items() if m["dose"] == "full"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.95)
        axes = fig.subplots(2, 4, sharex=True, sharey=True)
        letters = iter("abcdefgh")
        for row, s in enumerate(SETS):
            for col, d in enumerate(FORCED):
                ax = axes[row][col]
                ax.set_xscale("function", functions=(_fwd, _inv))
                ax.set_yscale("log")
                ax.set_xlim(xlo / 1.6, 1.6)
                ax.set_ylim(ylo / 1.4, 1.4)
                ax.axvspan(xlo / 1.6, FLOOR, color=S.SHADE, lw=0, zorder=0)
                # constant P x C, one per decade, above 1 % only (the axis is compressed below)
                for k in range(-6, 1):
                    sv = 10.0**k
                    xx = np.geomspace(max(sv / 1.4, FLOOR), 1.6, 20)
                    if xx[0] < 1.6:
                        ax.plot(
                            xx,
                            sv / xx,
                            ls=(0, (1, 2.2)),
                            lw=0.45,
                            color=S.REF,
                            alpha=0.55,
                            zorder=1,
                        )
                n = None
                for ln in res["lines"].values():
                    c = ln.get("cells", {}).get(f"{s}|{d}")
                    if not c:
                        continue
                    p = c["primary"]
                    n = p["n_applicable"] if n is None else max(n, p["n_applicable"])
                    if "P" not in p:
                        continue
                    col_, lsty, _ = DOSE_STYLE[ln["dose"]]
                    mk = SEED_MARKERS[ln["seed"] % 3]
                    pts = []
                    for end, key in ((False, "base"), (True, "end")):
                        px, cx = p["P"][key], p["C"][key]
                        x, clipped = _xy(px["est"], xlo)
                        y = max(cx["est"], ylo)
                        pts.append((x, y))
                        xl, xh = max(px["ci"][0], xlo), max(px["ci"][1], xlo)
                        yl, yh = max(cx["ci"][0], ylo), max(cx["ci"][1], ylo)
                        ax.plot(
                            [xl, xh],
                            [y, y],
                            color=col_,
                            lw=0.45,
                            alpha=0.6,
                            zorder=2,
                            solid_capstyle="butt",
                        )
                        ax.plot(
                            [x, x],
                            [yl, yh],
                            color=col_,
                            lw=0.45,
                            alpha=0.6,
                            zorder=2,
                            solid_capstyle="butt",
                        )
                        ax.plot(x, y, marker="<" if clipped else mk, ms=3.4 if end else 3.2, ls="none", zorder=4,
                                mfc=col_ if end and not clipped else S.WHITE, mec=col_, mew=0.75)  # fmt: skip
                        if end and ln.get("alarm"):
                            ax.plot(x, y, marker="x", ms=2.6, mew=0.6, color=S.INK, zorder=5)
                    ax.annotate("", xy=pts[1], xytext=pts[0], zorder=3,
                                arrowprops={"arrowstyle": "-|>,head_length=0.32,head_width=0.16", "color": col_, "lw": 0.9,
                                            "linestyle": lsty, "shrinkA": 2.2, "shrinkB": 2.6})  # fmt: skip
                for cid in full:
                    c = res["levels"][cid].get(s, {}).get(d, {})
                    if c.get("n", 0) >= N_MIN:
                        x, _ = _xy(c["P_first"]["est"], xlo)
                        ax.plot(
                            x,
                            max(c["C"]["est"], ylo),
                            marker="*",
                            ms=5.0,
                            color=S.INK,
                            mec=S.WHITE,
                            mew=0.3,
                            ls="none",
                            zorder=4,
                        )
                n = n or 0
                if n == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "never applicable",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=S.FS_TICK,
                        color=S.INK2,
                    )
                elif n < N_MIN:
                    ax.text(
                        0.5,
                        0.5,
                        f"n = {n} < {N_MIN}:\nnot read",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=S.FS_TICK,
                        color=S.INK2,
                    )
                S.panel_title(ax, next(letters), f"{d}, n = {n}", size=S.FS_TICK + 0.5)
                lo_dec = int(math.floor(math.log10(xlo)))
                # labels at 100 %, 1 % and one decade inside the compressed zone (more would collide)
                ax.xaxis.set_major_locator(
                    FixedLocator(
                        [1.0, 0.01] + ([10.0 ** max(lo_dec + 2, -5)] if lo_dec <= -4 else [])
                    )
                )
                ax.xaxis.set_minor_locator(
                    FixedLocator([10.0**k for k in range(0, lo_dec - 1, -1)])
                )
                ax.xaxis.set_major_formatter(FuncFormatter(_pct_label))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_locator(
                    FixedLocator(
                        [10.0**k for k in range(0, int(math.floor(math.log10(ylo))) - 1, -1)]
                    )
                )
                ax.yaxis.set_major_formatter(FuncFormatter(_pct_label))
                ax.yaxis.set_minor_formatter(NullFormatter())
            axes[row][0].set_ylabel(f"{SET_LABEL[s]}\nsolved when forced")
        for ax in axes[1]:
            ax.set_xlabel("opens with d (exact P)")
        hs = []
        for dose, (c, ls, lab) in DOSE_STYLE.items():
            xs = xs_end.get(dose)
            xl = f", RL end x {min(xs):.2g}–{max(xs):.2g}" if xs else ""
            hs.append(Line2D([], [], color=c, ls=ls, lw=0.9, label=lab + xl))
        hs += [
            Line2D(
                [],
                [],
                ls="none",
                marker=m,
                mfc=S.WHITE,
                mec=S.INK2,
                mew=0.75,
                ms=3.4,
                label=f"seed {i}",
            )
            for i, m in enumerate(SEED_MARKERS)
        ]
        hs.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                mfc=S.INK2,
                mec=S.INK2,
                ms=3.4,
                label="filled: RL end",
            )
        )
        if full:
            hs.append(
                Line2D([], [], ls="none", marker="*", ms=5, color=S.INK, label="full-corpus base")
            )
        if any(ln.get("alarm") for ln in res["lines"].values()):
            hs.append(
                Line2D([], [], ls="none", marker="x", ms=3, mew=0.6, color=S.INK, label="alarm end")
            )
        hs.append(
            Patch(facecolor=S.SHADE, edgecolor="none", label="< 1 %: not read, axis ×3 compressed")
        )
        hs.append(Line2D([], [], ls=(0, (1, 2.2)), lw=0.6, color=S.REF, label="P × C constant"))
        fig.legend(
            handles=hs, loc="outside lower center", ncol=5, frameon=False, fontsize=S.FS_TICK - 0.5
        )
        paths = S.save(fig, out)
        S.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in render(json.loads(Path(a.results).read_text()), Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
