# ruff: noqa: E501
"""Token-level profile figure: where the distance between each checkpoint and s2300 sits, position by position.

    fig_tokens.py [--summary /scratch/cowifornia/nd-rl/olmo_elicit/tokens_summary.json] [--out /scratch/cowifornia/nd-rl/olmo_elicit/figures/tokens_profile] [--set all]

All on s2300's own responses (teacher-forced, step-0 pass), against RL step, per segment (total / thinking / answer):
(a) share of positions with approximate JS(t, s2300) > 0.01 (filled) and > 0.1 nats (hollow, dotted);
(b) share of positions where s2300's sampled token is NOT top-1 (filled) / NOT in the top 5 (hollow, dotted) under t,
    log scale; s2300's own point is the sampling floor;
(c) NLL excess over s2300, nats per token (= the L0 gap);
(d) share of the net excess carried by the top 1 % / 5 % / 10 % of positions (total and answer);
(e, f) share of the net excess in each decile of s2300's entropy at the position, one line per checkpoint.
Intervals: 95 % bootstrap over responses. Definitions in analyze_tokens.py. Style: code/ndbench/figures/style.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fig_edl import CKPTS, SEG_STYLE, _empty, _yerr, ckpt_colour  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from code.ndbench.figures import style as st  # noqa: E402

SUMMARY = Path("/scratch/cowifornia/nd-rl/olmo_elicit/tokens_summary.json")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/figures/tokens_profile")
SEGS = ("total", "think", "answer")
Q_LS = {"1pct": ":", "5pct": "-", "10pct": "--"}


def _step_axis(ax) -> None:
    ax.set_xlim(-80, 2380)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2300])
    ax.set_xticklabels(["0", "500", "1000", "1500", "", "2300"])
    ax.set_xlabel("RL step")
    st.light_grid(ax)


def _series(ax, res, names, tset, seg, key, *, transform=None, hollow=False, ls=None, with_ci=True):
    sty = SEG_STYLE[seg]
    xs, ys, cis = [], [], []
    for c in names:
        d = res[c]["sets"].get(tset, {}).get(seg, {})
        v = d.get(key)
        if v is None or not np.isfinite(v):
            continue
        lo, hi = d.get(key + "_ci", [np.nan, np.nan]) if with_ci else [np.nan, np.nan]
        if transform:
            v, lo, hi = (
                transform(v),
                transform(hi),
                transform(lo),
            )  # decreasing transform flips the interval
        xs.append(res[c]["step"])
        ys.append(v)
        cis.append([lo, hi])
    if not xs:
        return False
    ax.plot(xs, ys, color=sty["color"], ls=ls or sty["ls"], lw=1.0, zorder=2)
    for x, y, cc in zip(xs, ys, cis, strict=True):
        if np.isfinite(cc[0]):
            ax.errorbar(
                [x],
                [y],
                yerr=_yerr(y, sorted(cc)),
                fmt="none",
                ecolor=sty["color"],
                elinewidth=0.6,
                capsize=1.2,
                zorder=2,
            )
        ax.plot(
            [x],
            [y],
            marker=sty["marker"],
            ms=3.0,
            color=sty["color"],
            mfc="white" if hollow else sty["color"],
            mew=0.8,
            ls="none",
            zorder=3,
        )
    return True


def make_figure(summary: dict, tset: str = "all"):
    res = summary.get("checkpoints", {})
    names = [c for c in CKPTS if c in res]
    others = [c for c in names if not res[c].get("is_reference")]
    fig = st.new_figure(st.FULL_W, 6.9)
    axs = fig.subplots(3, 2)

    ax = axs[0, 0]
    drawn = False
    for seg in SEGS:
        drawn |= _series(ax, res, others, tset, seg, "share_js_gt_0.01")
        drawn |= _series(ax, res, others, tset, seg, "share_js_gt_0.1", hollow=True, ls=":")
    st.panel_title(ax, "a", "positions with JS$(t, s2300)$ > 0.01 / > 0.1")
    ax.set_ylabel("share of positions")
    _step_axis(ax)
    if not drawn:
        _empty(ax, "no checkpoints yet")

    ax = axs[0, 1]
    drawn = False
    for seg in SEGS:
        drawn |= _series(ax, res, names, tset, seg, "share_top1", transform=lambda v: 1 - v)
        drawn |= _series(
            ax, res, names, tset, seg, "share_top5", transform=lambda v: 1 - v, hollow=True, ls=":"
        )
    ax.set_yscale("log")
    st.panel_title(ax, "b", "s2300's token not top-1 / not in top 5 under $t$")
    ax.set_ylabel("share of positions (log)")
    _step_axis(ax)
    if not drawn:
        _empty(ax, "no checkpoints yet")

    ax = axs[1, 0]
    drawn = False
    for seg in SEGS:
        drawn |= _series(ax, res, names, tset, seg, "excess_mean")
    ax.axhline(0, color=st.REF, lw=0.6, zorder=1)
    st.panel_title(ax, "c", "NLL excess over s2300 (the $L_0$ gap)")
    ax.set_ylabel("nats per token")
    _step_axis(ax)
    if not drawn:
        _empty(ax, "no checkpoints yet")
    else:
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=SEG_STYLE[s]["color"],
                    marker=SEG_STYLE[s]["marker"],
                    ls=SEG_STYLE[s]["ls"],
                    ms=3.0,
                    lw=1.0,
                    label=SEG_STYLE[s]["label"],
                )
                for s in SEGS
            ],
            fontsize=st.FS_TICK - 0.5,
            loc="upper right",
            title="segment (a-d)",
            title_fontsize=st.FS_TICK - 0.5,
        )

    ax = axs[1, 1]
    drawn = False
    for seg in ("total", "answer"):
        for q, ls in Q_LS.items():
            drawn |= _series(
                ax, res, others, tset, seg, f"excess_share_top{q}_net", ls=ls, with_ci=False
            )
    ax.axhline(1, color=st.REF, lw=0.6, zorder=1)
    st.panel_title(ax, "d", "net excess in the top 1 / 5 / 10 % of positions")
    ax.set_ylabel("share of excess")
    _step_axis(ax)
    if drawn:
        ax.legend(
            handles=[
                Line2D([], [], color=st.INK, ls=ls, lw=1.0, label=f"top {q.replace('pct', ' %')}")
                for q, ls in Q_LS.items()
            ],
            fontsize=st.FS_TICK - 0.5,
            loc="best",
        )
    else:
        _empty(ax, "no checkpoints yet")

    for col, seg in enumerate(("total", "answer")):
        ax = axs[2, col]
        drawn = []
        for c in others:
            dec = res[c]["sets"].get(tset, {}).get(seg, {}).get("entropy_deciles", [])
            if not dec:
                continue
            x = [b["decile"] for b in dec]
            y = [b["share_of_net_excess"] for b in dec]
            off = res[c].get("off_released_line")
            ax.plot(
                x,
                y,
                color=ckpt_colour(c),
                lw=1.0,
                ls="--" if off else "-",
                marker="o",
                ms=2.6,
                mfc="white" if off else ckpt_colour(c),
                mew=0.7,
                zorder=2,
            )
            drawn.append(c)
        ax.axhline(0.1, color=st.REF, lw=0.6, ls=":", zorder=1)
        st.panel_title(ax, "ef"[col], f"excess by s2300 entropy decile, {seg}")
        ax.set_xlabel("decile of s2300's entropy at the position (1 = lowest)")
        ax.set_ylabel("share of net excess")
        ax.set_xticks(range(1, 11))
        st.light_grid(ax)
        if not drawn:
            _empty(ax, "no checkpoints yet")
    ck_h = [
        Line2D(
            [],
            [],
            color=ckpt_colour(c),
            marker="o",
            ms=2.6,
            lw=1.0,
            ls="--" if res[c].get("off_released_line") else "-",
            mfc="white" if res[c].get("off_released_line") else ckpt_colour(c),
            label=f"{c} ({res[c]['step']})",
        )
        for c in others
    ]
    if ck_h:
        fig.legend(
            handles=ck_h,
            fontsize=st.FS_TICK,
            loc="outside lower center",
            ncol=min(len(ck_h), 5),
            title="(e, f) checkpoint (RL step); dotted line = uniform 0.1.   (a, b) filled: JS > 0.01 / not top-1; hollow: JS > 0.1 / not in top 5",
            title_fontsize=st.FS_TICK - 0.5,
        )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--set", dest="tset", choices=["all", "train", "test", "reach"], default="all")
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    with st.paper_style():
        fig = make_figure(s, a.tset)
        paths = st.save(fig, a.out)
        st.close(fig)
    print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
