# ruff: noqa: E501
"""In-context figure: NLL of the final model's responses with k of its other responses in context, across checkpoints.

    fig_icl.py [--summary /scratch/cowifornia/nd-rl/olmo_elicit/icl_summary.json] [--out /scratch/cowifornia/nd-rl/olmo_elicit/figures/icl_curve] [--set test|reach]

(a, b) L(k) in nats per response token against RL step, one line per k (k = 0 is step 0's L_0), total and answer;
(c, d) L against k, one line per checkpoint (colour = checkpoint, as in fig_edl), total and answer. 95 % intervals over
targets and demo draws. Works with any subset of checkpoints and ks. Drawn through code/ndbench/figures/style.py.
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

from fig_edl import CKPTS, _empty, _yerr, ckpt_colour  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from code.ndbench.figures import style as st  # noqa: E402

SUMMARY = Path("/scratch/cowifornia/nd-rl/olmo_elicit/icl_summary.json")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/figures/icl_curve")
K_STYLE = {
    0: ("#000000", "D", "-"),
    1: ("#56B4E9", "o", "--"),
    2: ("#0072B2", "s", "-."),
    4: ("#D55E00", "^", "-"),
    8: ("#CC79A7", "v", ":"),
}


def k_style(k: int):
    return K_STYLE.get(k, ("#6b6b6b", "x", ":"))


def make_figure(summary: dict, tset: str = "test"):
    res = summary.get("checkpoints", {})
    names = [c for c in CKPTS if c in res]
    fig = st.new_figure(st.FULL_W, 4.7)
    axs = fig.subplots(2, 2)
    for col, seg in enumerate(("total", "answer")):
        ax = axs[0, col]
        ks = sorted({int(k) for c in names for k in res[c].get(tset, {}).get(seg, {})})
        for k in ks:
            cs = [c for c in names if str(k) in res[c][tset][seg]]
            if not cs:
                continue
            colr, mk, ls = k_style(k)
            xs = [res[c]["step"] for c in cs]
            ys = [res[c][tset][seg][str(k)]["L"] for c in cs]
            ax.plot(xs, ys, color=colr, ls=ls, lw=1.0, zorder=2)
            for c, x, y in zip(cs, xs, ys, strict=True):
                ax.errorbar(
                    [x],
                    [y],
                    yerr=_yerr(y, res[c][tset][seg][str(k)]["ci"]),
                    fmt="none",
                    ecolor=colr,
                    elinewidth=0.6,
                    capsize=1.4,
                    zorder=2,
                )
                ax.plot(
                    [x],
                    [y],
                    marker=mk,
                    ms=3.2,
                    color=colr,
                    mfc="white" if res[c].get("off_released_line") else colr,
                    mew=0.8,
                    ls="none",
                    zorder=3,
                )
        if not ks:
            _empty(ax, "no in-context rows yet")
        else:
            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=k_style(k)[0],
                        marker=k_style(k)[1],
                        ls=k_style(k)[2],
                        ms=3.2,
                        lw=1.0,
                        label=f"k = {k}",
                    )
                    for k in ks
                ],
                fontsize=st.FS_TICK - 0.5,
                loc="best",
            )
        st.panel_title(ax, "ab"[col], f"$L(k)$ over RL steps, {seg} ({tset})")
        ax.set_xlabel("RL step")
        ax.set_ylabel("nats per token")
        ax.set_xlim(-80, 2380)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2300])
        ax.set_xticklabels(["0", "500", "1000", "1500", "", "2300"])
        st.light_grid(ax)

        ax = axs[1, col]
        drawn = []
        for c in names:
            d = res[c].get(tset, {}).get(seg, {})
            if not d:
                continue
            ks_c = sorted(int(k) for k in d)
            x = np.asarray(ks_c, float)
            y = np.asarray([d[str(k)]["L"] for k in ks_c])
            colr = ckpt_colour(c)
            off = res[c].get("off_released_line")
            ax.plot(x, y, color=colr, lw=1.0, ls="--" if off else "-", zorder=2)
            for k, xi, yi in zip(ks_c, x, y, strict=True):
                ax.errorbar(
                    [xi],
                    [yi],
                    yerr=_yerr(yi, d[str(k)]["ci"]),
                    fmt="none",
                    ecolor=colr,
                    elinewidth=0.6,
                    capsize=1.2,
                    zorder=2,
                )
            ax.plot(
                x,
                y,
                marker="o",
                ms=3.0,
                color=colr,
                mfc="white" if off else colr,
                mew=0.7,
                ls="none",
                zorder=3,
            )
            drawn.append(c)
        if drawn:
            kk = sorted({int(k) for c in drawn for k in res[c][tset][seg]})
            ax.set_xticks(kk)
        else:
            _empty(ax, "no in-context rows yet")
        st.panel_title(ax, "cd"[col], f"$L$ against demonstrations $k$, {seg}")
        ax.set_xlabel("demonstrations in context, $k$")
        ax.set_ylabel("nats per token")
        st.light_grid(ax)
        if col == 1 and drawn:
            hs = [
                Line2D(
                    [],
                    [],
                    color=ckpt_colour(c),
                    marker="o",
                    ms=3.0,
                    lw=1.0,
                    mfc="white" if res[c].get("off_released_line") else ckpt_colour(c),
                    ls="--" if res[c].get("off_released_line") else "-",
                    label=f"{c} ({res[c]['step']})",
                )
                for c in drawn
            ]
            fig.legend(
                handles=hs,
                fontsize=st.FS_TICK,
                loc="outside lower center",
                title="(c, d) checkpoint (RL step)",
                title_fontsize=st.FS_TICK,
                ncol=len(hs) if len(hs) <= 6 else 5,
            )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--set", dest="tset", choices=["test", "reach"], default="test")
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    with st.paper_style():
        fig = make_figure(s, a.tset)
        paths = st.save(fig, a.out)
        st.close(fig)
    print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
