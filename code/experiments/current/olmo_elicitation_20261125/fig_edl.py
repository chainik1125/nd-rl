# ruff: noqa: E501
"""Stage 1 figure: likelihood and EDL of the s2300 responses across Olmo 3.1 Think 32B RL checkpoints.

    fig_edl.py [--summary /scratch/cowifornia/nd-rl/olmo_elicit/edl_summary.json] [--out /scratch/cowifornia/nd-rl/olmo_elicit/figures/edl_curve]
               [--conv norm|sum] [--rule mdl|min]

Reads analyze_edl.py's summary; works with any subset of checkpoints (a checkpoint with only step 0 appears in panel a).
(a) L_0, nats per response token of the final model's responses under each checkpoint before fine-tuning (train + test
    prompts), total / thinking / answer, 95 % prompt-bootstrap bands.
(b) EDL* per response token (default: length-normalised, lr by the shortest online code length, + ln(#lrs)), total /
    thinking / answer, each segment with its own lr; 95 % intervals over test prompts (and orders where there are two).
    s2300 is the null control (PREREG C1: its EDL* should be near 0); the panel says so.
(c, d) EDL per token over n' (examples seen), one line per checkpoint (colour = RL step), total and answer.
Hollow markers: a checkpoint off the released line (step 200). Drawn through code/ndbench/figures/style.py.

Second figure (POST HOC readouts, --learning-out, default figures/edl_learning):
(a, b) L_test against n' in {0, 8, 16, 32, 64} per checkpoint at the segment's code-length lr, total and answer, with
    s2300's L_0 (test) as a horizontal line; no per-point intervals (the unpaired prompt intervals are in the JSON and
    are shared by all checkpoints; the paired comparisons are (c) and posthoc.delta_L);
(c) delta EDL* = EDL*(t) - EDL*(s2300) per token (length-normalised) over RL step, total and answer, paired intervals;
(d) share of the gap to s2300 closed by 64 examples, (L_0(t) - L_64(t)) / (L_0(t) - L_0(s2300)), test (filled) and
    reach (hollow), total and answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib import colormaps  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from code.ndbench.figures import style as st  # noqa: E402

SUMMARY = Path("/scratch/cowifornia/nd-rl/olmo_elicit/edl_summary.json")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/figures/edl_curve")
OUT_LEARNING = Path("/scratch/cowifornia/nd-rl/olmo_elicit/figures/edl_learning")
FINAL = "s2300"
CKPTS = ["dpo", "s0050", "s0100", "s0200", "s0500", "s0750", "s1100", "s1600", "s2300"]
# segment encodings (Okabe-Ito): never colour alone
SEG_STYLE = {
    "total": {"color": "#000000", "marker": "D", "ls": "-", "label": "total"},
    "think": {"color": "#0072B2", "marker": "o", "ls": "--", "label": "thinking"},
    "answer": {"color": "#D55E00", "marker": "s", "ls": "-", "label": "answer"},
}
STEP_CMAP = colormaps["viridis"]


def ckpt_colour(name: str) -> str:
    """Ordinal in RL step (position among all nine checkpoints), so a checkpoint keeps its colour as runs come in and
    the unevenly spaced early steps stay distinguishable."""
    from matplotlib.colors import to_hex

    return to_hex(STEP_CMAP(0.02 + 0.83 * CKPTS.index(name) / (len(CKPTS) - 1)))


def _yerr(v: float, c: list[float]) -> list[list[float]]:
    lo, hi = c
    return [
        [max(0.0, v - lo) if np.isfinite(lo) else 0.0],
        [max(0.0, hi - v) if np.isfinite(hi) else 0.0],
    ]


def _empty(ax, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=st.FS_TICK,
        color=st.REF,
    )


def series(ax, xs, ys, cis, sty, off, band: bool) -> None:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size == 0:
        return
    ax.plot(xs, ys, color=sty["color"], ls=sty["ls"], lw=1.0, zorder=2)
    if band and xs.size > 1:
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        ax.fill_between(xs, lo, hi, color=sty["color"], alpha=st.BAND_ALPHA, lw=0, zorder=1)
    for x, y, c, o in zip(xs, ys, cis, off, strict=True):
        if not band:
            ax.errorbar(
                [x],
                [y],
                yerr=_yerr(y, c),
                fmt="none",
                ecolor=sty["color"],
                elinewidth=0.7,
                capsize=1.6,
                zorder=2,
            )
        ax.plot(
            [x],
            [y],
            marker=sty["marker"],
            ms=3.4,
            color=sty["color"],
            mfc="white" if o else sty["color"],
            mew=0.8,
            ls="none",
            zorder=3,
        )


def make_figure(summary: dict, conv: str = "norm", rule: str = "mdl"):
    res = summary.get("checkpoints", {})
    names = [c for c in CKPTS if c in res]
    fig = st.new_figure(st.FULL_W, 4.7)
    axs = fig.subplots(2, 2)
    ax_a, ax_b, ax_c, ax_d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    any_off = any(res[c].get("off_released_line") for c in names)

    # (a) likelihood
    src = "train+test"
    pts = [c for c in names if src in res[c].get("L0", {})] or []
    if not pts:
        src = "test"
        pts = [c for c in names if src in res[c].get("L0", {})]
    n_prompts = res[pts[0]]["L0"][src]["total"]["examples"] if pts else 0
    for seg in ("total", "think", "answer"):
        series(
            ax_a,
            [res[c]["step"] for c in pts],
            [res[c]["L0"][src][seg]["nats_per_tok"] for c in pts],
            [res[c]["L0"][src][seg]["ci"] for c in pts],
            SEG_STYLE[seg],
            [res[c].get("off_released_line") for c in pts],
            band=True,
        )
    if not pts:
        _empty(ax_a, "no step-0 results yet")
    st.panel_title(ax_a, "a", f"$L_0$ of s2300's responses ({n_prompts} prompts)")
    ax_a.set_ylabel("nats per token")

    # (b) EDL*
    lab = {"norm": "length-normalised", "sum": "per-example sums"}[conv]
    any_b = False
    for seg in ("total", "think", "answer"):
        ck = [c for c in names if seg in res[c].get("EDL_star", {}).get(conv, {}).get(rule, {})]
        if ck:
            any_b = True
        e = [res[c]["EDL_star"][conv][rule][seg] for c in ck]
        series(
            ax_b,
            [res[c]["step"] for c in ck],
            [x["per_tok"] for x in e],
            [x["per_tok_ci"] for x in e],
            SEG_STYLE[seg],
            [res[c].get("off_released_line") for c in ck],
            band=False,
        )
    if any_b:
        ax_b.axhline(0, color=st.REF, lw=0.6, ls=":", zorder=0)
        fe = res.get(FINAL, {}).get("EDL_star", {}).get(conv, {}).get(rule, {})
        if fe:  # between the answer line and the thinking / total lines, left of s2300's intervals
            ys = [fe[g]["per_tok"] for g in ("total", "think") if g in fe]
            y = 0.5 * (fe["answer"]["per_tok"] + max(ys)) if "answer" in fe and ys else 0.0
            ax_b.text(
                1700,
                y,
                "s2300 = null control\n(C1 expects EDL* near 0)",
                ha="center",
                va="center",
                fontsize=st.FS_TICK - 1,
                color=st.REF,
            )
    else:
        _empty(ax_b, "no EDL rows yet")
    st.panel_title(ax_b, "b", f"EDL* per token ({lab})")
    ax_b.set_ylabel("nats per token")

    # (c, d) EDL per token over n'
    for ax, seg, letter in ((ax_c, "total", "c"), (ax_d, "answer", "d")):
        drawn = []
        for c in names:
            cv = res[c].get("curve", {}).get(conv, {}).get(seg)
            if not cv or not cv["n"]:
                continue
            col = ckpt_colour(c)
            off = res[c].get("off_released_line")
            x, y = np.asarray(cv["n"], float), np.asarray(cv["per_tok"], float)
            ax.plot(x, y, color=col, lw=1.0, ls="--" if off else "-", zorder=2)
            for xi, yi, ci in zip(x, y, cv["ci"], strict=True):
                ax.errorbar(
                    [xi],
                    [yi],
                    yerr=_yerr(yi, ci),
                    fmt="none",
                    ecolor=col,
                    elinewidth=0.6,
                    capsize=1.2,
                    zorder=2,
                )
            ax.plot(
                x,
                y,
                marker="o",
                ms=3.0,
                color=col,
                mfc="white" if off else col,
                mew=0.7,
                ls="none",
                zorder=3,
            )
            drawn.append(c)
        if drawn:
            ax.axhline(0, color=st.REF, lw=0.6, ls=":", zorder=0)
            ax.set_xscale("log", base=2)
            ns = sorted({int(n) for c in drawn for n in res[c]["curve"][conv][seg]["n"]})
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.minorticks_off()
        else:
            _empty(ax, "no EDL rows yet")
        st.panel_title(ax, letter, f"EDL per token over $n'$, {SEG_STYLE[seg]['label']}")
        ax.set_xlabel("training examples seen, $n'$")
        ax.set_ylabel("nats per token")
        st.light_grid(ax)
        if letter == "d" and drawn:
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

    for ax in (ax_a, ax_b):
        ax.set_xlabel("RL step")
        ax.set_xlim(-80, 2380)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2300])
        ax.set_xticklabels(["0", "500", "1000", "1500", "", "2300"])
        st.light_grid(ax)
    hs = [
        Line2D(
            [],
            [],
            color=SEG_STYLE[s]["color"],
            ls=SEG_STYLE[s]["ls"],
            marker=SEG_STYLE[s]["marker"],
            ms=3.4,
            lw=1.0,
            label=SEG_STYLE[s]["label"],
        )
        for s in ("total", "think", "answer")
    ]
    if any_off:
        hs.append(
            Line2D(
                [],
                [],
                color=st.REF,
                marker="o",
                mfc="white",
                ls="none",
                ms=3.4,
                label="off the released line",
            )
        )
    ax_a.legend(handles=hs, fontsize=st.FS_TICK - 0.5, loc="best")
    return fig


def _step_axis(ax) -> None:
    ax.set_xlabel("RL step")
    ax.set_xlim(-80, 2380)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2300])
    ax.set_xticklabels(["0", "500", "1000", "1500", "", "2300"])
    st.light_grid(ax)


def make_learning_figure(summary: dict):
    """POST HOC readouts of analyze_edl.py's `posthoc` section (see the module docstring)."""
    res = summary.get("checkpoints", {})
    ph = summary.get("posthoc", {})
    fig = st.new_figure(st.FULL_W, 4.9)
    axs = fig.subplots(2, 2)
    curves = ph.get("L_test_curve", {})
    drawn: list[str] = []
    for ax, seg, letter in ((axs[0, 0], "total", "a"), (axs[0, 1], "answer", "b")):
        for c in [x for x in CKPTS if x in curves]:
            cv = curves[c].get(seg)
            if not cv or not cv["n"]:
                continue
            col = ckpt_colour(c)
            off = res.get(c, {}).get("off_released_line")
            # no per-point intervals: the unpaired prompt intervals (in the JSON) are shared by all checkpoints and
            # hide the curves; the paired comparisons are panel (c) and posthoc.delta_L
            x = np.asarray(cv["n"], float)
            y = np.asarray(cv["L"], float)
            ax.plot(x, y, color=col, lw=1.0, ls="--" if off else "-", zorder=2)
            ax.plot(
                x,
                y,
                marker="o",
                ms=2.8,
                color=col,
                mfc="white" if off else col,
                mew=0.7,
                ls="none",
                zorder=3,
            )
            if c not in drawn:
                drawn.append(c)
        l0f = res.get(FINAL, {}).get("L0", {}).get("test", {}).get(seg)
        if l0f:
            ax.axhline(l0f["nats_per_tok"], color=st.REF, lw=0.8, ls=":", zorder=1)
        if not any(seg in curves[c] for c in curves):
            _empty(ax, "no post hoc readouts yet")
        ax.set_xticks([0, 8, 16, 32, 64])
        st.panel_title(ax, letter, f"$L_{{test}}$ over $n'$, {SEG_STYLE[seg]['label']}")
        ax.set_xlabel("training examples seen, $n'$")
        ax.set_ylabel("nats per token")
        st.light_grid(ax)

    ax = axs[1, 0]
    d = ph.get("delta_EDL_star", {})
    any_c = False
    for seg in ("total", "answer"):
        ck = [c for c in CKPTS if c in d and seg in d[c].get("norm", {})]
        if not ck:
            continue
        any_c = True
        e = [d[c]["norm"][seg] for c in ck]
        series(
            ax,
            [
                res[c]["step"] + (-15 if seg == "total" else 15) for c in ck
            ],  # dodge: the intervals overlap
            [x["per_tok"] for x in e],
            [x["per_tok_ci"] for x in e],
            SEG_STYLE[seg],
            [res[c].get("off_released_line") for c in ck],
            band=False,
        )
    ax.axhline(0, color=st.REF, lw=0.6, ls=":", zorder=0)
    if any_c:
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=SEG_STYLE[s]["color"],
                    ls=SEG_STYLE[s]["ls"],
                    marker=SEG_STYLE[s]["marker"],
                    ms=3.2,
                    lw=1.0,
                    label=SEG_STYLE[s]["label"],
                )
                for s in ("total", "answer")
            ],
            fontsize=st.FS_TICK - 0.5,
            loc="upper right",
        )
    else:
        _empty(ax, "no post hoc readouts yet")
    st.panel_title(ax, "c", r"$\Delta$EDL* = EDL*$(t)$ $-$ EDL*(s2300), per token")
    ax.set_ylabel("nats per token")
    _step_axis(ax)

    ax = axs[1, 1]
    rt = ph.get("reach_transfer", {})
    any_d = False
    for seg in ("total", "answer"):
        sty = SEG_STYLE[seg]
        for key, hollow, ls, dx in (("test", False, sty["ls"], -12), ("reach", True, ":", 12)):
            ck = [c for c in CKPTS if c in rt and f"gap_share_{key}" in rt[c].get(seg, {})]
            if not ck:
                continue
            any_d = True
            xs = np.array([res[c]["step"] + dx for c in ck], float)
            ys = np.array([rt[c][seg][f"gap_share_{key}"] for c in ck])
            cis = [rt[c][seg][f"gap_share_{key}_ci"] for c in ck]
            ax.plot(xs, ys, color=sty["color"], ls=ls, lw=1.0, zorder=2)
            for x, y, cc in zip(xs, ys, cis, strict=True):
                ax.errorbar(
                    [x],
                    [y],
                    yerr=_yerr(y, cc),
                    fmt="none",
                    ecolor=sty["color"],
                    elinewidth=0.6,
                    capsize=1.2,
                    zorder=2,
                )
            ax.plot(
                xs,
                ys,
                marker=sty["marker"],
                ms=3.0,
                color=sty["color"],
                mfc="white" if hollow else sty["color"],
                mew=0.8,
                ls="none",
                zorder=3,
            )
    ax.axhline(1, color=st.REF, lw=0.6, ls=":", zorder=0)
    ax.axhline(0, color=st.REF, lw=0.6, ls=":", zorder=0)
    if any_d:
        ax.legend(
            handles=[
                Line2D(
                    [], [], color=st.INK, marker="o", ms=3.0, ls="-", lw=1.0, label="test prompts"
                ),
                Line2D(
                    [],
                    [],
                    color=st.INK,
                    marker="o",
                    mfc="white",
                    ms=3.0,
                    ls=":",
                    lw=1.0,
                    label="reach prompts",
                ),
            ],
            fontsize=st.FS_TICK - 0.5,
            loc="lower left",
        )
    else:
        _empty(ax, "no post hoc readouts yet")
    st.panel_title(ax, "d", "gap to s2300 closed by 64 examples")
    ax.set_ylabel("share of the $L_0$ gap")
    _step_axis(ax)
    if drawn:
        hs = [
            Line2D(
                [],
                [],
                color=ckpt_colour(c),
                marker="o",
                ms=2.8,
                lw=1.0,
                mfc="white" if res.get(c, {}).get("off_released_line") else ckpt_colour(c),
                ls="--" if res.get(c, {}).get("off_released_line") else "-",
                label=f"{c} ({res.get(c, {}).get('step', '')})",
            )
            for c in drawn
        ]
        fig.legend(
            handles=hs,
            fontsize=st.FS_TICK,
            loc="outside lower center",
            title="(a, b) checkpoint (RL step); dotted line: s2300's $L_0$ on the test prompts. All panels are POST HOC readouts",
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
    ap.add_argument("--conv", choices=["norm", "sum"], default="norm")
    ap.add_argument("--rule", choices=["mdl", "min"], default="mdl")
    ap.add_argument(
        "--learning-out",
        type=Path,
        default=OUT_LEARNING,
        help="the post hoc learning figure (written when the summary has a posthoc section)",
    )
    a = ap.parse_args()
    summary = json.loads(a.summary.read_text())
    with st.paper_style():
        fig = make_figure(summary, a.conv, a.rule)
        paths = st.save(fig, a.out)
        st.close(fig)
        if summary.get("posthoc", {}).get("L_test_curve"):
            fig = make_learning_figure(summary)
            paths += st.save(fig, a.learning_out)
            st.close(fig)
    print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
