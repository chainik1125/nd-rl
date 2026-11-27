"""Figure 3, what the tuning does to the prover: Lean-accepted share of attempts against training step.

Untagged, marker_plain, k 8, step limit 51, from the measurement summaries only:
  - the base: LH/measure/<c>_s<k>/marker_plain__none.summary.json (`lh_totals.accepted_pct`);
  - a rated run: its probe (LH/arms/<run>/probe/probe.json; steps 5, 10, ... up to the first snapshot past +30, and
    the last, step 400) and its full measurements (LH/arms/<run>/measure/step<N>/...summary.json);
  - its no-preference control (DPO with the members swapped at random; fine-tune on the sampled members of the same
    pairs): full measurements only, at the rated run's bracketing steps and step 400.
Panel (a) DPO (both trainer seeds; hollow markers: a second base), panel (b) the fine-tune arm. A short vertical tick
marks each rated run's +30 crossing (the read-out's interpolated step). The x axis is broken between 40 and 400.
Point estimates: each point is 4,000 attempts (500 theorems x 8).

    PY fig_accept.py [--lh LH] [--read READ_JSON] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from figcommon import CORPUS_STYLE, NAME, S, load_read, parser, save

BREAK_AT, LAST_X = 45.0, 52.0  # step 400 is drawn at x = 52


def xpos(step: float) -> float:
    return LAST_X if step > 100 else float(step)


def _acc(p: Path) -> float | None:
    try:
        return float(json.loads(p.read_text())["lh_totals"]["accepted_pct"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def curve(lh: Path, run: str) -> dict[int, float]:
    """step -> accepted % for a tuned run: full measurements first, the probe where there is none."""
    out: dict[int, float] = {}
    d = lh / "arms" / run
    pj = d / "probe" / "probe.json"
    if pj.exists():
        for lab, v in json.loads(pj.read_text()).get("snapshots", {}).items():
            m = re.fullmatch(r"step(\d+)", lab)
            if m and v.get("accepted_pct") is not None:
                out[int(m.group(1))] = float(v["accepted_pct"])
    for sd in sorted((d / "measure").glob("step*")):
        m = re.fullmatch(r"step(\d+)", sd.name)
        a = _acc(sd / "marker_plain__none.summary.json")
        if m and a is not None:
            out[int(m.group(1))] = a
    return out


def base_acc(lh: Path, corpus_token: str) -> float | None:
    c, bs = (corpus_token[:-2], 1) if corpus_token.endswith("b1") else (corpus_token, 0)
    return _acc(lh / "measure" / f"{c}_s{bs}" / "marker_plain__none.summary.json")


def crossings(res: dict[str, Any]) -> dict[str, float]:
    out = {}
    for v in (res.get("passes") or {}).values():
        for run, r in (v.get("runs") or {}).items():
            if r.get("step") is not None:
                out[run] = float(r["step"])
    return out


def runs(lh: Path, kind: str) -> list[tuple[str, str, str, int]]:
    """(rated run, control run, corpus token, trainer seed) present on disk."""
    pre, ctl = ("dpo", "dposwap") if kind == "dpo" else ("sft", "sftsam")
    out = []
    for d in sorted((lh / "arms").glob(f"{pre}_*_reit_s*")):
        m = re.fullmatch(rf"{pre}_(c\w+?)_reit_s(\d+)", d.name)
        if m:
            out.append(
                (d.name, f"{ctl}_{m.group(1)}_reit_s{m.group(2)}", m.group(1), int(m.group(2)))
            )
    return out


def draw(ax: Any, lh: Path, kind: str, cross: dict[str, float]) -> None:
    for rated, ctl, tok, _seed in runs(lh, kind):
        c = tok[:-2] if tok.endswith("b1") else tok
        st = CORPUS_STYLE[c]
        hollow = tok.endswith("b1")
        b = base_acc(lh, tok)
        for run, ls in ((rated, "-"), (ctl, (0, (3, 1.6)))):
            cv = curve(lh, run)
            if b is not None:
                cv = {0: b, **cv}
            if len(cv) < 2:
                continue
            xs = sorted(cv)
            near = [s for s in xs if s <= 100]
            far = [s for s in xs if s > 100]
            mk = {"marker": st["marker"], "markersize": 3.0, "markerfacecolor": S.WHITE if hollow else st["color"],
                  "markeredgecolor": st["color"] if hollow else S.WHITE, "markeredgewidth": 0.6}  # fmt: skip
            alpha = 0.95 if run == rated else 0.75
            ax.plot(
                near,
                [cv[s] for s in near],
                linestyle=ls,
                color=st["color"],
                linewidth=0.9,
                alpha=alpha,
                **mk,
            )
            if (
                near and far
            ):  # no measurement between the last early step and step 400: a thin dotted joint
                ax.plot([xpos(near[-1]), xpos(far[0])], [cv[near[-1]], cv[far[0]]], linestyle=(0, (1, 1.5)),
                        color=st["color"], linewidth=0.6, alpha=alpha)  # fmt: skip
                ax.plot(
                    [xpos(far[0])],
                    [cv[far[0]]],
                    linestyle="none",
                    color=st["color"],
                    alpha=alpha,
                    **mk,
                )
            if run == rated and rated in cross:
                s0 = cross[rated]
                lo = max(s for s in xs if s <= s0)
                hi = min(s for s in xs if s >= s0)
                yv = cv[lo] if hi == lo else cv[lo] + (cv[hi] - cv[lo]) * (s0 - lo) / (hi - lo)
                ax.plot(
                    [s0, s0],
                    [yv - 3.5, yv + 3.5],
                    color=st["color"],
                    linewidth=1.3,
                    solid_capstyle="butt",
                )


def main(argv: list[str] | None = None) -> list[Path]:
    a = parser(__doc__).parse_args(argv)
    res, src = load_read(a)
    print(f"read-out: {src}")
    lh = Path(a.lh)
    cross = crossings(res)
    from matplotlib.lines import Line2D

    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.45)
        axs = fig.subplots(1, 2, sharey=True)
        for ax, kind, letter, title in (
            (axs[0], "dpo", "a", "preference tuning (DPO), rated vs. swapped control"),
            (axs[1], "sft", "b", "fine-tune, preferred vs. sampled members"),
        ):
            draw(ax, lh, kind, cross)
            S.panel_title(ax, letter, title)
            ax.set_xlim(-2, LAST_X + 3)
            ax.set_xticks([0, 10, 20, 30, 40, LAST_X])
            ax.set_xticklabels(["base", "10", "20", "30", "40", "400"])
            ax.axvspan(BREAK_AT - 1.2, BREAK_AT + 1.2, color=S.SHADE, linewidth=0)
            ax.text(BREAK_AT, 1.5, "//", ha="center", fontsize=S.FS_TICK, color=S.REF)
            ax.set_xlabel("training step (64 pairs or proofs per step)")
            ax.set_ylim(0, 80)
            S.light_grid(ax, "y")
        axs[0].set_ylabel("Lean-accepted attempts (%)")
        hs = [Line2D([], [], color=CORPUS_STYLE[c]["color"], marker=CORPUS_STYLE[c]["marker"], markersize=3.0,
                     linewidth=0.9, markeredgecolor=S.WHITE, markeredgewidth=0.5, label=NAME[c])
              for c in ("c24", "c9", "c4", "c0")]  # fmt: skip
        hs += [Line2D([], [], color=S.INK, linewidth=0.9, label="rated run"),
               Line2D([], [], color=S.INK, linewidth=0.9, linestyle=(0, (3, 1.6)), label="no-preference control"),
               Line2D([], [], color=S.INK, linewidth=1.3, marker="|", markersize=6, linestyle="none",
                      label="+30 crossing")]  # fmt: skip
        fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=S.FS_TICK - 0.5)
        return save(fig, Path(a.out), "fig3_accept")


if __name__ == "__main__":
    main()
