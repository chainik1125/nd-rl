"""Figure 4, on-policy RL: a learned reward model for A-form reiteration plus GRPO with a KL anchor (the "reward line"),
against the same GRPO with acceptance only (the "plain line"), each against its base.

(a), (b): each line's A-share minus the base's, per habit, at the line's last measured iteration (40), untagged,
marker_plain, Lean-accepted proofs, 95 % bootstrap intervals over theorems (LH/read/summary_extra.json, computed with
the read-out's estimators; a class under 50 sites has no interval and a small mark). (a) the all-linked corpus, (b) the
4-linked corpus, where the notes are unlinked coins. Filled: RL seed 0; hollow: RL seed 1 when present (PREREG
Amendment 10). (c) The GRPO training samples of the all-linked corpus (Lean-accepted proofs with notes): the share whose
notes are all in author A's form, per iteration (40 chunks of the sample file in order).

    PY fig_grpo.py [--lh LH] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from figcommon import NAME, OI, S, est, load_extra, parser, save

LINE_STYLE = {
    "reward": {
        "color": S.shade(OI["reddish_purple"], -0.15),
        "marker": "^",
        "label": "reward-model line",
    },
    "plain": {
        "color": S.shade(OI["orange"], -0.3),
        "marker": "v",
        "label": "plain line (acceptance only)",
    },
}
SHAPE = {
    "atom": "P", "~a": "¬P", "~c": "¬(…)",
    "&aa": "P∧Q", "&ac": "P∧(…)", "&ca": "(…)∧P", "&cc": "(…)∧(…)",
    "vaa": "P∨Q", "vac": "P∨(…)", "vca": "(…)∨P", "vcc": "(…)∨(…)",
    ">aa": "P→Q", ">ac": "P→(…)", ">ca": "(…)→P", ">cc": "(…)→(…)",
}  # fmt: skip
ROWS = [("reit", "reiteration (rated)"), ("order", "step order"), ("restate", "restate"),
        ("_notes", "all notes, pooled")] + [(f"note_{k}", f"note on {v}") for k, v in SHAPE.items()]  # fmt: skip


def last_points(gv: dict[str, Any], corpus: str) -> dict[tuple[str, int], dict[str, Any]]:
    """(line, seed) -> the entry of the line's last measured iteration."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for e in gv.values():
        if e["corpus"] != corpus:
            continue
        k = (e["line"], int(e["seed"]))
        if k not in out or e["iteration"] > out[k]["iteration"]:
            out[k] = e
    return out


def value(e: dict[str, Any], h: str) -> tuple[tuple[float, float, float] | None, bool]:
    """((point, lo, hi) or None, thin)."""
    if h == "_notes":
        return est(e["notes_pooled"]["vs_base"]), False
    v = est((e.get("vs_base") or {}).get(h))
    if v is not None:
        return v, False
    a, b = (e.get("share") or {}).get(h), (e.get("base_share") or {}).get(h)
    if a is None or b is None:
        return None, True
    return (a - b, a - b, a - b), True


def panel(ax: Any, gv: dict[str, Any], corpus: str) -> list[int]:
    pts = last_points(gv, corpus)
    n = len(ROWS)
    its = []
    for (line, seed), e in sorted(pts.items()):
        st = LINE_STYLE[line]
        its.append(e["iteration"])
        dy = (0.17 if line == "reward" else -0.17) + (0.08 if seed else 0.0)
        for i, (h, _lab) in enumerate(ROWS):
            v, thin = value(e, h)
            if v is None:
                continue
            y = n - 1 - i + dy
            if not thin:
                ax.errorbar(v[0], y, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=st["color"],
                            elinewidth=0.7, capsize=1.0)  # fmt: skip
            ax.plot(v[0], y, linestyle="none", marker=st["marker"], markersize=2.6 if thin else 3.6,
                    markerfacecolor=S.WHITE if seed else st["color"], markeredgecolor=st["color"] if seed else S.WHITE,
                    markeredgewidth=0.6)  # fmt: skip
    ax.axvline(0, color=S.REF, linewidth=0.6)
    ax.axhline(n - 4.5, color=S.GRID, linewidth=0.6)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-0.7, n - 0.3)
    S.light_grid(ax, "x")
    return sorted(set(its))


def main(argv: list[str] | None = None) -> list[Path]:
    a = parser(__doc__).parse_args(argv)
    extra = load_extra(Path(a.lh))
    gv = extra.get("grpo_vs_base") or {}
    if not gv:
        raise SystemExit(
            "LH/read/summary_extra.json has no grpo_vs_base: run summary_extra.py first"
        )
    from matplotlib.lines import Line2D

    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.35)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2])
        for ax, c, letter in ((ax1, "c24", "a"), (ax2, "c4", "b")):
            its = panel(ax, gv, c)
            S.panel_title(
                ax, letter, f"{NAME[c]}, iteration {'/'.join(map(str, its))}", size=S.FS_TICK + 0.5
            )
            ax.set_xlabel("A-share change vs. base (points)")
        ax1.set_yticks(range(len(ROWS)))
        ax1.set_yticklabels([lab for _, lab in reversed(ROWS)], fontsize=S.FS_TICK - 0.5)
        ax2.tick_params(labelleft=False)
        mix = extra.get("mixture") or {}
        for run, m in sorted(mix.items()):
            line = "reward" if "_rlhf" in run else "plain"
            seed = int(run.rsplit("_s", 1)[1])
            st = LINE_STYLE[line]
            ys = [100 * ch["all_A"] if ch.get("all_A") is not None else None for ch in m["chunks"]]
            xs = [i + 1 for i in range(len(ys))]
            ax3.plot(
                xs,
                ys,
                color=st["color"],
                linewidth=1.1,
                linestyle="-" if seed == 0 else (0, (3, 1.6)),
            )
            if seed == 0:
                ax3.text(39, 78 if line == "reward" else 33, st["label"].split(" (")[0], ha="right",
                         fontsize=S.FS_TICK - 1, color=st["color"])  # fmt: skip
        ax3.set_xlim(0, 41)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("GRPO iteration")
        ax3.set_ylabel("sampled proofs with all notes A's (%)")
        S.panel_title(ax3, "c", "samples, all linked", size=S.FS_TICK + 0.5)
        S.light_grid(ax3, "y")
        hs = [Line2D([], [], linestyle="none", marker=v["marker"], markersize=5.5, markerfacecolor=v["color"],
                     markeredgecolor=S.WHITE, label=v["label"]) for v in LINE_STYLE.values()]  # fmt: skip
        if any(int(e["seed"]) for e in gv.values()):
            hs.append(Line2D([], [], linestyle="none", marker="o", markersize=5.5, markerfacecolor=S.WHITE,
                             markeredgecolor=S.INK, label="hollow / dashed: RL seed 1"))  # fmt: skip
        fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=S.FS_TICK - 0.5)
        return save(fig, Path(a.out), "fig4_grpo")


if __name__ == "__main__":
    main()
