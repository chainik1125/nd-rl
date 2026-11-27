"""Figure 1, the stage-1 gate: does a base write a whole proof as one author when no tag tells it which?

For each first base (seed 0), untagged, pooled sets (marker_plain and route_unseen), from LH/gate/gate_v2.json: the
G2 link, i.e. the A-share of later sites of OTHER habits when the proof's first site is in author A's form minus when it
is in B's form, in points, with the gate's 95 % bootstrap interval over theorems. Three links: a note to the later notes
of other premise classes (the pre-registered gate, bar +30 on the all-linked base), a note to the natural habits, and a
natural habit to the other natural habits.

    PY fig_gate.py [--lh LH] [--gate LH/gate/gate_v2.json] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from figcommon import NAME, S, est, load_json, parser, save

BASES = ["c24", "c9", "c4", "c0"]
LINKS = [
    ("notes|notes", "a note → later notes of other classes (the gate)", S.INK, "o", True),
    ("notes|natural", "a note → the natural habits", S.INK2, "s", False),
    ("natural|natural", "a natural habit → other natural habits", S.REF, "^", False),
]
BAR = 30.0


def main(argv: list[str] | None = None) -> list[Path]:
    ap = parser(__doc__)
    ap.add_argument("--gate", default=None)
    a = ap.parse_args(argv)
    g = load_json(Path(a.gate) if a.gate else Path(a.lh) / "gate" / "gate_v2.json")
    pooled = g["bases"]["pooled"]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 1.9)
        ax = fig.add_subplot(1, 1, 1)
        n = len(BASES)
        for j, (key, label, color, marker, filled) in enumerate(LINKS):
            for i, b in enumerate(BASES):
                v = est(((pooled.get(f"{b}_s0") or {}).get("G2") or {}).get(key))
                if v is None:
                    continue
                y = n - 1 - i + (1 - j) * 0.22
                ax.errorbar(v[0], y, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=color,
                            elinewidth=0.8, capsize=1.6)  # fmt: skip
                ax.plot(v[0], y, linestyle="none", marker=marker, markersize=4.2, markeredgewidth=0.8,
                        markerfacecolor=color if filled else S.WHITE, markeredgecolor=color,
                        label=label if i == 0 else None)  # fmt: skip
                if key == "notes|notes" or b == "c24":
                    ax.annotate(f"{v[0]:+.0f}", (v[2], y), xytext=(3, 0), textcoords="offset points",
                                va="center", fontsize=S.FS_TICK - 0.5, color=S.INK2)  # fmt: skip
        ax.axvline(0, color=S.REF, linewidth=0.6)
        ax.axvline(BAR, color=S.REF, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.text(
            BAR,
            n - 0.45,
            " pre-registered bar (+30)",
            fontsize=S.FS_TICK - 0.5,
            color=S.REF,
            va="bottom",
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels([NAME[b] for b in reversed(BASES)])
        ax.set_ylim(-0.6, n - 0.2)
        ax.set_xlim(-15, 110)
        ax.set_xlabel(
            "link between habits without a tag (points; A-share of later sites after an A first site minus after a B one)"
        )
        S.light_grid(ax, "x")
        ax.legend(loc="lower right", fontsize=S.FS_TICK - 0.5, handletextpad=0.3)
        return save(fig, Path(a.out), "fig1_gate")


if __name__ == "__main__":
    main()
