"""Figure 5, the performance control (Leon's): do the habits and the note lines cost the prover theorems?

Untagged pass@1 (the Lean-accepted share of 8 attempts per theorem, 500 marker_plain theorems, step limit 51) of every
base, with 95 % bootstrap intervals over theorems: the first bases and the Nov 14 bases without notes from the gate
(LH/gate/gate_v2.json), second bases from LH/read/summary_extra.json when present (else marked pending). Hollow grey
marks: pass@8. The printed contrasts are the gate's pre-registered flags (paired over theorems) and, from
summary_extra.json, second base minus first base.

    PY fig_perf.py [--lh LH] [--gate GATE_JSON] [--out DIR]
"""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from figcommon import CORPUS_STYLE, NAME, S, est, fmt, load_extra, load_json, parser, save

ROWS = [
    ("control_half_s0", "control_half", None),
    ("authors_half_s0", "authors_half", None),
    ("c0n_s0", "c0n", None),
    ("c0_s0", "c0", "c0"),
    ("c9_s0", "c9", "c9"),
    ("c24_s0", "c24", "c24"),
    ("c24_s1", "c24", "c24"),
    ("c4_s0", "c4", "c4"),
    ("c4_s1", "c4", "c4"),
]


def main(argv: list[str] | None = None) -> list[Path]:
    ap = parser(__doc__)
    ap.add_argument("--gate", default=None)
    a = ap.parse_args(argv)
    lh = Path(a.lh)
    g = load_json(Path(a.gate) if a.gate else lh / "gate" / "gate_v2.json")
    gp = g["bases"]["marker_plain"]
    extra = load_extra(lh)
    lev = (extra.get("bases") or {}).get("levels") or {}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.5)
        ax = fig.add_subplot(1, 1, 1)
        n = len(ROWS)
        labels = []
        for i, (b, nm, c) in enumerate(ROWS):
            y = n - 1 - i
            second = b.endswith("_s1")
            labels.append(NAME[nm] + (", 2nd base" if second else ""))
            perf: dict[str, Any] = (gp.get(b) or {}).get("perf") or lev.get(b) or {}
            p1, p8 = est(perf.get("pass1")), est(perf.get("pass8"))
            st = CORPUS_STYLE.get(c or "", {"color": S.REF, "marker": "o"})
            if p1 is None:
                ax.text(
                    30.5,
                    y,
                    "pending",
                    va="center",
                    fontsize=S.FS_TICK - 0.5,
                    color=S.REF,
                    style="italic",
                )
                continue
            ax.errorbar(p1[0], y, xerr=[[p1[0] - p1[1]], [p1[2] - p1[0]]], fmt="none", ecolor=st["color"],
                        elinewidth=0.9, capsize=1.6)  # fmt: skip
            ax.plot(p1[0], y, linestyle="none", marker=st["marker"], markersize=4.4, markeredgewidth=0.8,
                    markerfacecolor=S.WHITE if second else st["color"],
                    markeredgecolor=st["color"] if second else S.WHITE)  # fmt: skip
            ax.annotate(f"{p1[0]:.1f} %", (p1[1], y), xytext=(-4, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=S.FS_TICK - 1, color=S.INK2)  # fmt: skip
            if p8 is not None:
                ax.errorbar(p8[0], y, xerr=[[p8[0] - p8[1]], [p8[2] - p8[0]]], fmt="none", ecolor=S.REF,
                            elinewidth=0.6, capsize=1.2)  # fmt: skip
                ax.plot(p8[0], y, linestyle="none", marker=st["marker"], markersize=3.6, markeredgewidth=0.7,
                        markerfacecolor=S.WHITE, markeredgecolor=S.REF)  # fmt: skip
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(reversed(labels)))
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(25, 100)
        ax.set_xlabel("untagged pass@1 on marker_plain (%), filled; pass@8, hollow grey")
        S.light_grid(ax, "x")
        flags = g["perf_flag"]["marker_plain"]
        sm = (extra.get("bases") or {}).get("second_minus_first") or {}
        print("pass@1 differences (points, paired over theorems):")
        for k in ("c0_s0-c0n_s0", "c0_s0-control_half_s0", "c24_s0-authors_half_s0"):
            print(f"  {k}: {fmt((flags.get(k) or {}).get('pass1_diff'))}")
        for c, v in sm.items():
            print(f"  {c} second base - first base: {fmt(v.get('pass1'))}")
        return save(fig, Path(a.out), "fig5_perf")


if __name__ == "__main__":
    main()
