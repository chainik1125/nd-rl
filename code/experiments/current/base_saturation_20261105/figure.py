"""base_saturation figure: pass@k curves on Charles149 (seed 0, k1024 reads), from `table.json`.

    python -m code.experiments.current.base_saturation_20261105.figure

Panel (a): the pretraining-only arms (base, A, the CPT dose B0.25 / CPT / B4, C). Panel (b): RL
from each start (base -> RL, CPT -> D1, C -> D2). D1 and D2 share their start's hue, dashed with
open square markers; the paper's RL keeps its usual vermillion. Writes passk.{pdf,svg,png} next
to this file, in `code/ndbench/figures/style.py`'s style.
"""

from __future__ import annotations

import json
from pathlib import Path

from code.ndbench.figures import style as S

HERE = Path(__file__).resolve().parent
OI = S.OKABE_ITO
GREEN = OI["bluish_green"]
STY = {
    "base": S.ModelStyle("base", OI["blue"], "o"),
    "A": S.ModelStyle("A (anneal)", OI["orange"], "D"),
    "B0.25": S.ModelStyle("B0.25 (CPT 0.25×)", S.shade(GREEN, 0.45), "v"),
    "CPT": S.ModelStyle("CPT (1×)", GREEN, "^"),
    "B4": S.ModelStyle("B4 (CPT 4×)", S.shade(GREEN, -0.45), "P"),
    "C": S.ModelStyle("C (Stage 1 2×, to 0)", OI["reddish_purple"], "X"),
    "RL": S.ModelStyle("RL (from base)", OI["vermillion"], "s"),
    "D1": S.ModelStyle("D1 (RL from CPT)", GREEN, "s", linestyle=(0, (4, 1.6)), filled=False),
    "D2": S.ModelStyle(
        "D2 (RL from C)", OI["reddish_purple"], "s", linestyle=(0, (4, 1.6)), filled=False
    ),
}
PANELS = [
    ("a", "pretraining only", ["base", "A", "B0.25", "CPT", "B4", "C"]),
    ("b", "RL from each start", ["base", "RL", "CPT", "D1", "C", "D2"]),
]


def main() -> int:
    t = json.loads((HERE / "table.json").read_text())
    curve = {m: {int(k): v for k, v in c.items()} for m, c in t["curve"].items()}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35)
        axes = fig.subplots(1, 2, sharey=True)
        for ax, (letter, title, models) in zip(axes, PANELS, strict=True):
            for m in models:
                if m not in curve:
                    continue
                ks = sorted(curve[m])
                st = STY[m]
                ax.plot(ks, [curve[m][k] / 100 for k in ks], **st.line_kw())  # % -> fraction
            S.log2_k_axis(ax, 1024)
            S.pct_axis(ax)
            S.light_grid(ax)
            ax.set_xlabel("k (attempts per problem)")
            S.panel_title(ax, letter, f"Charles149, {title}")
            ax.legend(loc="lower right")
        axes[0].set_ylabel("pass@k (% of 149)")
        S.save(fig, HERE / "passk")
        S.close(fig)
    print(f"wrote {HERE / 'passk'}.{{pdf,svg,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
