"""Old models (2026-10-07 runs, pre-search Stage 1): GRPO with thinking vs the EI no-think baseline.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.old_grpo_vs_ei

The meeting version of `fig_rl_matched` (Leon, 2026-10-09): only the comparison he asked about,
GRPO-trained thinking (effort levels, read at B 256, seeds 0 and 1) against the EI-trained no-think
baseline (seed 0), at matched training attempts (EI round r <-> GRPO iteration 39 r). Colours follow
`fig_tts_pooled`: blue is thinking, orange is the EI no-think baseline. Values and intervals are the
ones `figs.fig_matched` and `final_tables.py` print (same rows, same 4,000-draw theorem resample).
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from code.experiments.current.thinking_traces_20261005.matched_eval import figs as F
from code.ndbench.figures import style as S

STEPS = [("r1", "it39"), ("r2", "it78")]
# (look, dir tag, 0 = EI / 1 = GRPO step, budget, x offset)
SERIES = [
    (F.Look("EI, no thinking (baseline)", F.ORANGE, "s", F.DASH), "eiT0", 0, "B0", -0.07),
    (F.Look("GRPO + thinking, seed 0", F.BLUE, "o"), "grpoT1v1e", 1, "B256", 0.0),
    (
        replace(F.Look("GRPO + thinking, seed 1", F.BLUE, "o"), hollow=True),
        "grpoT1v1es1",
        1,
        "B256",
        0.07,
    ),
]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    a = ap.parse_args(argv)
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sets = [("h12", F.H12, F.Rows(None, tmp)), ("t1320", F.T1320, F.Rows(F.valid_1320(), tmp))]
        with S.paper_style():
            fig = S.new_figure(S.FULL_W, 4.0, layout="constrained")
            axs = fig.subplots(2, 2, sharex=True)
            for r, (s, title, rows) in enumerate(sets):
                for c, k in enumerate((1, 5)):
                    ax = axs[r][c]
                    for look, tag, grpo, b, dx in SERIES:
                        es = [
                            F.est(
                                rows.arm(F.M / f"{s}_{tag}_{st[grpo]}" / f"rows_plain_{b}.jsonl"), k
                            )
                            for st in STEPS
                        ]
                        for st, e in zip(STEPS, es, strict=True):
                            if e is not None:
                                print(
                                    f"{s:6s} pass@{k} {look.label:28s} {st[grpo]:5s} "
                                    f"{100 * e[0]:5.1f} [{100 * e[1]:5.1f}, {100 * e[2]:5.1f}]"
                                )
                        F.points(ax, [i + dx for i in range(2)], es, look)
                    S.panel_title(ax, "abcd"[2 * r + c], title)
                    ax.set_ylabel(f"pass@{k}")
                    ax.set_xlim(-0.35, 1.35)
                    ax.set_xticks(
                        [0, 1],
                        ["1 round /\n39 iterations", "2 rounds /\n78 iterations"],
                    )
                    F.finish(ax)
            fig.supxlabel(
                "training: EI rounds / GRPO iterations (1 round = 39 iterations = 160k attempts)",
                fontsize=S.FS_LABEL,
            )
            handles = [look.handle() for look, *_ in SERIES]
            fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
            S.save(fig, out / "fig_old_grpo_vs_ei")
            S.close(fig)
    print(f"wrote {out / 'fig_old_grpo_vs_ei'}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
