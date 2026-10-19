"""Proof-length distributions of every training stage and test set (Leon, 2026-10-09).

    ND_OFFLINE=1 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.length_dists

One panel per set, in training order, then the test sets: % of the set at each proof length.
Length measures differ by source and are named in each panel: K12 and rl_hard12 store the
generator's proof length, the search proofs their own length, hard712 and the 13-20 and 21-30 tests
the minimal length, the hard12 test and textbook_dev a stored reference length (textbook_train's
14 problems store none). Proofs longer than 30 lines go in the last bin.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import valid_1320
from code.ndbench.figures import style as S

R = Path("/scratch/cowifornia/nd-rl/think/runs")
K12 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
SEARCH = Path("/scratch/cowifornia/nd-rl/think/traces/search/full/plain_e256.jsonl")
TB = Path("/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72")
BLUE, ORANGE = "#2a78d6", "#eb6834"
MAX = 30  # last bin: 30+
Row = dict[str, Any]


def rows(path: Path) -> Iterable[Row]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def lengths(path: Path, key: str, keep: Callable[[Row], bool] = lambda r: True) -> list[int]:
    return [int(r[key]) for r in rows(path) if keep(r) and r.get(key) is not None]


def sets() -> list[tuple[str, str, list[int], bool]]:
    """(label, length measure, lengths, is a training set)."""
    v1320 = valid_1320()
    return [
        ("pretraining theorems (K12)", "generator proof", lengths(K12, "n_lines"), True),
        (
            "pretraining proofs the model learns from",
            "search's own proof",
            lengths(SEARCH, "n_lines"),
            True,
        ),
        (
            "RL stage 1: rl_hard12",
            "generator proof",
            lengths(R / "pools/rl_hard12.jsonl", "n_lines"),
            True,
        ),
        (
            "RL stage 2: hard712 sample a",
            "minimal",
            lengths(R / "pools/hard712/pool_s5k_a.jsonl", "min_lines"),
            True,
        ),
        (
            "RL stage 3: hard712 sample b",
            "minimal",
            lengths(R / "pools/hard712/pool_s5k_b.jsonl", "min_lines"),
            True,
        ),
        (
            "test: hard12 (in distribution)",
            "reference",
            lengths(R / "restart2_hard12/hard12_test_1k.jsonl", "reference_lines"),
            False,
        ),
        (
            "test: textbook72 (58 of the 72 store a length)",
            "reference",
            lengths(TB / "textbook_dev.jsonl", "reference_lines"),
            False,
        ),
        (
            "test: 13–20 lines",
            "minimal",
            lengths(
                R / "eval/matched/sets/t1320_n1000.jsonl",
                "reference_lines",
                lambda r: r["name"] in v1320,
            ),
            False,
        ),
        (
            "test: 21–30 lines",
            "minimal",
            lengths(R / "pools/test_21_30/test_21_30.jsonl", "reference_lines"),
            False,
        ),
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    a = ap.parse_args(argv)
    data = sets()
    xs = np.arange(1, MAX + 1)
    dump = []
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 0.62 * len(data) + 0.5, layout="constrained")
        axs = fig.subplots(len(data), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (label, measure, ls, train) in zip(axs, data, strict=True):
            c = Counter(min(x, MAX) for x in ls)
            n = len(ls)
            pct = np.array([100 * c.get(int(x), 0) / n for x in xs])
            ax.bar(xs, pct, width=0.8, color=BLUE if train else ORANGE, lw=0)
            ax.axvline(12.5, color=S.INK, lw=0.5, ls=(0, (2, 2)), zorder=0)
            ax.set_ylim(0, max(pct) * 1.75)
            ax.set_yticks([0, round(float(max(pct)))])
            ax.set_yticklabels(["0", f"{round(float(max(pct)))}%"])
            ax.text(
                0.01,
                0.93,
                f"{label}  ({n:,}, {measure} length)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=S.FS_TICK,
                bbox={"facecolor": S.WHITE, "edgecolor": "none", "pad": 0.6},
            )
            ax.text(
                0.99,
                0.93,
                f"mean {np.mean(ls):.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=S.FS_TICK,
                bbox={"facecolor": S.WHITE, "edgecolor": "none", "pad": 0.6},
            )
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            dump.append(
                {
                    "set": label,
                    "measure": measure,
                    "n": n,
                    "pct_by_lines": dict(zip(xs.tolist(), pct.tolist(), strict=True)),
                }
            )
        axs[0].text(
            12.8,
            axs[0].get_ylim()[1] * 0.55,
            "longest pretraining\ntheorem (12 lines)",
            ha="left",
            va="center",
            fontsize=S.FS_TICK,
            color=S.INK,
        )
        axs[-1].set_xticks(
            [2, 6, 9, 12, 15, 20, 25, 30], ["2", "6", "9", "12", "15", "20", "25", "30+"]
        )
        axs[-1].set_xlim(0.5, MAX + 0.5)
        axs[-1].set_xlabel("proof length (lines)")
        fig.supylabel("% of the set", fontsize=S.FS_LABEL)
        S.save(fig, Path(a.out) / "fig_length_dists")
        S.close(fig)
    (Path(a.out) / "fig_length_dists.json").write_text(json.dumps(dump, indent=1) + "\n")
    print(f"wrote {a.out}/fig_length_dists.{{pdf,svg,png,json}}")


if __name__ == "__main__":
    main()
