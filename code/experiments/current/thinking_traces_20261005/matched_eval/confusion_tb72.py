"""Confusion matrices, EI vs GRPO, per problem (meeting action item, 2026-10-09).

    ND_OFFLINE=1 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.confusion_tb72 \
        [--set tb72|charles149]

For each pair at matched training attempts (EI round r <-> GRPO iteration 39 r; the GRPO recipe
v2 pairs: 78 iterations <-> 2 EI rounds on the same curriculum), every problem is "solved" by a
model if any of its 64 attempts is Lean-verified. `--set charles149` uses Charles's release
2026-10-02 reduced to the 149 problems that share no class with any training pool
(`charles_eval/build_clean.py`); the v2 reads cover all 173 and are cut to those 149. The 2 x 2
counts (both, EI only, GRPO only, neither) go to a markdown table with the problem names of the
off-diagonal cells; the figures plot each problem's pass rate under EI (x) against GRPO (y):
`fig_confusion_<set>` the old and current pairs, `fig_confusion_<set>_v2` the v2 pairs. Pairs
whose rows are missing (runs still going) are skipped and appear on the next run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from code.ndbench.figures import style as S

E = Path("/scratch/cowifornia/nd-rl/think/runs/eval")
M, TTS, CH, V2 = E / "matched", E / "tts", E / "charles", E / "v2"
CLEAN = Path("/scratch/cowifornia/nd-rl/think/runs/pools/charles_1002/charles_clean.jsonl")
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#6b6b6b"


@dataclass(frozen=True)
class Pair:
    title: str
    ei: Path
    grpo: Path


PAIRS = [
    Pair(
        "old, no thinking, seed 0",
        M / "tb72_eiT0_r2/rows_plain_B0.jsonl",
        M / "tb72_grpoT0_it78/rows_plain_B0.jsonl",
    ),
    Pair(
        "old, thinking, seed 0",
        M / "tb72_eiT1v1e_r2/rows_plain_B256.jsonl",
        M / "tb72_grpoT1v1e_it78/rows_plain_B256.jsonl",
    ),
    Pair(
        "old, thinking, GRPO seed 1",
        M / "tb72_eiT1v1e_r2/rows_plain_B256.jsonl",
        M / "tb72_grpoT1v1es1_it78/rows_plain_B256.jsonl",
    ),
    Pair(
        "current, thinking, seed 0",
        M / "tb72_eiS1g3_h712_r2/rows_plain_B64.jsonl",
        TTS / "tb72_grpo_g2s0_k64/rows_plain_B64.jsonl",
    ),
    Pair(
        "current, thinking, seed 1",
        TTS / "tb72_think_r2s1_k64/rows_plain_B64.jsonl",
        TTS / "tb72_grpo_g2s1_k64/rows_plain_B64.jsonl",
    ),
]


def _v2(read: str, arm: str, seed: int, b: int) -> Path:
    return V2 / f"{read}_{arm}_s{seed}_plain_k64" / f"rows_plain_B{b}.jsonl"


def v2_pairs(read: str) -> list[Pair]:
    """GRPO recipe v2, seed 0: each GRPO arm against the EI arm with the same sampling."""
    out = []
    for title, ei, grpo, b in (
        ("thinking, curriculum only", "ei_think", "grpo_think", 64),
        ("thinking + verifier (redraw)", "ei_think_v", "grpo_think_v", 64),
        ("thinking + verifier (stop)", "ei_think_v", "grpo_think_vs", 64),
        ("no thinking, curriculum only", "ei_nothink", "grpo_nothink", 0),
        ("no thinking + verifier (redraw)", "ei_nothink_v", "grpo_nothink_v", 0),
        ("no thinking + verifier (stop)", "ei_nothink_v", "grpo_nothink_vs", 0),
        ("thinking, curriculum only, GRPO at half lr", "ei_think", "grpo_think_h", 64),
        ("thinking + verifier (stop), GRPO at half lr", "ei_think_v", "grpo_think_vs_h", 64),
        ("no thinking, curriculum only, GRPO at half lr", "ei_nothink", "grpo_nothink_h", 0),
    ):
        out.append(Pair(f"v2, {title}, seed 0", _v2(read, ei, 0, b), _v2(read, grpo, 0, b)))
    return out


CHARLES_PAIRS = [
    Pair(
        "old, no thinking, seed 0",
        CH / "ch149_eiT0_r2/rows_plain_B0.jsonl",
        CH / "ch149_grpoT0_it78/rows_plain_B0.jsonl",
    ),
    Pair(
        "old, thinking, seed 0",
        CH / "ch149_eiT1v1e_r2/rows_plain_B256.jsonl",
        CH / "ch149_grpoT1v1e_it78/rows_plain_B256.jsonl",
    ),
    Pair(
        "old, thinking, GRPO seed 1",
        CH / "ch149_eiT1v1e_r2/rows_plain_B256.jsonl",
        CH / "ch149_grpoT1v1es1_it78/rows_plain_B256.jsonl",
    ),
    Pair(
        "current, thinking, seed 0",
        CH / "ch149_think_r2s0/rows_plain_B64.jsonl",
        CH / "ch149_grpo_g2s0/rows_plain_B64.jsonl",
    ),
    Pair(
        "current, thinking, seed 1",
        CH / "ch149_think_r2s1/rows_plain_B64.jsonl",
        CH / "ch149_grpo_g2s1/rows_plain_B64.jsonl",
    ),
]
# set -> (title, pairs of the main figure, pairs of the `_v2` figure)
SETS = {
    "tb72": ("textbook72", PAIRS, v2_pairs("tb72")),
    "charles149": ("Charles149", CHARLES_PAIRS, v2_pairs("charles173")),
}


def keep_names(set_: str) -> set[str] | None:
    if set_ != "charles149":
        return None
    with open(CLEAN) as f:
        return {json.loads(line)["name"] for line in f}


def rates(path: Path) -> dict[str, tuple[int, int]]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["name"]] = (int(r["n_ok"]), int(r["k"]))
    return out


def draw(
    done: list[Pair], keep: set[str] | None, title_set: str, base: Path
) -> tuple[list[str], list[str]]:
    """One scatter panel per pair, saved to `base`; returns the table rows and the off-diagonal
    problem lists."""
    md, lists = [], []
    with S.paper_style():
        ncol = 3
        nrow = -(-len(done) // ncol)
        fig = S.new_figure(S.FULL_W, 2.0 * nrow, layout="constrained")
        grid = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)
        axs = list(grid.flat)[: len(done)]
        for ax in list(grid.flat)[len(done) :]:
            ax.set_visible(False)
        rng = np.random.default_rng(0)
        for ax, p in zip(axs, done, strict=True):
            ei, gr = rates(p.ei), rates(p.grpo)
            names = sorted((set(ei) & set(gr)) if keep is None else (set(ei) & set(gr) & keep))
            x = np.array([ei[n][0] / ei[n][1] for n in names])
            y = np.array([gr[n][0] / gr[n][1] for n in names])
            se, sg = x > 0, y > 0
            both, ei_only, gr_only, neither = (
                int((se & sg).sum()),
                int((se & ~sg).sum()),
                int((~se & sg).sum()),
                int((~se & ~sg).sum()),
            )
            title = p.title.replace("\n", " ")
            md.append(
                f"| {title} | {both} | {ei_only} | {gr_only} | {neither} | "
                f"{100 * x.mean():.1f}% | {100 * y.mean():.1f}% |"
            )
            cells = list(zip(names, se, sg, strict=True))
            only_ei = ", ".join(n for n, s, t in cells if s and not t) or "-"
            only_gr = ", ".join(n for n, s, t in cells if t and not s) or "-"
            lists.append(f"- {title}: EI only: {only_ei}; GRPO only: {only_gr}")
            jx = rng.uniform(-0.012, 0.012, len(x))
            jy = rng.uniform(-0.012, 0.012, len(y))
            col = np.where(se & ~sg, ORANGE, np.where(~se & sg, BLUE, GRAY))
            ax.scatter(x + jx, y + jy, s=6, c=col, lw=0, alpha=0.85)
            ax.plot([0, 1], [0, 1], color=S.INK, lw=0.5, ls=(0, (2, 2)))
            counts = f"EI only {ei_only} · GRPO only {gr_only}\nboth {both} · neither {neither}"
            ax.set_title(f"{p.title.removeprefix('v2, ')}\n{counts}", fontsize=S.FS_TICK)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
            ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
            S.light_grid(ax)
        for row in grid:
            row[0].set_ylabel("GRPO pass rate")
        fig.supxlabel(
            f"EI pass rate (per {title_set} problem, 64 attempts; EI round 2 vs GRPO iteration 78)",
            fontsize=S.FS_LABEL,
        )
        key = ((ORANGE, "solved by EI only"), (BLUE, "by GRPO only"), (GRAY, "by both or neither"))
        handles = [
            Line2D([], [], ls="none", marker="o", ms=3, mfc=c, mec="none", label=t) for c, t in key
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        S.save(fig, base)
        S.close(fig)
    return md, lists


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--set", choices=sorted(SETS), default="tb72")
    ap.add_argument("--table", default=None, help="default: matched_eval/confusion_<set>.md")
    a = ap.parse_args(argv)
    title_set, *groups = SETS[a.set]
    keep = keep_names(a.set)
    here = "code/experiments/current/thinking_traces_20261005/matched_eval"
    table = a.table or f"{here}/confusion_{a.set}.md"
    md = [
        f"# {title_set}: EI vs GRPO, problem by problem (generated by `confusion_tb72.py`)",
        "",
        "Solved = at least one Lean-verified proof in 64 attempts. EI round r is matched to GRPO",
        "iteration 39 r on sampled attempts (v2: 78 iterations vs 2 EI rounds, same curriculum).",
        "",
        "| pair | both | EI only | GRPO only | neither | EI pass@1 | GRPO pass@1 |",
        "|---|---|---|---|---|---|---|",
    ]
    lists: list[str] = []
    wrote = []
    for pairs, sfx in zip(groups, ("", "_v2"), strict=True):
        done = [p for p in pairs if p.ei.exists() and p.grpo.exists()]
        for p in pairs:
            if p not in done:
                print(f"skipped (rows not there yet): {p.title}")
        if not done:
            continue
        base = Path(a.out) / f"fig_confusion_{a.set}{sfx}"
        rows, offdiag = draw(done, keep, title_set, base)
        md += rows
        lists += offdiag
        wrote.append(f"{base}.{{pdf,svg,png}}")
    Path(table).write_text(
        "\n".join(md + ["", "Problems in the off-diagonal cells:", ""] + lists) + "\n"
    )
    print("\n".join(md))
    print(f"wrote {', '.join(wrote) or 'no figure'} and {table}")


if __name__ == "__main__":
    main()
