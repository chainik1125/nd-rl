"""The recipe against EI + verifier on Charles149, by proof length: grouped bars per length bin.

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.charles_length_hist \
        [--draws 4000] [--v2 RUNS/eval/v2] [--out FIGDIR] [--table PATH]

RECIPE.md's headline pair, GRPO "merged recipe + KL 0" (`grpo_think_vs_mb_kkl0`) against "EI +
verifier" (`ei_think_v`), on Charles149 only (never textbook72: no textbook72 file is read). The
same reads as RECIPE.md: plain arm at B64, 64 attempts per problem, Stage-1 seeds 0, 1, 2 (every
seed both arms have), through `v2_results`'s loader. Everything but the figure is `length_split`'s:
the reference lengths (`references`; Charles149's are search-minimal), the Charles149 bins
(`BINS`), and per bin `stats.ci` / `stats.paired_diff` (`bins_of`: hierarchical bootstrap over
seeds and problems, seeds paired). The "all" row is checked against RECIPE.md's Charles149 numbers.

Writes `fig_charles_length_hist` (PDF/SVG/PNG) to `--out` and the table to `--table`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Patch

from code.experiments.current.thinking_traces_20261005.matched_eval import length_split as LSP
from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as V
from code.ndbench.figures import style as S
from code.thinking import stats

SET = "charles149"
PAIR = LSP.Pair("kkl0_ei_v", "grpo_think_vs_mb_kkl0", "ei_think_v")
LABEL = {PAIR.a: "GRPO, merged recipe + KL 0", PAIR.b: "EI + verifier"}
SHORT = {PAIR.a: "GRPO", PAIR.b: "EI"}
# style.py's validated categorical pair (base / rl): Okabe-Ito blue, vermillion hatched.
COL = {PAIR.a: S.OKABE_ITO["blue"], PAIR.b: S.OKABE_ITO["vermillion"]}
HATCH = {PAIR.a: None, PAIR.b: "//////"}
#: RECIPE.md (leon/main, "Evidence"; v2_results_ynez.md §1 and §3), Charles149, plain reads, 3
#: seeds, in `V.fmt`'s format: the "all" row must reproduce these.
RECIPE = {
    "a1": "70.5 [64.2, 76.3]",
    "b1": "63.7 [55.8, 71.0]",
    "d1": "+6.9 [+1.0, +13.2]",
    "a5": "78.9 [72.6, 84.8]",
    "b5": "75.9 [68.9, 82.1]",
    "d5": "+3.1 [-2.2, +8.8]",
}
BAR_W = 0.3  # bar width (one bin group is 1.0 wide)
BAR_GAP = 0.028  # between the two bars of a group (about 2 px at the figure's size)
ALL_GAP = 0.55  # extra space before the "all" group
OUT = "experiment-summaries/2026-10-05-thinking-traces/figures/ynez"
TABLE = "code/experiments/current/thinking_traces_20261005/matched_eval/charles_length_hist_ynez.md"


def load(ld: V.Loader, draws: int) -> tuple[list[LSP.Cell], dict[str, LSP.Length]]:
    """The Charles149 cells (each bin, then "all") and every problem's reference length."""
    ref = LSP.references(ld.keep, (SET,))[SET]
    L = {n: LSP.Length(v, True) for n, v in ref.items()}
    seeds = V.common_seeds(ld, [PAIR.a, PAIR.b], SET)
    x, y = ld.arm(PAIR.a, SET, seeds), ld.arm(PAIR.b, SET, seeds)
    if x is None or y is None:
        raise SystemExit(f"{V.SET[SET].title}: no seed with both arms read")
    if set(x.names) != set(y.names) or set(x.names) - set(L):
        raise ValueError("the arms' problems differ, or a problem has no reference length")
    cells = LSP.bins_of(x, y, L, SET, seeds, draws)
    if sum(c.n for c in cells if c.bin != LSP.ALL) != x.T:
        raise ValueError("the bins do not cover every problem exactly once")
    return cells, L


def recipe_check(cells: Sequence[LSP.Cell]) -> list[str]:
    """One line per RECIPE.md number: what the "all" row gives and whether it matches."""
    c = next(c for c in cells if c.bin == LSP.ALL)
    out = []
    for key, want in RECIPE.items():
        got = V.fmt(c.est[key], key.startswith("d"))
        out.append(f"{key}: {got} ({'matches' if got == want else f'MISMATCH, RECIPE.md {want}'})")
    return out


# ------------------------------------------------------------------------------------- figure


def _pt(v: float) -> str:
    """A fraction as signed points, one decimal, with a true minus sign."""
    return f"{100 * v:+.1f}".replace("-", "−")


def _bars(ax: Axes, xs: Sequence[float], groups: Sequence[LSP.Cell], k: int) -> ErrorbarContainer:
    whisker: ErrorbarContainer | None = None
    for j, arm in enumerate((PAIR.a, PAIR.b)):
        key = f"{'ab'[j]}{k}"
        off = (2 * j - 1) * (BAR_W + BAR_GAP) / 2
        pos = [x + off for x in xs]
        est = [c.est[key].est for c in groups]
        ax.bar(
            pos,
            est,
            width=BAR_W,
            color=COL[arm],
            hatch=HATCH[arm],
            edgecolor=S.WHITE if HATCH[arm] else COL[arm],
            lw=0,
            zorder=2,
        )
        whisker = ax.errorbar(
            pos,
            est,
            yerr=[
                [e - c.est[key].lo for e, c in zip(est, groups, strict=True)],
                [c.est[key].hi - e for e, c in zip(est, groups, strict=True)],
            ],
            fmt="none",
            ecolor=S.INK2,
            elinewidth=0.6,
            capsize=1.6,
            capthick=0.6,
            zorder=3,
        )
    for x, c in zip(xs, groups, strict=True):  # GRPO − EI above each pair
        d = c.est[f"d{k}"]
        ax.text(
            x,
            1.035,
            f"{_pt(d.est)}\n[{_pt(d.lo)}, {_pt(d.hi)}]",
            ha="center",
            va="bottom",
            fontsize=S.FS_SMALL,
            color=S.INK2,
            linespacing=1.05,
        )
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(S.PctFormatter())
    ax.spines["left"].set_bounds(0, 1)
    S.light_grid(ax, "y")
    ax.set_xticks(
        list(xs),
        [
            f"{LSP.bin_label(c.bin)}{'' if c.bin == LSP.ALL else ' lines'}\nn = {c.n}"
            for c in groups
        ],
    )
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    assert whisker is not None
    return whisker


def _histogram(ax: Axes, L: dict[str, LSP.Length]) -> None:
    c = Counter(x.lines for x in L.values())
    lo, hi = min(c), max(c)
    ax.bar(
        sorted(c),
        [c[v] for v in sorted(c)],
        width=0.86,
        color=S.mix(S.INK2, S.WHITE, 0.3),
        lw=0,
        zorder=2,
    )
    bins = LSP.BINS[SET]
    for b in bins[1:]:
        ax.axvline(b[0] - 0.5, color=S.REF, lw=0.5, ls=(0, (1.2, 1.6)), zorder=1)
    top = max(c.values())
    for b in bins:
        mid = (max(b[0], lo) + min(b[1], hi)) / 2
        ax.text(
            mid,
            1.13 * top,
            f"{LSP.bin_label(b)}",
            ha="center",
            va="bottom",
            fontsize=S.FS_SMALL,
            color=S.REF,
        )
    ax.set_ylim(0, 1.3 * top)
    ax.set_xlim(lo - 0.7, hi + 0.7)
    ax.set_xticks(range(lo, hi + 1, 1 if hi - lo <= 24 else 2))
    S.light_grid(ax, "y")


def figure(cells: Sequence[LSP.Cell], L: dict[str, LSP.Length], out: Path) -> None:
    groups = [*sorted((c for c in cells if c.bin != LSP.ALL), key=lambda c: c.bin)]
    groups += [c for c in cells if c.bin == LSP.ALL]
    xs = [float(i) for i in range(len(groups) - 1)] + [len(groups) - 1 + ALL_GAP]
    seeds = groups[-1].seeds
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 5.4, layout="constrained")
        axs = fig.subplots(3, 1, squeeze=False, height_ratios=[1.0, 1.0, 0.62])[:, 0]
        whiskers = []
        for r, k in enumerate((1, 5)):
            ax: Axes = axs[r]
            whiskers.append(_bars(ax, xs, groups, k))
            lab = "first try" if k == 1 else "within 5 tries"
            S.panel_title(
                ax,
                "ab"[r],
                f"pass@{k} ({lab}) by proof length; above each pair: "
                f"{SHORT[PAIR.a]} − {SHORT[PAIR.b]} in points [95% interval]",
                size=S.FS_TICK,
            )
            ax.set_ylabel(f"pass@{k} (% of the bin)")
        ax = axs[2]
        _histogram(ax, L)
        S.panel_title(
            ax, "c", "Charles149 problems at each proof length (dotted: bin edges)", size=S.FS_TICK
        )
        ax.set_ylabel("problems")
        ax.set_xlabel(
            "proof length: ND lines of the search-minimal reference proof "
            "(premise lines in, QED out)"
        )
        handles = [
            Patch(facecolor=COL[a], edgecolor=S.WHITE, hatch=HATCH[a], lw=0, label=LABEL[a])
            for a in (PAIR.a, PAIR.b)
        ]
        fig.legend(
            handles=[*handles, whiskers[0]],
            labels=[
                LABEL[PAIR.a],
                LABEL[PAIR.b],
                f"95% interval (seeds {', '.join(map(str, seeds))} pooled, 64 attempts each)",
            ],
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        S.save(fig, out / "fig_charles_length_hist")
        S.close(fig)


# -------------------------------------------------------------------------------------- table


def table(cells: Sequence[LSP.Cell], checks: Sequence[str], draws: int, out: Path) -> list[str]:
    a, b = SHORT[PAIR.a], SHORT[PAIR.b]
    seeds = next(c for c in cells if c.bin == LSP.ALL).seeds
    md = [
        "# Charles149 by proof length: GRPO (merged recipe + KL 0) and EI + verifier",
        "",
        "Generated by `charles_length_hist.py`; do not edit.",
        "",
        f"- {a}: {LABEL[PAIR.a]} (`{PAIR.a}`); {b}: {LABEL[PAIR.b]} (`{PAIR.b}`).",
        "- Reads: plain arm at B64, 64 attempts per problem, Stage-1 seeds "
        f"{', '.join(map(str, seeds))} pooled (ynez), the reads RECIPE.md uses.",
        "- pass@k in % of the bin's problems; 95% hierarchical-bootstrap intervals over seeds and",
        f"  problems ({draws} draws), seeds paired. Δ = {a} − {b}, in points.",
        "- Proof length: ND lines of the search-minimal reference proof (premise lines in, QED",
        "  out; `length_split.references`); bins are `length_split.BINS`.",
        f"- Figure: `fig_charles_length_hist` in `{out}/`.",
        "",
        f"| bin (lines) | n problems | {a} pass@1 [95%] | {b} pass@1 [95%] | Δ pass@1 [95%] | "
        f"{a} pass@5 [95%] | {b} pass@5 [95%] | Δ pass@5 [95%] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda c: (c.bin == LSP.ALL, c.bin)):
        row = [LSP.bin_label(c.bin), str(c.n)]
        for k in (1, 5):
            row += [V.fmt(c.est[f"a{k}"]), V.fmt(c.est[f"b{k}"]), V.fmt(c.est[f"d{k}"], True)]
        md.append("| " + " | ".join(row) + " |")
    md += [
        "",
        "The all row against RECIPE.md's Charles149 numbers (v2_results_ynez.md §1, §3):",
        "",
    ]
    md += [f"- {line}" for line in checks]
    return md


# --------------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(V.E / "v2"), help="the v2 reads")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--table", default=TABLE)
    a = ap.parse_args(argv)
    ld = V.Loader(Path(a.v2), V.subsets())
    cells, L = load(ld, a.draws)
    checks = recipe_check(cells)
    out = Path(a.out)
    figure(cells, L, out)
    Path(a.table).write_text("\n".join(table(cells, checks, a.draws, out)) + "\n")
    print("\n".join(checks))
    print(f"wrote {a.table} and {out}/fig_charles_length_hist.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
