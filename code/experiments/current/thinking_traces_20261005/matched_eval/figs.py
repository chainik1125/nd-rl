"""Figures for the 2026-10-07 RL comparison: GRPO vs EI, thinking vs no-think GRPO, the
thinking-arm variants, budget forcing and graded Stage 1 (restyled for the paper on 2026-10-08).

    code/.venv/bin/python -m code.experiments.current.thinking_traces_20261005.matched_eval.figs \
        [--out experiment-summaries/2026-10-05-thinking-traces/figures] \
        [--only fig_rl_matched ...] [--verbose]

Reads the stored eval rows under /scratch/cowifornia/nd-rl/think/runs/eval/ (CPU only) and writes a
PDF/SVG/PNG triple per figure. Each figure's caption is the hand-written `<name>.caption.md` next to
it; `--verbose` prints every plotted value so captions can be checked against the data.

Points are pass@k (unbiased, Chen et al. 2021) with 95% percentile intervals of the hierarchical
bootstrap in `code.thinking.stats`: 4,000 draws over theorems, and over seeds too where an arm pools
two seeds. These are the draws `final_tables.py` uses, so plotted values match `final_tables.md`.
The 13-20 reads use the 986 valid items (`valid_1320`). Thinking arms are read at B 256, their full
think budget; no-think arms at B 0. textbook72 (72 problems, nothing resolved) is left out of the RL
figures and kept only in the forcing and graded Stage-1 figures, where the effects are large.

| figure | message |
| --- | --- |
| `fig_rl_matched` | GRPO vs EI at matched sampled attempts, hard12 and 13-20, pass@1 and @5 |
| `fig_rl_t1320_curves` | thinking vs no-think GRPO on 13-20 over training, seeds 1 and 0 |
| `fig_rl_t1320_tokens` | the same at matched sampled tokens per problem (it39, seeds 0+1) |
| `fig_rl_variants` | accuracy against tokens per attempt for every arm at it78 |
| `fig_forcing` | a forced minimum think length lowers pass@1 (Stage-1 and RL'd models) |
| `fig_graded_stage1` | Stage 1 on graded-length traces: pass@1 against effort level |

Encoding, fixed across figures (the validated dataviz categorical palette): colour follows the arm
(no-think GRPO orange, effort-level thinking blue, EI aqua, one budget grey, one budget + token
cost yellow, graded traces magenta, graded + LCPO green). Thinking is a solid line with circles,
no thinking a dashed line with squares; EI markers are hollow. In the variants figure filled
markers are seed 0 and hollow ones seed 1. Several slots are below 3:1 contrast on white, so each
series also has its own marker or line style and a direct label or legend entry, and all text is
in ink colours.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator

from code.ndbench.figures import style as S
from code.thinking import stats

R = Path("/scratch/cowifornia/nd-rl/think/runs")
E = R / "eval"
M = E / "matched"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN = "#e87ba4", "#008300"
GRAY = "#6b6b6b"
DRAWS = 4000  # as final_tables.py: the same theorem resample, so intervals match its tables
BAND = 0.18
DASH: Any = (0, (3.5, 1.6))
NUDGE = 0.05  # variants figure: two seeds within 10% of each other in tokens are drawn +-5% apart
_verbose = False

H12 = "hard12 test (in distribution)"
T1320 = "13–20-line proofs (out of distribution)"


def valid_1320() -> set[str]:
    fin = {}
    with open(R / "pools/test_13_20/test_13_20.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fin[r["name"]] = r
    ok = set()
    with open(M / "sets/t1320_n1000.jsonl") as f:
        for line in f:
            r = json.loads(line)
            g = fin.get(r["name"])
            if (
                g
                and g["prompt"] == r["prompt"]
                and g.get("reference_lines") == r.get("reference_lines")
            ):
                ok.add(r["name"])
    return ok


class Rows:
    """Loads one rows file as a one-seed arm, filtered to `keep` names when given (cached)."""

    def __init__(self, keep: set[str] | None, tmp: Path) -> None:
        self.keep, self.tmp, self.n = keep, tmp, 0
        self._arms: dict[tuple[str, ...], stats.Arm | None] = {}

    def _filtered(self, path: Path) -> Path:
        if self.keep is None:
            return path
        self.n += 1
        out = self.tmp / f"{self.n}_{path.parent.name}_{path.name}"
        with open(path) as f, open(out, "w") as g:
            g.writelines(line for line in f if json.loads(line)["name"] in self.keep)
        return out

    def arm(self, path: Path) -> stats.Arm | None:
        return self.pooled([path])

    def pooled(self, paths: Sequence[Path]) -> stats.Arm | None:
        """One arm with one rows file per seed (seeds are resampled too when there are several)."""
        key = tuple(str(p) for p in paths)
        if key not in self._arms:
            ok = all(p.exists() for p in paths)
            self._arms[key] = (
                stats.load_arm(
                    "+".join(p.parent.name for p in paths),
                    [str(self._filtered(p)) for p in paths],
                )
                if ok
                else None
            )
        return self._arms[key]


Est = tuple[float, float, float]


def est(arm: stats.Arm | None, k: int = 1) -> Est | None:
    if arm is None:
        return None
    e = stats.ci(arm, k, draws=DRAWS)
    return e.est, e.lo, e.hi


def tokens_per_attempt(arm: stats.Arm) -> float:
    assert arm.tokens is not None
    return float(arm.tokens.sum() / arm.n.sum())


def log(fig: str, series: str, x: object, e: Est | None, extra: str = "") -> None:
    if _verbose and e is not None:
        print(
            f"{fig:20s} {series:42s} {x!s:>8s}  {100 * e[0]:5.1f} "
            f"[{100 * e[1]:5.1f}, {100 * e[2]:5.1f}] {extra}"
        )


# ------------------------------------------------------------------------------------ encoding


@dataclass(frozen=True)
class Look:
    """One series' identity: colour, marker, line style, filled or hollow markers."""

    label: str
    color: str
    marker: str
    ls: Any = "-"
    hollow: bool = False
    ms: float = 3.8

    def mkw(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "ms": self.ms,
            "mfc": S.WHITE if self.hollow else self.color,
            "mec": self.color if self.hollow else S.WHITE,
            "mew": 0.9 if self.hollow else 0.5,
        }

    def handle(self, *, line: bool = True) -> Line2D:
        return Line2D(
            [],
            [],
            color=self.color,
            ls=self.ls if line else "none",
            lw=1.3,
            label=self.label,
            **self.mkw(),
        )


GRPO_NT = Look("GRPO, no thinking", ORANGE, "s", DASH)
GRPO_T = Look("GRPO, thinking", BLUE, "o")
EI_NT = Look("EI, no thinking", AQUA, "s", DASH, hollow=True)
EI_T = Look("EI, thinking", AQUA, "o", hollow=True)
ONE_BUDGET = Look("one budget", GRAY, "D", ms=3.4)
TOKEN_COST = Look("one budget + token cost", YELLOW, "v", ms=4.2)
GRADED = Look("graded traces", MAGENTA, "^", ms=4.2)
LCPO = Look("graded + LCPO", GREEN, "P", ms=4.2)


def band_line(ax: Axes, xs: Sequence[float], es: Sequence[Est | None], look: Look) -> None:
    """A line with markers and a shaded 95% band."""
    pts = [(x, e) for x, e in zip(xs, es, strict=True) if e is not None]
    if not pts:
        return
    x = [p[0] for p in pts]
    ax.fill_between(
        x, [p[1][1] for p in pts], [p[1][2] for p in pts], color=look.color, alpha=BAND, lw=0
    )
    ax.plot(x, [p[1][0] for p in pts], color=look.color, ls=look.ls, lw=1.3, **look.mkw())


def points(
    ax: Axes,
    xs: Sequence[float],
    es: Sequence[Est | None],
    look: Look,
    *,
    line: bool = True,
    alpha: float = 1.0,
    lw: float = 1.3,
) -> None:
    """Points with 95% error bars, optionally joined by the series' line."""
    pts = [(x, e) for x, e in zip(xs, es, strict=True) if e is not None]
    if not pts:
        return
    x = [p[0] for p in pts]
    y = [p[1][0] for p in pts]
    err = [[p[1][0] - p[1][1] for p in pts], [p[1][2] - p[1][0] for p in pts]]
    ax.errorbar(
        x,
        y,
        yerr=err,
        color=look.color,
        ls=look.ls if line else "none",
        lw=lw,
        elinewidth=0.8,
        capsize=1.6,
        capthick=0.8,
        alpha=alpha,
        **look.mkw(),
    )


def end_label(
    ax: Axes, x: float, y: float, text: str, *, dx: float = 4, dy: float = 0, ha: str = "left"
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=S.FS_TICK,
        color=S.INK2,
        va="center",
        ha=ha,
    )


def tokens_axis(ax: Axes, ticks: Sequence[float]) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.set_xticklabels([f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def finish(ax: Axes, nbins: int = 6) -> None:
    """Percent y-axis and light grid. Steps of 1, 2 or 5 only: `PctFormatter` picks its decimals
    from the step, so a 2.5-point step would print 7.5% as "8%"."""
    S.pct_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))
    S.light_grid(ax)


# --------------------------------------------------------------------------------------- figures


def fig_matched(rows_all: Rows, rows_1320: Rows, out: Path) -> None:
    """GRPO vs EI at matched sampled attempts: EI round r <-> GRPO iteration 39 r, no thinking
    (B 0) and thinking (B 256), on hard12 and 13-20, pass@1 and @5. Seed 0 with intervals; GRPO
    seed 1 as light lines without intervals, to show the seed spread."""
    steps = [("r1", "it39"), ("r2", "it78")]
    seed1 = {"hollow": True, "ms": 3.2}
    # (look, dir tag, 0 = EI / 1 = GRPO step, budget, x offset, light seed-1 line)
    series = [
        (EI_NT, "eiT0", 0, "B0", -0.1125, False),
        (EI_T, "eiT1v1e", 0, "B256", -0.0675, False),
        (replace(GRPO_NT, label="GRPO seed 0, no thinking"), "grpoT0", 1, "B0", -0.0225, False),
        (replace(GRPO_T, label="GRPO seed 0, thinking"), "grpoT1v1e", 1, "B256", 0.0225, False),
        (
            replace(GRPO_NT, label="GRPO seed 1, no thinking", **seed1),
            "grpoT0s1",
            1,
            "B0",
            0.0675,
            True,
        ),
        (
            replace(GRPO_T, label="GRPO seed 1, thinking", **seed1),
            "grpoT1v1es1",
            1,
            "B256",
            0.1125,
            True,
        ),
    ]
    sets = [("h12", H12, rows_all), ("t1320", T1320, rows_1320)]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.7, layout="constrained")
        axs = fig.subplots(2, 2, sharex=True)
        for r, (s, title, rows) in enumerate(sets):
            for c, k in enumerate((1, 5)):
                ax = axs[r][c]
                for look, tag, grpo, b, dx, light in series:
                    es = [
                        est(rows.arm(M / f"{s}_{tag}_{st[grpo]}" / f"rows_plain_{b}.jsonl"), k)
                        for st in steps
                    ]
                    for st, e in zip(steps, es, strict=True):
                        log("fig_rl_matched", f"{s} {tag} pass@{k}", st[grpo], e)
                    xs = [i + dx for i in range(2)]
                    if light:
                        ys = [e[0] if e else np.nan for e in es]
                        ax.plot(
                            xs, ys, color=look.color, ls=look.ls, lw=0.9, alpha=0.5, **look.mkw()
                        )
                    else:
                        points(ax, xs, es, look)
                S.panel_title(ax, "abcd"[2 * r + c], title)
                ax.set_ylabel(f"pass@{k}")
                ax.set_xlim(-0.35, 1.35)
                ax.set_xticks(
                    [0, 1],
                    [
                        "1 round / 39 iterations\n(160k attempts)",
                        "2 rounds / 78 iterations\n(320k attempts)",
                    ],
                )
                finish(ax)
        fig.supxlabel(
            "EI rounds / GRPO iterations, matched on sampled attempts", fontsize=S.FS_LABEL
        )
        handles = []
        for look, *_, light in series:
            h = look.handle()
            if light:
                h.set_alpha(0.5)
                h.set_linewidth(0.9)
            handles.append(h)
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        S.save(fig, out / "fig_rl_matched")
        S.close(fig)


def _think_dir(seed: int, it: int) -> Path:
    if seed == 0:
        return M / (f"t1320_grpoT1v1e_it{it}" if it in (39, 78) else f"diag/t1320_grpoT1v1e_it{it}")
    return M / (
        f"t1320_grpoT1v1es1_it{it}" if it in (39, 78) else f"curve/t1320_grpoT1v1es1_it{it}"
    )


def fig_curves(rows: Rows, out: Path) -> None:
    """13-20 pass@1 and pass@5 over GRPO iterations: thinking (B 256) at every evaluated
    checkpoint vs no-think GRPO (B 0, evaluated at it39 and it78 only), seed 1 and seed 0."""
    its = {1: [10, 20, 25, 30, 35, 39, 45, 50, 55, 60, 65, 70, 75, 78], 0: [39, 49, 52, 60, 70, 78]}
    nt_dir = {0: "t1320_grpoT0", 1: "t1320_grpoT0s1"}
    think = replace(GRPO_T, label="thinking (GRPO, effort levels)")
    nothink = replace(GRPO_NT, label="no thinking (GRPO), measured at iterations 39 and 78 only")
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 4.4, layout="constrained")
        axs = fig.subplots(2, 2, sharex=True, sharey="col")
        for r, seed in enumerate((1, 0)):
            for c, k in enumerate((1, 5)):
                ax = axs[r][c]
                if seed == 0:
                    ax.axvspan(48, 52, color=S.SHADE, lw=0, zorder=0)
                    ax.text(
                        50,
                        0.98,
                        "clip-low spike",
                        ha="center",
                        va="top",
                        fontsize=S.FS_SMALL,
                        color=S.INK2,
                        transform=ax.get_xaxis_transform(),
                    )
                es = [
                    est(rows.arm(_think_dir(seed, i) / "rows_plain_B256.jsonl"), k)
                    for i in its[seed]
                ]
                for i, e in zip(its[seed], es, strict=True):
                    log("fig_rl_t1320_curves", f"seed {seed} thinking pass@{k}", i, e)
                band_line(ax, its[seed], es, think)
                nt = [
                    est(rows.arm(M / f"{nt_dir[seed]}_it{i}" / "rows_plain_B0.jsonl"), k)
                    for i in (39, 78)
                ]
                for i, e in zip((39, 78), nt, strict=True):
                    log("fig_rl_t1320_curves", f"seed {seed} no thinking pass@{k}", i, e)
                points(ax, [39, 78], nt, nothink)
                S.panel_title(ax, "abcd"[2 * r + c], f"seed {seed}")
                ax.set_ylabel(f"pass@{k}")
                ax.set_xlim(0, 82)
                ax.set_xticks([0, 20, 39, 60, 78])
                finish(ax, nbins=8)
        for ax in axs[1]:
            ax.set_xlabel("GRPO iteration")
        fig.legend(
            handles=[think.handle(), nothink.handle()],
            loc="outside upper center",
            ncol=2,
            frameon=False,
        )
        S.save(fig, out / "fig_rl_t1320_curves")
        S.close(fig)


def fig_tokens(rows_all: Rows, rows_1320: Rows, out: Path) -> None:
    """pass@k at matched sampled tokens per problem, thinking (B 256) vs no-think GRPO at it39,
    seeds 0 and 1 pooled into one two-seed arm each, on hard12 and 13-20."""
    panels = [
        ("h12", H12, rows_all, [200, 500, 1000, 2000, 5000]),
        ("t1320", T1320, rows_1320, [300, 1000, 3000]),
    ]
    arms = [
        (replace(GRPO_T, label="thinking"), ("grpoT1v1e", "grpoT1v1es1"), "B256"),
        (replace(GRPO_NT, label="no thinking"), ("grpoT0", "grpoT0s1"), "B0"),
    ]
    grid = np.geomspace(100, 20000, 61)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35, layout="constrained")
        axs = fig.subplots(1, 2)
        for j, (ax, (s, title, rows, ticks)) in enumerate(zip(axs, panels, strict=True)):
            for look, tags, b in arms:
                a = rows.pooled([M / f"{s}_{t}_it39" / f"rows_plain_{b}.jsonl" for t in tags])
                if a is None or a.tokens is None:
                    continue
                es = [e for e in stats.at_tokens(a, list(grid), draws=DRAWS) if e.in_range]
                xs = [float(e.tokens) for e in es if e.tokens is not None]
                vals: list[Est | None] = [(e.est, e.lo, e.hi) for e in es]
                for x, e in zip(xs, vals, strict=True):
                    log("fig_rl_t1320_tokens", f"{s} {look.label}", round(x), e)
                band_line(ax, xs, vals, replace(look, marker="None"))
                end_label(ax, xs[-1], vals[-1][0] if vals[-1] else 0, look.label)
            tokens_axis(ax, ticks)
            ax.set_xlabel("sampled tokens per problem")
            S.panel_title(ax, "ab"[j], title)
            finish(ax)
        axs[0].set_ylabel("pass@k at matched tokens")
        S.save(fig, out / "fig_rl_t1320_tokens")
        S.close(fig)


def fig_variants(rows_all: Rows, rows_1320: Rows, out: Path) -> None:
    """Accuracy against tokens per attempt at it78: one point per run, seeds of an arm joined."""
    arms = [
        (replace(GRPO_NT, label="no thinking"), ("grpoT0", "grpoT0s1"), "B0"),
        (replace(GRPO_T, label="effort levels"), ("grpoT1v1e", "grpoT1v1es1"), "B256"),
        (ONE_BUDGET, ("T1v1_b256",), "B256"),
        (TOKEN_COST, ("T1v1_b256_lp02",), "B256"),
        (GRADED, ("T1v1g4e",), "B256"),
        (LCPO, ("T1v1g4e_lcpo",), "B256"),
    ]
    panels = [
        ("h12", "hard12 test", rows_all, 1, [150, 300, 600], (115, 1050)),
        ("t1320", "13–20-line proofs", rows_1320, 1, [300, 1000, 2000], (215, 2400)),
        ("t1320", "13–20-line proofs", rows_1320, 5, [300, 1000, 2000], (215, 2400)),
    ]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.6, layout="constrained")
        axs = fig.subplots(1, 3)
        for j, (ax, (s, title, rows, k, ticks, xlim)) in enumerate(zip(axs, panels, strict=True)):
            for look, tags, b in arms:
                runs = []
                for t in tags:
                    a = rows.arm(M / f"{s}_{t}_it78" / f"rows_plain_{b}.jsonl")
                    e = est(a, k)
                    if a is not None and e is not None:
                        runs.append((tokens_per_attempt(a), e))
                        log("fig_rl_variants", f"{s} {t} pass@{k}", round(runs[-1][0]), e)
                xs = [x for x, _ in runs]
                if len(xs) == 2 and abs(np.log(xs[1] / xs[0])) < np.log(1 + 2 * NUDGE):
                    # seeds at nearly the same cost: offset by +-NUDGE so the bars stay apart
                    gm, sign = float(np.sqrt(xs[0] * xs[1])), 1 if xs[1] >= xs[0] else -1
                    xs = [gm * (1 + NUDGE) ** -sign, gm * (1 + NUDGE) ** sign]
                for sd, (x, (_, e)) in enumerate(zip(xs, runs, strict=True)):
                    big = replace(look, hollow=sd == 1, ms=look.ms + 1.4)
                    points(ax, [x], [e], big, line=False)
                if len(runs) == 2:
                    ys = [e[0] for _, e in runs]
                    ax.plot(xs, ys, color=look.color, lw=0.7, zorder=1)
            tokens_axis(ax, ticks)
            ax.set_xlim(*xlim)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("tokens per attempt")
            ax.set_ylabel(f"pass@{k}")
            S.panel_title(ax, "abc"[j], title)
            finish(ax)
        handles = [look.handle(line=False) for look, *_ in arms]
        handles += [
            Line2D(
                [],
                [],
                ls="none",
                label="seed 0",
                color=S.INK2,
                **replace(GRPO_T, color=S.INK2).mkw(),
            ),
            Line2D(
                [],
                [],
                ls="none",
                label="seed 1",
                color=S.INK2,
                **replace(GRPO_T, color=S.INK2, hollow=True).mkw(),
            ),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
        S.save(fig, out / "fig_rl_variants")
        S.close(fig)


def fig_forcing(rows: Rows, out: Path) -> None:
    """Budget forcing: pass@1 against the forced minimum think tokens per step (cap 256), for the
    Stage-1 T1v1 model and the RL'd one-budget model (GRPO it78)."""
    mins = [0, 64, 96, 128]
    models = [
        (replace(ONE_BUDGET, label="RL'd model (one budget, GRPO iteration 78)"), "b256_it78", 2.5),
        (
            Look("Stage-1 model (before RL)", BLUE, "o", (0, (1, 1.4)), hollow=True),
            "stage1_T1v1",
            -2.5,
        ),
    ]
    panels = [("h12s", "hard12 test (300 problems)"), ("tb72", "textbook72 (72 problems)")]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.5, layout="constrained")
        axs = fig.subplots(1, 2)
        for j, (ax, (s, title)) in enumerate(zip(axs, panels, strict=True)):
            for look, tag, dx in models:
                es = [
                    est(
                        rows.arm(
                            E
                            / "followup"
                            / f"force_{s}_{tag}_B256_min{m}"
                            / "rows_plain_B256.jsonl"
                        )
                    )
                    for m in mins
                ]
                for m, e in zip(mins, es, strict=True):
                    log("fig_forcing", f"{s} {tag}", m, e)
                points(ax, [m + dx for m in mins], es, look)
            S.panel_title(ax, "ab"[j], title)
            ax.set_xlabel("forced minimum think tokens per step")
            ax.set_xticks(mins)
            ax.set_xlim(-10, 138)
            ax.set_ylim(bottom=0)
            finish(ax)
        axs[0].set_ylabel("pass@1")
        fig.legend(
            handles=[look.handle() for look, *_ in models],
            loc="outside upper center",
            ncol=2,
            frameon=False,
        )
        S.save(fig, out / "fig_forcing")
        S.close(fig)


def fig_graded(rows: Rows, out: Path) -> None:
    """Stage-1 effort curve: graded-length traces (T1v1g4) at each effort level, against the
    original traces (T1v1) at B 0 and B 256 and the no-think Stage-1 model (T0) as references."""
    levels = [0, 16, 64, 256]
    panels = [
        (
            "hard12 test (1,000 problems)",
            M / "h12_T1v1g4_stage1",
            E / "h12_T1v1_s0",
            [E / "h12_T0_s1", E / "h12_T0_s2"],
        ),
        (
            "textbook72 (72 problems)",
            M / "tb72_T1v1g4_stage1",
            E / "tb72_T1v1_s0",
            [E / "tb72_T0_s0", E / "tb72_T0_s1", E / "tb72_T0_s2"],
        ),
    ]
    graded = replace(GRADED, label="graded-length traces", ms=4.4)
    orig = Look("original traces, read at budget 0 and 256", BLUE, "o", "none", hollow=True, ms=4.2)
    t0 = Line2D(
        [], [], color=ORANGE, ls=DASH, lw=1.1, label="no-thinking Stage-1 model (mean of seeds)"
    )
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.55, layout="constrained")
        axs = fig.subplots(1, 2)
        for j, (ax, (title, g4, t1, t0s)) in enumerate(zip(axs, panels, strict=True)):
            xs = list(range(len(levels)))
            p0 = [p / "rows_plain_B0.jsonl" for p in t0s if (p / "rows_plain_B0.jsonl").exists()]
            if p0:
                e0 = est(rows.pooled(p0))
                assert e0 is not None
                log("fig_graded_stage1", f"{title} T0 ({len(p0)} seeds)", "-", e0)
                ax.axhspan(e0[1], e0[2], color=ORANGE, alpha=0.10, lw=0, zorder=0)
                ax.axhline(e0[0], color=ORANGE, ls=DASH, lw=1.1, zorder=1)
            ge = [est(rows.arm(g4 / f"rows_plain_B{b}.jsonl")) for b in levels]
            te = [est(rows.arm(t1 / f"rows_plain_B{b}.jsonl")) for b in (0, 256)]
            for b, e in zip(levels, ge, strict=True):
                log("fig_graded_stage1", f"{title} T1v1g4", b, e)
            for b, e in zip((0, 256), te, strict=True):
                log("fig_graded_stage1", f"{title} T1v1", b, e)
            points(ax, [x - 0.06 for x in xs], ge, graded)
            points(ax, [0.06, 3.06], te, orig, line=False)
            ax.set_xticks(xs, [str(b) for b in levels])
            ax.set_xlabel("effort level (think budget per step, tokens)")
            ax.set_xlim(-0.35, 3.35)
            S.panel_title(ax, "ab"[j], title)
            finish(ax)
        axs[0].set_ylabel("pass@1, Stage 1 (before RL)")
        fig.legend(
            handles=[graded.handle(), orig.handle(line=False), t0],
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        S.save(fig, out / "fig_graded_stage1")
        S.close(fig)


FIGURES: dict[str, Callable[[Rows, Rows, Path], None]] = {
    "fig_rl_matched": lambda a, t, o: fig_matched(a, t, o),
    "fig_rl_t1320_curves": lambda a, t, o: fig_curves(t, o),
    "fig_rl_t1320_tokens": lambda a, t, o: fig_tokens(a, t, o),
    "fig_rl_variants": lambda a, t, o: fig_variants(a, t, o),
    "fig_forcing": lambda a, t, o: fig_forcing(a, o),
    "fig_graded_stage1": lambda a, t, o: fig_graded(a, o),
}


def main(argv: Sequence[str] | None = None) -> None:
    global _verbose
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="render only these figures")
    ap.add_argument("--verbose", action="store_true", help="print every plotted value")
    a = ap.parse_args(argv)
    _verbose = a.verbose
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        r1320, rall = Rows(valid_1320(), tmp), Rows(None, tmp)
        for name, fn in FIGURES.items():
            if not a.only or name in a.only:
                fn(rall, r1320, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
