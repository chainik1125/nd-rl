"""Charles149: where GRPO's pass@1 lead over EI + verifier comes from, problem by problem.

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.charles_overlap \
        [--draws 4000] [--v2 RUNS/eval/v2] [--out FIGDIR] [--table PATH]

The pair and reads of `charles_length_hist`: GRPO "merged recipe + KL 0" (`grpo_think_vs_mb_kkl0`)
against "EI + verifier" (`ei_think_v`), Charles149 only (no textbook72 file is read), plain arm at
B64, 64 attempts per problem, Stage-1 seeds 0, 1, 2. Leon's hypothesis: the problems where GRPO
beats EI at pass@1 are ones EI already solves within 5 tries (GRPO makes existing ability
reliable), and a separate set is too hard for either arm.

Per problem i, arm A and seed s, c_A,s(i) successes of n attempts; per-problem pass@k_A(i) is the
mean over seeds of the unbiased pass@k (`stats.Arm.values`). Seeds are pooled only for "solved at
least once in all 192 attempts".

1. **Coverage.** pass@k for k = 1..64 (mean over problems), 95% intervals and the GRPO − EI gap
   from `stats.ci` / `stats.paired_diff` with seeds paired (`bins_of`'s bootstrap, RNG seed 0).
2. **Where the gap comes from (cross-fit).** Problems are grouped by pass@5 on the even attempts
   (32 per seed): EI's alone, or the mean of both arms'. Each group's pass@1 and pass@5 and its
   *contribution* (the sum over its problems of Δ / 149, in points) are read on the odd attempts
   only, so grouping and measurement share no sampling noise. Contributions add up to the overall
   odd-half Δ in every bootstrap draw (one problem and seed resample for all groups; group
   membership fixed). Robustness: other thresholds, the halves swapped, and the full data grouped
   and measured on the same attempts (no cross-fit, to show the selection effect). Also each
   group's pass@k up to k = 32 on the odd half (sharpening or coverage), and the contributions in
   finer pass@5 bins.
3. **The hard core.** Problems with full-data pass@5 < 20% in both arms, and with no success in any
   of either arm's 192 attempts; their proof lengths and share of the gap; then the reverse cases
   (one arm's pass@5 >= 90% where the other's is < 20%).

Writes `fig_charles_overlap_scatter`, `fig_charles_passk` and `fig_charles_gap_decomp`
(PDF/SVG/PNG) to `--out` and the tables to `--table`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from code.experiments.current.thinking_traces_20261005.matched_eval import (
    charles_length_hist as CLH,
)
from code.experiments.current.thinking_traces_20261005.matched_eval import length_split as LSP
from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as V
from code.ndbench.figures import style as S
from code.thinking import stats

F64 = NDArray[np.float64]
Bool = NDArray[np.bool_]

SET, PAIR, SHORT, LABEL, COL = CLH.SET, CLH.PAIR, CLH.SHORT, CLH.LABEL, CLH.COL
A, B = SHORT[PAIR.a], SHORT[PAIR.b]
KS = tuple(range(1, 65))
K_REPORT = (1, 5, 16, 64)
#: Groups by pass@5 on the selection half, best first: >= HI, [LO, HI), < LO.
GROUPS = ("reliable", "sometimes", "rarely")
GNAME = {"reliable": "reliable", "sometimes": "sometimes", "rarely": "rarely or never"}
#: Okabe-Ito hues not used for the arms (GRPO blue, EI vermillion); a marker per group as well.
GCOL = {
    "reliable": S.OKABE_ITO["bluish_green"],
    "sometimes": S.OKABE_ITO["orange"],
    "rarely": S.OKABE_ITO["reddish_purple"],
}
GMK = {"reliable": "o", "sometimes": "s", "rarely": "^"}
MAIN = (0.9, 0.2)  # (HI, LO)
ROBUST = ((0.8, 0.1), (0.95, 0.3))
BY = {"ei": f"{B}'s pass@5", "both": "mean of both arms' pass@5"}
HARD = 0.2  # hard core: pass@5 below this in both arms
EASY = 0.9  # the reverse cases: one arm at or above this, the other below HARD
OUT = CLH.OUT
TABLE = "code/experiments/current/thinking_traces_20261005/matched_eval/charles_overlap_ynez.md"


@dataclass
class Data:
    """Both arms on Charles149: every attempt, and the even / odd attempts of each read."""

    full: tuple[stats.Arm, stats.Arm]
    even: tuple[stats.Arm, stats.Arm]
    odd: tuple[stats.Arm, stats.Arm]
    lines: F64  # reference length per problem (the arms' order)
    seeds: list[int]

    @property
    def names(self) -> list[str]:
        return self.full[0].names

    @property
    def T(self) -> int:
        return self.full[0].T


def load(ld: V.Loader) -> Data:
    ref = LSP.references(ld.keep, (SET,))[SET]
    seeds = V.common_seeds(ld, [PAIR.a, PAIR.b], SET)
    x, y = ld.arm(PAIR.a, SET, seeds), ld.arm(PAIR.b, SET, seeds)
    if x is None or y is None:
        raise SystemExit(f"{V.SET[SET].title}: no seed with both arms read")
    half = {
        (arm, par): LSP.half_arm(ld, arm, SET, seeds, par)
        for arm in (PAIR.a, PAIR.b)
        for par in (0, 1)
    }
    for arm, full in ((PAIR.a, x), (PAIR.b, y)):
        ev, od = half[(arm, 0)], half[(arm, 1)]
        if not (full.names == ev.names == od.names == y.names):
            raise ValueError(f"{arm}: the halves or arms cover other problems")
        if not (np.array_equal(ev.c + od.c, full.c) and (ev.n == 32).all() and (od.n == 32).all()):
            raise ValueError(f"{arm}: the even and odd halves do not add up to the full read")
    if set(x.names) - set(ref):
        raise ValueError("a problem has no reference length")
    return Data(
        (x, y),
        (half[(PAIR.a, 0)], half[(PAIR.b, 0)]),
        (half[(PAIR.a, 1)], half[(PAIR.b, 1)]),
        np.array([ref[n] for n in x.names], dtype=np.float64),
        seeds,
    )


def p_at(arm: stats.Arm, k: int) -> F64:
    """Per problem: the mean over seeds of the unbiased pass@k."""
    return arm.values(k).mean(0)


# ---------------------------------------------------------------------------------- coverage


@dataclass
class Coverage:
    a: list[stats.Estimate]
    b: list[stats.Estimate]
    d: list[stats.Estimate]
    solved: dict[str, int]  # at least one success in all 192 attempts: a, b, both, either, ...


def coverage(D: Data, draws: int) -> Coverage:
    x, y = D.full
    sa, sb = x.c.sum(0) > 0, y.c.sum(0) > 0
    return Coverage(
        stats.ci_curve(x, KS, draws=draws),
        stats.ci_curve(y, KS, draws=draws),
        [stats.paired_diff(x, y, k, paired_seeds=True, draws=draws) for k in KS],
        {
            A: int(sa.sum()),
            B: int(sb.sum()),
            "both": int((sa & sb).sum()),
            "either": int((sa | sb).sum()),
            "neither": int((~sa & ~sb).sum()),
            f"{A} only": int((sa & ~sb).sum()),
            f"{B} only": int((~sa & sb).sum()),
        },
    )


# ------------------------------------------------------------------------------ decomposition


def _est(label: str, est: float, d: F64, k: int, T: int) -> stats.Estimate:
    lo, hi = np.quantile(d, [0.025, 0.975])
    return stats.Estimate(
        label,
        est,
        float(lo),
        float(hi),
        float(np.std(d, ddof=1)),
        k=k,
        T=T,
        p_boot=stats.boot_p(d),
        draws=len(d),
        samples=d,
    )


def contributions(
    x: stats.Arm, y: stats.Arm, masks: dict[str, Bool], k: int, draws: int
) -> tuple[dict[str, stats.Estimate], stats.Estimate]:
    """Per mask: the sum over its problems of the per-problem Δ pass@k, divided by every problem
    (points of the overall Δ), and the overall Δ. One problem resample and one seed resample
    (paired: both arms start from the same Stage-1 models) for every mask, the same draws as
    `stats.paired_diff` with seeds paired, so in every draw the masks' contributions add up to the
    overall Δ when the masks partition the problems."""
    dr = stats.make_draws(x.T, draws=draws)
    U = dr.seed_weights(x.S)
    Dm = x.values(k) - y.values(k)
    UD = (U @ Dm) / x.S  # draws x problems: seed-resampled per-problem Δ
    sw = dr.W.sum(1)
    per = Dm.mean(0)
    out = {
        g: _est(g, float(per[m].sum() / x.T), (UD * dr.W * m).sum(1) / sw, k, int(m.sum()))
        for g, m in masks.items()
    }
    return out, _est("all", float(per.mean()), (UD * dr.W).sum(1) / sw, k, x.T)


def groups_of(p5: F64, hi: float, lo: float) -> dict[str, Bool]:
    return {"reliable": p5 >= hi, "sometimes": (p5 >= lo) & (p5 < hi), "rarely": p5 < lo}


@dataclass
class Part:
    group: str
    n: int
    est: dict[str, stats.Estimate]  # group means on the measurement half: a1, b1, d1, a5, b5, d5
    con: dict[int, stats.Estimate]  # k -> contribution (points of the overall Δ)


@dataclass
class Decomp:
    by: str  # "ei" or "both"
    hi: float
    lo: float
    split: str  # "even → odd", "odd → even" or "none (full data)"
    parts: list[Part]
    total: dict[int, stats.Estimate]
    names: dict[str, list[str]] = field(default_factory=lambda: {})

    def part(self, g: str) -> Part | None:
        return next((p for p in self.parts if p.group == g), None)


def decompose(
    sel: tuple[stats.Arm, stats.Arm],
    meas: tuple[stats.Arm, stats.Arm],
    by: str,
    hi: float,
    lo: float,
    split: str,
    draws: int,
) -> Decomp:
    p5 = p_at(sel[1], 5) if by == "ei" else (p_at(sel[0], 5) + p_at(sel[1], 5)) / 2
    masks = groups_of(p5, hi, lo)
    xa, xb = meas
    con: dict[int, dict[str, stats.Estimate]] = {}
    total: dict[int, stats.Estimate] = {}
    for k in (1, 5):
        con[k], total[k] = contributions(xa, xb, masks, k, draws)
    parts, names = [], {}
    for g in GROUPS:
        nm = [n for n, ok in zip(xa.names, masks[g], strict=True) if ok]
        names[g] = nm
        parts.append(
            Part(
                g,
                len(nm),
                LSP.estimate(xa, xb, nm, draws, (1, 5)) if nm else {},
                {k: con[k][g] for k in (1, 5)},
            )
        )
    return Decomp(by, hi, lo, split, parts, total, names)


def decompositions(D: Data, draws: int) -> list[Decomp]:
    """The main cross-fits (both groupings at every threshold pair), the halves swapped and the
    full data without a cross-fit at the main thresholds."""
    out = []
    for by in ("ei", "both"):
        for hi, lo in (MAIN, *ROBUST):
            out.append(decompose(D.even, D.odd, by, hi, lo, "even → odd", draws))
        out.append(decompose(D.odd, D.even, by, *MAIN, "odd → even", draws))
        out.append(decompose(D.full, D.full, by, *MAIN, "none (full data)", draws))
    return out


def find(ds: Sequence[Decomp], by: str, th: tuple[float, float], split: str) -> Decomp:
    return next(d for d in ds if d.by == by and (d.hi, d.lo) == th and d.split == split)


def sum_check(d: Decomp, tol: float = 1e-9) -> tuple[list[str], bool]:
    """Whether the contributions add up to the overall Δ, as a point estimate and in every draw."""
    out, ok = [], True
    for k in (1, 5):
        tot = d.total[k]
        s = sum(p.con[k].est for p in d.parts)
        assert tot.samples is not None
        drift = tot.samples.copy()
        for p in d.parts:
            c = p.con[k].samples
            assert c is not None
            drift -= c
        gap = max(abs(s - tot.est), float(np.abs(drift).max()))
        ok &= gap < tol
        out.append(
            f"Δ pass@{k}: contributions sum to {100 * s:+.2f} points, the overall {d.split} Δ is "
            f"{100 * tot.est:+.2f} (largest gap, estimate or any draw: {100 * gap:.0e} points)"
        )
    return out, ok


#: Finer bins of the selection half's pass@5 for the contribution profile: exactly 0, [lo, hi), 1.
FINE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.2),
    (0.2, 0.5),
    (0.5, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
    (1.0, 1.0),
)
K_GROUP = (1, 5, 16, 32)  # the odd half has 32 attempts per seed


def fine_label(b: tuple[float, float]) -> str:
    lo, hi = b
    return f"{100 * lo:.0f}%" if lo == hi else f"[{100 * lo:.0f}%, {100 * hi:.0f}%)"


def in_fine(p5: F64, b: tuple[float, float]) -> Bool:
    lo, hi = b
    if lo == hi:
        return p5 == lo
    return ((p5 > lo) if lo == 0 else (p5 >= lo)) & (p5 < hi)


@dataclass
class Profile:
    by: str
    n: list[int]
    con: dict[int, list[stats.Estimate]]  # k -> per FINE bin
    total: dict[int, stats.Estimate]


def profile(D: Data, by: str, draws: int) -> Profile:
    """Contributions (odd attempts) by finer bins of pass@5 on the even attempts."""
    ea, eb = D.even
    p5 = p_at(eb, 5) if by == "ei" else (p_at(ea, 5) + p_at(eb, 5)) / 2
    masks = {fine_label(b): in_fine(p5, b) for b in FINE}
    if int(sum(m.sum() for m in masks.values())) != D.T:
        raise ValueError("the finer bins do not cover every problem exactly once")
    con: dict[int, list[stats.Estimate]] = {}
    total: dict[int, stats.Estimate] = {}
    for k in (1, 5):
        c, total[k] = contributions(*D.odd, masks, k, draws)
        con[k] = [c[fine_label(b)] for b in FINE]
    return Profile(by, [int(m.sum()) for m in masks.values()], con, total)


@dataclass
class GroupK:
    """Per group of a decomposition: pass@k of both arms on the measurement half, and how many
    problems each arm solved at least once there (seeds pooled)."""

    group: str
    n: int
    est: dict[str, stats.Estimate]
    solved: tuple[int, int]


def group_curves(meas: tuple[stats.Arm, stats.Arm], d: Decomp, draws: int) -> list[GroupK]:
    xa, xb = meas
    out = []
    for g in (*GROUPS, "all"):
        nm = list(xa.names) if g == "all" else d.names[g]
        if not nm:
            continue
        sa, sb = xa.subset(nm), xb.subset(nm)
        out.append(
            GroupK(
                g,
                len(nm),
                LSP.estimate(xa, xb, nm, draws, K_GROUP),
                (int((sa.c.sum(0) > 0).sum()), int((sb.c.sum(0) > 0).sum())),
            )
        )
    return out


# ------------------------------------------------------------------------------------ hard core


@dataclass
class Hard:
    core: Bool  # both arms' full-data pass@5 < HARD
    zero: Bool  # no success in any of either arm's 192 attempts
    a_only: Bool  # GRPO pass@5 >= EASY, EI < HARD
    b_only: Bool  # EI pass@5 >= EASY, GRPO < HARD
    con: dict[str, dict[int, stats.Estimate]]  # mask -> k -> contribution (full data)
    cross: dict[int, stats.Estimate]  # core defined on the even attempts, Δ on the odd ones
    n_cross: int


def hard(D: Data, draws: int) -> Hard:
    x, y = D.full
    pa, pb = p_at(x, 5), p_at(y, 5)
    core = (pa < HARD) & (pb < HARD)
    zero = (x.c.sum(0) == 0) & (y.c.sum(0) == 0)
    a_only = (pa >= EASY) & (pb < HARD)
    b_only = (pb >= EASY) & (pa < HARD)
    masks = {"core": core, "zero": zero, "a_only": a_only, "b_only": b_only}
    con: dict[str, dict[int, stats.Estimate]] = {m: {} for m in masks}
    for k in (1, 5):
        c, _ = contributions(x, y, masks, k, draws)
        for m in masks:
            con[m][k] = c[m]
    ea, eb = D.even
    ce = (p_at(ea, 5) < HARD) & (p_at(eb, 5) < HARD)
    cross = {k: contributions(*D.odd, {"core": ce}, k, draws)[0]["core"] for k in (1, 5)}
    return Hard(core, zero, a_only, b_only, con, cross, int(ce.sum()))


def describe(v: F64) -> str:
    if len(v) == 0:
        return "—"
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return (
        f"{len(v)} | {v.mean():.1f} | {med:.0f} | {q1:.0f}–{q3:.0f} | {v.min():.0f}–{v.max():.0f}"
    )


def bin_counts(v: F64) -> list[int]:
    return [int(((v >= b[0]) & (v <= b[1])).sum()) for b in LSP.BINS[SET]]


# ------------------------------------------------------------------------------------- figures


def _points(v: float, _pos: float | None = None) -> str:
    """Tick label: a fraction as whole signed points (as `length_split`'s Δ axes)."""
    return f"{100 * v:+.0f}".replace("-", "−") if round(100 * v) else "0"


def _pt(v: float, signed: bool = True) -> str:
    return (f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}").replace("-", "−")


def _corner_jitter(
    x: F64, y: F64, rng: np.random.Generator, r: float = 0.022
) -> tuple[F64, F64, dict[str, int]]:
    """Spread the problems that sit exactly on (0, 0) or (1, 1) into a small cloud just inside
    the corner, so their number shows; every other point stays where it is."""
    x, y = x.copy(), y.copy()
    counts = {}
    for name, c, s in (("0", 0.0, 1.0), ("1", 1.0, -1.0)):
        m = (x == c) & (y == c)
        counts[name] = int(m.sum())
        x[m] = c + s * rng.uniform(0.0, r, int(m.sum()))
        y[m] = c + s * rng.uniform(0.0, r, int(m.sum()))
    return x, y, counts


def fig_scatter(D: Data, cov: Coverage, main: Decomp, out: Path) -> None:
    """Per problem, GRPO against EI at pass@1 and pass@5 (every attempt), coloured by the main
    grouping. The counts sit in empty bands above and below the unit square, never on points."""
    x, y = D.full
    grp = np.empty(D.T, dtype=object)
    for g in GROUPS:
        grp[np.isin(D.names, main.names[g])] = g
    band = 0.11  # empty margin above 100% and below 0% for the counts
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.4, layout="constrained")
        axs = fig.subplots(1, 2, squeeze=False)[0]
        for j, k in enumerate((1, 5)):
            ax: Axes = axs[j]
            pa, pb = p_at(x, k), p_at(y, k)
            ex, ey, corner = _corner_jitter(pb, pa, np.random.default_rng(k))
            ax.plot([0, 1], [0, 1], color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
            for g in GROUPS:
                m = grp == g
                ax.scatter(
                    ex[m],
                    ey[m],
                    s=11,
                    marker=GMK[g],
                    color=GCOL[g],
                    alpha=0.72,
                    edgecolors=S.WHITE,
                    linewidths=0.35,
                    zorder=3,
                )
            ax.set_xlim(-0.035, 1.035)
            ax.set_ylim(-band, 1 + band)
            ax.set_aspect("equal")
            for axis in ("x", "y"):
                S.pct_axis(ax, axis, nbins=5)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.spines["left"].set_bounds(0, 1)
            S.light_grid(ax, "both")
            ax.set_xlabel(f"{B} pass@{k} (mean over seeds 0–2)")
            ax.set_ylabel(f"{A} pass@{k} (mean over seeds 0–2)")
            above, below = int((pa > pb).sum()), int((pa < pb).sum())
            S.panel_title(
                ax,
                "ab"[j],
                f"pass@{k}, all 192 attempts per arm: Δ {_pt(cov.d[k - 1].est)} points",
                size=S.FS_TICK,
            )
            top, bot = 1 + band / 2, -band / 2
            kw: dict[str, Any] = {"fontsize": S.FS_SMALL, "color": S.INK2, "va": "center"}
            ax.text(0.0, top, f"{A} better: {above}", ha="left", **kw)
            ax.text(1.03, bot, f"{B} better: {below}", ha="right", **kw)
            ax.text(1.03, top, f"{corner['1']} at (100%, 100%) ↓", ha="right", **kw)
            ax.text(0.0, bot, f"↑ {corner['0']} at (0%, 0%)", ha="left", **kw)
        th = f"{100 * main.hi:.0f}%", f"{100 * main.lo:.0f}%"
        lab = {"reliable": f"≥ {th[0]}", "sometimes": f"{th[1]}–{th[0]}", "rarely": f"< {th[1]}"}
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker=GMK[g],
                ms=4,
                color=GCOL[g],
                mec=S.WHITE,
                mew=0.35,
                label=f"{GNAME[g]}, {lab[g]} (n = {len(main.names[g])})",
            )
            for g in GROUPS
        ]
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=3,
            frameon=False,
            title=f"colour: {B}'s pass@5 on the even attempts (32 per seed and problem)\n"
            f"'{A} better': problems above the diagonal; problems exactly at a corner are "
            "jittered inside it",
            title_fontsize=S.FS_SMALL,
        )
        S.save(fig, out / "fig_charles_overlap_scatter")
        S.close(fig)


def fig_passk(cov: Coverage, seeds: Sequence[int], out: Path) -> None:
    ks = np.array(KS, dtype=np.float64)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 3.3, layout="constrained")
        axs = fig.subplots(2, 1, sharex=True, squeeze=False, height_ratios=[1.0, 0.62])[:, 0]
        ax: Axes = axs[0]
        for arm, es, mk in ((PAIR.a, cov.a, "o"), (PAIR.b, cov.b, "s")):
            ax.fill_between(
                ks,
                [e.lo for e in es],
                [e.hi for e in es],
                color=COL[arm],
                alpha=S.BAND_ALPHA * 1.6,
                lw=0,
                zorder=1,
            )
            ax.plot(ks, [e.est for e in es], color=COL[arm], lw=1.1, zorder=3, label=LABEL[arm])
            ax.plot(
                list(K_REPORT),
                [es[k - 1].est for k in K_REPORT],
                ls="none",
                marker=mk,
                ms=3.4,
                mfc=COL[arm],
                mec=S.WHITE,
                mew=0.5,
                zorder=4,
            )
        ax.set_ylim(0.55, 1.0)
        S.pct_axis(ax, nbins=5)
        S.light_grid(ax, "y")
        ax.set_ylabel("pass@k (% of 149)")
        S.panel_title(
            ax, "a", "Charles149 pass@k, k = 1…64 (markers at k = 1, 5, 16, 64)", size=S.FS_TICK
        )
        ax.legend(loc="lower right", frameon=False)
        ax = axs[1]
        ax.axhline(0, color=S.INK, lw=0.6, zorder=2)
        ax.fill_between(
            ks,
            [e.lo for e in cov.d],
            [e.hi for e in cov.d],
            color=S.INK2,
            alpha=S.BAND_ALPHA * 1.6,
            lw=0,
            zorder=1,
        )
        ax.plot(ks, [e.est for e in cov.d], color=S.INK2, lw=1.1, zorder=3)
        for k in K_REPORT:
            e = cov.d[k - 1]
            ax.plot(k, e.est, marker="D", ms=3, color=S.INK2, mec=S.WHITE, mew=0.5, zorder=4)
            ax.annotate(
                f"{_pt(e.est)} [{_pt(e.lo)}, {_pt(e.hi)}]",
                (k, e.est),
                xytext=(0, 4),
                textcoords="offset points",
                ha="left" if k == 1 else "right" if k == 64 else "center",
                va="bottom",
                fontsize=S.FS_SMALL,
                color=S.INK2,
            )
        ax.yaxis.set_major_formatter(_points)
        S.light_grid(ax, "y")
        ax.set_ylabel(f"{A} − {B}\n(points)")
        S.panel_title(ax, "b", f"gap {A} − {B}, 95% interval", size=S.FS_TICK)
        lo = min(e.lo for e in cov.d)
        hi = max(e.hi for e in cov.d)
        ax.set_ylim(min(lo, 0) - 0.01, hi + 0.01)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        S.log2_k_axis(ax, 64, every=2)
        ax.set_xlabel(
            f"k (attempts per problem; log scale); bands: 95% intervals, seeds "
            f"{', '.join(map(str, seeds))} paired, problems resampled"
        )
        S.save(fig, out / "fig_charles_passk")
        S.close(fig)


def fig_gap_decomp(ds: Sequence[Decomp], out: Path) -> None:
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.75, layout="constrained")
        axs = fig.subplots(1, 2, sharey=True, squeeze=False)[0]
        w, gap = 0.36, 0.03
        for j, by in enumerate(("ei", "both")):
            d = find(ds, by, MAIN, "even → odd")
            ax: Axes = axs[j]
            ax.axhline(0, color=S.INK, lw=0.6, zorder=2)
            cats: list[tuple[str, str, dict[int, stats.Estimate], int]] = [
                (GNAME[p.group], GCOL[p.group], p.con, p.n) for p in d.parts
            ]
            cats.append(("all", S.mix(S.INK2, S.WHITE, 0.25), d.total, sum(p.n for p in d.parts)))
            xs = np.arange(len(cats), dtype=np.float64)
            xs[-1] += 0.35
            for i, (_lab, col, con, _n) in enumerate(cats):
                for r, k in enumerate((1, 5)):
                    e = con[k]
                    pos = xs[i] + (2 * r - 1) * (w + gap) / 2
                    ax.bar(
                        pos,
                        e.est,
                        width=w,
                        color=col if k == 1 else S.mix(col, S.WHITE, 0.45),
                        hatch=None if k == 1 else "//////",
                        edgecolor=S.WHITE if k == 5 else col,
                        lw=0,
                        zorder=3,
                    )
                    ax.errorbar(
                        pos,
                        e.est,
                        yerr=[[e.est - e.lo], [e.hi - e.est]],
                        fmt="none",
                        ecolor=S.INK2,
                        elinewidth=0.6,
                        capsize=1.6,
                        capthick=0.6,
                        zorder=4,
                    )
                    ax.text(
                        pos,
                        e.hi + 0.004,
                        _pt(e.est),
                        ha="center",
                        va="bottom",
                        fontsize=S.FS_SMALL,
                        color=S.INK2,
                    )
            ax.set_xticks(list(xs), [f"{lab}\nn = {n}" for lab, _c, _e, n in cats])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            ax.yaxis.set_major_formatter(_points)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
            S.light_grid(ax, "y")
            S.panel_title(
                ax,
                "ab"[j],
                f"grouped by {BY[by]} (even attempts)",
                size=S.FS_TICK,
            )
        axs[0].set_ylabel("contribution to the gap\n(points of the overall Δ)")
        shown = [
            e
            for by in ("ei", "both")
            for d in [find(ds, by, MAIN, "even → odd")]
            for e in [*d.total.values(), *(p.con[k] for p in d.parts for k in (1, 5))]
        ]
        lo, hi = min(e.lo for e in shown), max(e.hi for e in shown)
        axs[0].set_ylim(min(lo, 0) - 0.01, hi + 0.025)
        handles = [
            Patch(facecolor=S.INK2, lw=0, label="Δ pass@1"),
            Patch(
                facecolor=S.mix(S.INK2, S.WHITE, 0.45),
                edgecolor=S.WHITE,
                hatch="//////",
                lw=0,
                label="Δ pass@5",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=2,
            frameon=False,
            title=f"each group's contribution to {A} − {B}: the sum over its problems of Δ / 149, "
            "read on the odd attempts (95% intervals)\n"
            f"groups from pass@5 on the even attempts: reliable ≥ {100 * MAIN[0]:.0f}%, "
            f"sometimes {100 * MAIN[1]:.0f}–{100 * MAIN[0]:.0f}%, rarely or never "
            f"< {100 * MAIN[1]:.0f}%",
            title_fontsize=S.FS_SMALL,
        )
        S.save(fig, out / "fig_charles_gap_decomp")
        S.close(fig)


# -------------------------------------------------------------------------------------- tables


def _th(d: Decomp) -> str:
    return f"{100 * d.hi:.0f} / {100 * d.lo:.0f}"


def _share(e: stats.Estimate, tot: stats.Estimate) -> str:
    return f"{100 * e.est / tot.est:.0f}%" if abs(tot.est) > 1e-9 else "—"


def _main_table(d: Decomp) -> list[str]:
    md = [
        f"| group ({BY[d.by]}, even) | n | {A} pass@1 | {B} pass@1 | Δ pass@1 (group mean) | "
        f"contribution to Δ pass@1 (points) | share | {A} pass@5 | {B} pass@5 | "
        "Δ pass@5 (group mean) | contribution to Δ pass@5 (points) | share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lab = {
        "reliable": f"≥ {100 * d.hi:.0f}%",
        "sometimes": f"{100 * d.lo:.0f}–{100 * d.hi:.0f}%",
        "rarely": f"< {100 * d.lo:.0f}%",
    }
    for p in d.parts:
        row = [f"{GNAME[p.group]} ({lab[p.group]})", str(p.n)]
        for k in (1, 5):
            if p.n:
                row += [
                    V.fmt(p.est[f"a{k}"]),
                    V.fmt(p.est[f"b{k}"]),
                    V.fmt(p.est[f"d{k}"], True),
                ]
            else:
                row += ["—", "—", "—"]
            row += [V.fmt(p.con[k], True), _share(p.con[k], d.total[k])]
        md.append("| " + " | ".join(row) + " |")
    row = ["all", str(sum(p.n for p in d.parts))]
    for k in (1, 5):
        row += ["", "", "", V.fmt(d.total[k], True), "100%"]
    md.append("| " + " | ".join(row) + " |")
    return md


def _robust_table(ds: Sequence[Decomp]) -> list[str]:
    md = [
        "| grouped by | thresholds hi / lo (%) | select → measure | n (rel / some / rare) | "
        "Δ pass@1 contributions: reliable / sometimes / rarely | overall Δ pass@1 | "
        "Δ pass@5 contributions: reliable / sometimes / rarely | overall Δ pass@5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in ds:
        n = " / ".join(str(p.n) for p in d.parts)
        row = [BY[d.by], _th(d), d.split, n]
        for k in (1, 5):
            row += [
                " / ".join(V.fmt(p.con[k], True) for p in d.parts),
                V.fmt(d.total[k], True),
            ]
        md.append("| " + " | ".join(row) + " |")
    return md


def tables(
    D: Data,
    cov: Coverage,
    ds: Sequence[Decomp],
    prof: Sequence[Profile],
    gks: dict[str, list[GroupK]],
    H: Hard,
    checks: Sequence[str],
    draws: int,
    out: Path,
) -> list[str]:
    seeds = ", ".join(map(str, D.seeds))
    md = [
        f"# Charles149: where {A}'s pass@1 lead over {B} comes from",
        "",
        "Generated by `charles_overlap.py`; do not edit.",
        "",
        f"- {A}: {LABEL[PAIR.a]} (`{PAIR.a}`); {B}: {LABEL[PAIR.b]} (`{PAIR.b}`).",
        f"- Reads: plain arm at B64, 64 attempts per problem, Stage-1 seeds {seeds} (ynez), the "
        "reads of `charles_length_hist_ynez.md`. Charles149 only.",
        "- Per-problem pass@k: the mean over seeds of the unbiased pass@k; set values are the mean",
        f"  over problems, in %. Δ = {A} − {B}, in points. 95% intervals: hierarchical",
        f"  bootstrap over problems and seeds ({draws} draws, RNG seed 0), seeds paired (same",
        "  Stage-1 inits).",
        "- Cross-fit: each read's attempts split into even (0, 2, …) and odd (1, 3, …) positions,",
        "  32 per seed. Groups come from pass@5 on one half; everything in a group's row is read",
        "  on the other half, so grouping and measurement share no sampling noise. A group's",
        "  contribution is the sum over its problems of the per-problem Δ divided by 149:",
        "  contributions add up to the overall Δ of the measurement half. Group membership is held",
        "  fixed in the bootstrap (intervals are conditional on the grouping).",
        f"- Figures: `fig_charles_overlap_scatter`, `fig_charles_passk`, `fig_charles_gap_decomp` "
        f"in `{out}/`.",
        "",
        "## Checks",
        "",
        *[f"- {c}" for c in checks],
        "",
        "## 1. Coverage: pass@k for k = 1…64 (full data)",
        "",
        f"| k | {A} pass@k [95%] | {B} pass@k [95%] | Δ [95%] | p (bootstrap) |",
        "|---|---|---|---|---|",
    ]
    for k in K_REPORT:
        e = cov.d[k - 1]
        p = e.p_boot
        md.append(
            f"| {k} | {V.fmt(cov.a[k - 1])} | {V.fmt(cov.b[k - 1])} | {V.fmt(e, True)} | "
            f"{'—' if p is None else f'{p:.3f}'} |"
        )
    s = cov.solved
    md += [
        "",
        "Every k from 1 to 64, Δ in points: "
        + ", ".join(f"{k}: {100 * cov.d[k - 1].est:+.1f}" for k in KS if k in (1, 2, 3, 4, 5, 8))
        + ", "
        + ", ".join(f"{k}: {100 * cov.d[k - 1].est:+.1f}" for k in (12, 16, 24, 32, 48, 64))
        + ". The interval excludes 0 at k = "
        + _excl(cov.d)
        + ".",
        "",
        "Problems solved at least once in all 192 attempts (3 seeds × 64, pooled):",
        "",
        f"| {A} | {B} | both | either | neither | {A} only | {B} only |",
        "|---|---|---|---|---|---|---|",
        f"| {s[A]} | {s[B]} | {s['both']} | {s['either']} | {s['neither']} | {s[f'{A} only']} | "
        f"{s[f'{B} only']} |",
        "",
    ]
    for i, by in enumerate(("ei", "both")):
        d = find(ds, by, MAIN, "even → odd")
        md += [
            f"## 2{'ab'[i]}. Where the gap comes from: grouped by {BY[by]} on the even attempts",
            "",
            f"Groups at pass@5 ≥ {100 * MAIN[0]:.0f}%, {100 * MAIN[1]:.0f}–{100 * MAIN[0]:.0f}%, "
            f"< {100 * MAIN[1]:.0f}% (even attempts); every other column on the odd attempts.",
            "",
            *_main_table(d),
            "",
            *[f"- {c}." for c in sum_check(d)[0]],
            f"- The overall odd-half Δ pass@1 is {V.fmt(d.total[1], True)}, against "
            f"{V.fmt(cov.d[0], True)} on every attempt (same seeds; the halves differ only by "
            "sampling noise).",
            "",
        ]
    md += [
        "## 2c. Robustness: thresholds, halves swapped, and no cross-fit",
        "",
        "Contributions in points of the overall Δ of the measurement half [95%]. 'none (full",
        "data)' groups and measures on the same 64 attempts: it shows the selection effect the",
        "cross-fit removes (a problem whose pass@5 is high by chance lands in the 'reliable' group",
        "and carries that luck into its Δ).",
        "",
        *_robust_table(ds),
        "",
    ]
    md += [
        "## 2d. Sharpening or coverage: each group's pass@k on the odd attempts",
        "",
        "Main thresholds, cross-fit as in 2a/2b; k up to 32 (the odd half's attempts per seed).",
        "'solved' counts the group's problems with at least one success in the arm's 96 odd",
        "attempts (3 seeds × 32, pooled).",
        "",
        f"| grouped by | group | n | {A} pass@1 | {B} pass@1 | {A} pass@5 | {B} pass@5 | "
        f"{A} pass@16 | {B} pass@16 | {A} pass@32 | {B} pass@32 | Δ pass@32 | "
        f"solved {A} / {B} |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for by, rows in gks.items():
        for r in rows:
            e = r.est
            md.append(
                f"| {BY[by]} | {GNAME.get(r.group, r.group)} | {r.n} | "
                + " | ".join(V.fmt(e[f"{ab}{k}"]) for k in K_GROUP for ab in "ab")
                + f" | {V.fmt(e['d32'], True)} | {r.solved[0]} / {r.solved[1]} |"
            )
    md += [
        "",
        "## 2e. Profile: contributions by finer bins of pass@5 on the even attempts",
        "",
        "Points of the overall odd-half Δ [95%]; same draws as 2a/2b.",
        "",
        "| grouped by | pass@5 (even) | n | Δ pass@1 contribution | Δ pass@5 contribution |",
        "|---|---|---|---|---|",
    ]
    for pr in prof:
        for i, b in enumerate(FINE):
            md.append(
                f"| {BY[pr.by]} | {fine_label(b)} | {pr.n[i]} | {V.fmt(pr.con[1][i], True)} | "
                f"{V.fmt(pr.con[5][i], True)} |"
            )
        md.append(
            f"| {BY[pr.by]} | all | {sum(pr.n)} | {V.fmt(pr.total[1], True)} | "
            f"{V.fmt(pr.total[5], True)} |"
        )
    md += [""]
    x, y = D.full
    L = D.lines
    rest = ~H.core
    md += [
        "## 3. The hard core (full data)",
        "",
        f"- pass@5 < {100 * HARD:.0f}% in both arms: **{int(H.core.sum())}** problems; no success "
        f"in any of the 192 attempts of either arm: **{int(H.zero.sum())}** "
        f"({int((H.zero & H.core).sum())} of them among the {int(H.core.sum())}).",
        f"- Their share of the gap (contribution, points of the overall full-data Δ): "
        f"pass@5 < {100 * HARD:.0f}% in both: Δ pass@1 {V.fmt(H.con['core'][1], True)}, Δ pass@5 "
        f"{V.fmt(H.con['core'][5], True)}; never solved: 0 by definition. Near 0 by "
        "construction: a problem with pass@5 below 20% in both arms has pass@1 below about 5% in "
        "both, so its Δ is small.",
        f"- Cross-fit version (pass@5 < {100 * HARD:.0f}% in both arms on the even attempts: "
        f"{H.n_cross} problems; Δ on the odd ones): Δ pass@1 {V.fmt(H.cross[1], True)}, Δ pass@5 "
        f"{V.fmt(H.cross[5], True)} points.",
        "",
        "Proof length (ND lines of the search-minimal reference):",
        "",
        "| problems | n | mean | median | quartiles | range | "
        + " | ".join(f"{LSP.bin_label(b)}" for b in LSP.BINS[SET])
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(LSP.BINS[SET]),
    ]
    for lab, m in (
        (f"hard core (pass@5 < {100 * HARD:.0f}% both)", H.core),
        ("never solved by either arm", H.zero),
        ("the other problems", rest),
        ("all 149", np.ones(D.T, dtype=bool)),
    ):
        md.append(f"| {lab} | {describe(L[m])} | " + " | ".join(map(str, bin_counts(L[m]))) + " |")
    n_bin = bin_counts(L)
    md += [
        "",
        "Share of each length bin in the hard core: "
        + ", ".join(
            f"{LSP.bin_label(b)}: {c}/{n}"
            for b, c, n in zip(LSP.BINS[SET], bin_counts(L[H.core]), n_bin, strict=True)
        )
        + ".",
        "",
        f"### The reverse cases (full data): one arm's pass@5 ≥ {100 * EASY:.0f}%, the other's "
        f"< {100 * HARD:.0f}%",
        "",
        "| case | n | contribution to Δ pass@1 (points) | contribution to Δ pass@5 (points) |",
        "|---|---|---|---|",
        f"| {A} reliable, {B} rarely | {int(H.a_only.sum())} | "
        f"{V.fmt(H.con['a_only'][1], True)} | {V.fmt(H.con['a_only'][5], True)} |",
        f"| {B} reliable, {A} rarely | {int(H.b_only.sum())} | "
        f"{V.fmt(H.con['b_only'][1], True)} | {V.fmt(H.con['b_only'][5], True)} |",
        "",
        f"| case | problem | lines | {A} successes per seed (of 64) | {B} successes per seed | "
        f"{A} pass@5 | {B} pass@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    pa, pb = p_at(x, 5), p_at(y, 5)
    for lab, m in (
        (f"{A} reliable, {B} rarely", H.a_only),
        (f"{B} reliable, {A} rarely", H.b_only),
    ):
        for i in np.flatnonzero(m):
            md.append(
                f"| {lab} | `{D.names[i]}` | {L[i]:.0f} | {', '.join(map(str, x.c[:, i]))} | "
                f"{', '.join(map(str, y.c[:, i]))} | {100 * pa[i]:.0f}% | {100 * pb[i]:.0f}% |"
            )
    sa, sb = x.c.sum(0) > 0, y.c.sum(0) > 0
    md += [
        "",
        "### Solved by one arm only (at least one success in its 192 attempts, none in the "
        "other's)",
        "",
        f"| solved by | problem | lines | {A} successes per seed (of 64) | "
        f"{B} successes per seed |",
        "|---|---|---|---|---|",
    ]
    for lab, m in ((f"{A} only", sa & ~sb), (f"{B} only", sb & ~sa)):
        for i in np.flatnonzero(m):
            md.append(
                f"| {lab} | `{D.names[i]}` | {L[i]:.0f} | {', '.join(map(str, x.c[:, i]))} | "
                f"{', '.join(map(str, y.c[:, i]))} |"
            )
    md += [""]
    return md


def _excl(ds: Sequence[stats.Estimate]) -> str:
    ks = [e.k for e in ds if e.k is not None and (e.lo > 0 or e.hi < 0)]
    if not ks:
        return "no k"
    runs: list[tuple[int, int]] = []
    for k in ks:
        if runs and k == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], k)
        else:
            runs.append((k, k))
    return ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in runs)


def recipe_check(cov: Coverage) -> list[str]:
    """The full-data pass@1 and pass@5 against `charles_length_hist`'s "all" row (RECIPE.md)."""
    got = {
        "a1": V.fmt(cov.a[0]),
        "b1": V.fmt(cov.b[0]),
        "d1": V.fmt(cov.d[0], True),
        "a5": V.fmt(cov.a[4]),
        "b5": V.fmt(cov.b[4]),
        "d5": V.fmt(cov.d[4], True),
    }
    return [
        f"full data {key}: {got[key]} "
        f"({'matches' if got[key] == want else 'MISMATCH, charles_length_hist_ynez.md ' + want} "
        "`charles_length_hist_ynez.md`'s all row)"
        for key, want in CLH.RECIPE.items()
    ]


# ---------------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(V.E / "v2"), help="the v2 reads")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--table", default=TABLE)
    a = ap.parse_args(argv)
    ld = V.Loader(Path(a.v2), V.subsets())
    D = load(ld)
    cov = coverage(D, a.draws)
    ds = decompositions(D, a.draws)
    H = hard(D, a.draws)
    prof = [profile(D, by, a.draws) for by in ("ei", "both")]
    gks = {by: group_curves(D.odd, find(ds, by, MAIN, "even → odd"), a.draws) for by in BY}
    checks = recipe_check(cov)
    bad = [d for d in ds if not sum_check(d)[1]]
    checks.append(
        f"contributions add up to the overall Δ in all {len(ds)} decompositions (estimate and "
        "every draw)"
        if not bad
        else "CONTRIBUTIONS DO NOT ADD UP in "
        + "; ".join(f"{d.by} {_th(d)} {d.split}" for d in bad)
    )
    out = Path(a.out)
    fig_scatter(D, cov, find(ds, "ei", MAIN, "even → odd"), out)
    fig_passk(cov, D.seeds, out)
    fig_gap_decomp(ds, out)
    Path(a.table).write_text(
        "\n".join(tables(D, cov, ds, prof, gks, H, checks, a.draws, out)) + "\n"
    )
    print("\n".join(checks))
    print(f"wrote {a.table} and {out}/fig_charles_{{overlap_scatter,passk,gap_decomp}}.*")


if __name__ == "__main__":
    main()
