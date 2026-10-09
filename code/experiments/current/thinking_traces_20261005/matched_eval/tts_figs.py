"""Test-time-scaling uplift from synthetic thinking traces at matched inference compute.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.tts_figs \
        [--out experiment-summaries/2026-10-05-thinking-traces/figures] \
        [--table code/experiments/current/thinking_traces_20261005/matched_eval/\
tts_uplift_tables.md] [--draws 4000] [--verbose]

The question (Leon, 2026-10-09): does the small prover trained with synthetic thinking traces get
test-time-scaling uplift over a no-think baseline at matched inference compute? Each pair differs
only in the traces. "Thinking" is Stage-1 S1g3 (honest-search thinking traces with effort tokens)
and "baseline" is Stage-1 T0S (the same proofs without think blocks), each followed by the same
expert-iteration (EI) rounds: r1 on rl_hard12, r2 on a 5k sample (a) of hard712, r3 on a disjoint
5k sample (b). The thinking model is read at think budget 64 per step (B64; k = 32 attempts per
problem on 13-20 and hard712 dev, 64 on textbook72), the baseline at B0 (k = 64).

Metric: pass@k, which is "solved within k attempts" because Lean verifies every proof, at matched
sampled tokens per problem. `stats.at_tokens` reads pass@k* at k* = X / (sampled tokens per
attempt), interpolated linearly in log k, and `stats.paired_diff_at_tokens` gives thinking -
baseline at each X. Intervals are 95% percentile intervals of 4,000 bootstrap draws over problems;
each arm is one training seed, so seed variance is not in them. A point where X falls outside an
arm's sampled range (under 1 attempt or over k attempts) is not drawn, and is flagged in the tables.

Where the thinking model was also read at B0 (k = 64), it is a third curve, and the upper envelope
of its B0 and B64 curves is "the thinking model with the best budget per compute level". Below one
B64 attempt only B0 is possible, so the envelope is B0 there; above the B0 read's 64 attempts it is
not drawn. The envelope takes the maximum on every bootstrap draw (same problem resample for all
arms), so its interval includes the budget choice; the choice is still made after seeing this
set's results (two options), a mild optimistic bias.

Outputs: fig_tts_uplift (13-20, the 986 valid items of `figs.valid_1320`) and
fig_tts_uplift_other (textbook72, hard712 dev) as PDF/SVG/PNG plus a JSON of every plotted value;
and the results tables (markdown) behind THINKING_UPLIFT.md, with a sensitivity table that also
counts prompt prefill tokens as compute ("forward tokens"). Pairs whose rows are not there yet
(seed-1 round 3, the B0 reads) are skipped and picked up on the next run. CPU only.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import valid_1320
from code.ndbench.figures import style as S
from code.thinking import stats
from code.thinking.stats import F64, Bool

E = Path("/scratch/cowifornia/nd-rl/think/runs/eval")
TTS, MATCHED = E / "tts", E / "matched"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"
DRAWS = 4000
GRID = [float(x) for x in np.geomspace(300, 20000, 49)]  # sampled tokens per problem
TABLE_X = (1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
FWD_X = (6000.0, 12000.0, 24000.0, 48000.0, 96000.0)  # forward tokens (sensitivity table)
X_TICKS = (300, 1000, 3000, 10000)
BAND = 0.18
HL_LW, HL_ALPHA = 3.6, 0.3  # the best-budget highlight stroke
DASH: Any = (0, (3.5, 1.6))
DOT: Any = (0, (1.0, 1.3))
DASHDOT: Any = (0, (4.0, 1.4, 1.0, 1.4))
_verbose = False


def _rows(d: Path, budget: int) -> Path:
    return d / f"rows_plain_B{budget}.jsonl"


@dataclass(frozen=True)
class Pair:
    """One thinking/baseline pair: the thinking model at B64, the baseline at B0, and optionally
    the thinking model at B0. `optional` pairs are skipped while their rows are missing."""

    key: str
    set: str  # "t1320", "tb72" or "h712"
    title: str
    think: Path
    base: Path
    think_b0: Path | None = None
    optional: bool = False


T1320 = [
    Pair(
        "r1 s0",
        "t1320",
        "round 1, seed 0",
        _rows(TTS / "t1320_think_r1_k32", 64),
        _rows(TTS / "t1320_nothink_r1_k64", 0),
    ),
    Pair(
        "r2 s0",
        "t1320",
        "round 2, seed 0",
        _rows(TTS / "t1320_think_r2_k32", 64),
        _rows(TTS / "t1320_nothink_r2_k64", 0),
        _rows(TTS / "t1320_thinkB0_r2_k64", 0),
    ),
    Pair(
        "r2 s1",
        "t1320",
        "round 2, seed 1",
        _rows(TTS / "t1320_think_r2s1_k32", 64),
        _rows(TTS / "t1320_nothink_r2s1_k64", 0),
        _rows(TTS / "t1320_thinkB0_r2s1_k64", 0),
    ),
    Pair(
        "r3 s0",
        "t1320",
        "round 3, seed 0",
        _rows(TTS / "t1320_think_r3_k32", 64),
        _rows(TTS / "t1320_nothink_r3_k64", 0),
    ),
    Pair(
        "r3 s1",
        "t1320",
        "round 3, seed 1",
        _rows(TTS / "t1320_think_r3s1_k32", 64),
        _rows(TTS / "t1320_nothink_r3s1_k64", 0),
        optional=True,
    ),
]

OTHER = [
    Pair(
        "r2 s0",
        "tb72",
        "textbook72, round 2, seed 0",
        _rows(MATCHED / "tb72_eiS1g3_h712_r2", 64),
        _rows(MATCHED / "tb72_eiT0S_h712_r2", 0),
        _rows(MATCHED / "tb72_eiS1g3_h712_r2", 0),
    ),
    Pair(
        "r2 s1",
        "tb72",
        "textbook72, round 2, seed 1",
        _rows(TTS / "tb72_think_r2s1_k64", 64),
        _rows(TTS / "tb72_nothink_r2s1_k64", 0),
    ),
    Pair(
        "r3 s1",
        "tb72",
        "textbook72, round 3, seed 1",
        _rows(TTS / "tb72_think_r3s1_k64", 64),
        _rows(TTS / "tb72_nothink_r3s1_k64", 0),
        optional=True,
    ),
    Pair(
        "r2 s0",
        "h712",
        "hard712 dev, round 2, seed 0",
        _rows(TTS / "h712dev_think_r2_k32", 64),
        _rows(TTS / "h712dev_nothink_r2_k64", 0),
    ),
    Pair(
        "r3 s0",
        "h712",
        "hard712 dev, round 3, seed 0",
        _rows(TTS / "h712dev_think_r3_k32", 64),
        _rows(TTS / "h712dev_nothink_r3_k64", 0),
    ),
]

SET_NAMES = {
    "t1320": "13–20-line proofs (986)",
    "tb72": "textbook72 (72)",
    "h712": "hard712 dev (500)",
}


# ------------------------------------------------------------------------------------- loading


class Loader:
    """Rows files as one-seed arms (cached). 13-20 rows are filtered to the valid items; with
    cost "forward" each attempt's prompt prefill tokens are added to its sampled tokens. Rewritten
    copies keep the `rows_plain_B*.jsonl` name that `stats.load_arm` recognises."""

    def __init__(self, tmp: Path) -> None:
        self.tmp, self.n = tmp, 0
        self.keep = valid_1320()
        self._arms: dict[tuple[str, bool, str], stats.Arm] = {}

    def arm(self, path: Path, *, filt: bool, cost: str = "sampled") -> stats.Arm:
        key = (str(path), filt, cost)
        if key not in self._arms:
            src = path
            if filt or cost != "sampled":
                self.n += 1
                d = self.tmp / str(self.n)
                d.mkdir()
                src = d / path.name
                with open(path) as f, open(src, "w") as g:
                    for line in f:
                        r = json.loads(line)
                        if filt and r["name"] not in self.keep:
                            continue
                        if cost == "forward":
                            r["tokens"] = [
                                t + p for t, p in zip(r["tokens"], r["prefill"], strict=True)
                            ]
                        g.write(json.dumps(r) + "\n")
            self._arms[key] = stats.load_arm(path.parent.name, [str(src)])
        return self._arms[key]


# ------------------------------------------------------------------------------------ analysis


@dataclass
class Curve:
    """Values at token targets `x`: point estimate, 95% interval, in range, and bootstrap draws
    (draws x len(x)), all as fractions."""

    x: list[float]
    est: F64
    lo: F64
    hi: F64
    ok: Bool
    samples: F64
    pick_b0: Bool | None = None  # envelope only: where the point estimate takes B0

    def at(self, x: float) -> tuple[float, float, float, bool]:
        j = self.x.index(x)
        return float(self.est[j]), float(self.lo[j]), float(self.hi[j]), bool(self.ok[j])

    def as_dict(self) -> dict[str, Any]:
        d = {
            "x": self.x,
            "est": self.est.tolist(),
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "in_range": self.ok.tolist(),
        }
        if self.pick_b0 is not None:
            d["budget_0_chosen"] = self.pick_b0.tolist()
        return d


def _from_samples(x: list[float], est: F64, samples: F64, ok: Bool) -> Curve:
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    return Curve(x, est, lo, hi, ok, samples)


def tokens_curve(arm: stats.Arm, x: list[float], draws: int) -> Curve:
    es = stats.at_tokens(arm, x, draws=draws)
    return Curve(
        x,
        np.array([e.est for e in es]),
        np.array([e.lo for e in es]),
        np.array([e.hi for e in es]),
        np.array([e.in_range for e in es]),
        np.stack([np.asarray(e.samples) for e in es], axis=1),
    )


def envelope(b64: Curve, b0: Curve, tpa64: float) -> Curve:
    """Best of B0 and B64 at each target, per bootstrap draw. Below one B64 attempt it is B0;
    beyond B0's sampled range it is out of range."""
    x = np.array(b64.x)
    both = b64.ok & b0.ok
    b0_only = b0.ok & ~b64.ok & (x < tpa64)
    est = np.where(b0_only, b0.est, np.maximum(b64.est, b0.est))
    samples = np.where(b0_only[None, :], b0.samples, np.maximum(b64.samples, b0.samples))
    env = _from_samples(b64.x, est, samples, both | b0_only)
    env.pick_b0 = b0_only | (both & (b0.est > b64.est))
    return env


def difference(a: Curve, b: Curve) -> Curve:
    """a - b on the shared bootstrap draws (every `at_tokens` call with the same number of problems
    and draws resamples the same problems)."""
    d = _from_samples(a.x, a.est - b.est, a.samples - b.samples, a.ok & b.ok)
    d.pick_b0 = a.pick_b0
    return d


@dataclass
class Result:
    pair: Pair
    arms: dict[str, stats.Arm]
    curves: dict[str, Curve]
    diffs: dict[str, Curve]
    pass1: dict[str, stats.Estimate]
    pass1_diff: dict[str, stats.Estimate]
    tpa: dict[str, float] = field(default_factory=lambda: dict[str, float]())


def analyse(pair: Pair, loader: Loader, draws: int, *, cost: str = "sampled") -> Result | None:
    table_x = TABLE_X if cost == "sampled" else FWD_X
    paths = {"think": pair.think, "base": pair.base}
    if pair.think_b0 is not None and pair.think_b0.exists():
        paths["b0"] = pair.think_b0
    missing = [p for p in (pair.think, pair.base) if not p.exists()]
    if missing:
        if pair.optional:
            return None
        raise FileNotFoundError(f"{pair.set} {pair.key}: missing {missing}")
    filt = pair.set == "t1320"
    arms = {k: loader.arm(p, filt=filt, cost=cost) for k, p in paths.items()}
    x = sorted(set(GRID) | set(table_x))
    curves = {k: tokens_curve(a, x, draws) for k, a in arms.items()}
    tpa = {k: float(a.tokens_per_attempt()[0]) for k, a in arms.items()}
    diffs = {"think": difference(curves["think"], curves["base"])}
    # the stats module's own paired difference must agree with the shared-draw difference
    check = stats.paired_diff_at_tokens(arms["think"], arms["base"], list(table_x), draws=draws)
    for e in check:
        assert e.tokens is not None
        est, lo, hi, ok = diffs["think"].at(e.tokens)
        assert abs(est - e.est) < 1e-9 and abs(lo - e.lo) < 1e-9 and abs(hi - e.hi) < 1e-9
        assert ok == e.in_range
    pass1 = {k: stats.ci(a, 1, draws=draws) for k, a in arms.items()}
    pass1_diff = {"think": stats.paired_diff(arms["think"], arms["base"], 1, draws=draws)}
    if "b0" in arms:
        curves["env"] = envelope(curves["think"], curves["b0"], tpa["think"])
        diffs["env"] = difference(curves["env"], curves["base"])
        diffs["b0"] = difference(curves["b0"], curves["base"])
        pass1_diff["b0"] = stats.paired_diff(arms["b0"], arms["base"], 1, draws=draws)
    return Result(pair, arms, curves, diffs, pass1, pass1_diff, tpa)


def log_result(r: Result) -> None:
    if not _verbose:
        return
    p = r.pair
    for k, a in r.arms.items():
        e = r.pass1[k]
        print(
            f"{p.set} {p.key} {k:5s} T={a.T} k={a.kmax} tokens/attempt {r.tpa[k]:.0f} "
            f"pass@1 {100 * e.est:.1f} [{100 * e.lo:.1f}, {100 * e.hi:.1f}] "
            f"pass@{a.kmax} {100 * a.values(None).mean():.1f}"
        )
    for k, c in {**r.curves, **{f"diff {k}": d for k, d in r.diffs.items()}}.items():
        row = "  ".join(
            f"{x / 1000:g}k {100 * c.at(x)[0]:+.1f}{'' if c.at(x)[3] else '*'}" for x in TABLE_X
        )
        print(f"  {k:10s} {row}")


# ---------------------------------------------------------------------------------- the figures


@dataclass(frozen=True)
class Look:
    label: str
    color: str
    ls: Any
    marker: str | None = None
    lw: float = 1.3
    band: float = BAND


THINK = Look("thinking traces (64-token budget)", BLUE, "-", "o")
BASE = Look("no-think baseline (same data)", ORANGE, DASH, "s")
THINK_B0 = Look("thinking-trace model, thinking off (budget 0)", AQUA, DOT, "D", band=0.11)
ENV = Look("thinking-trace model, best budget per compute level", GRAY, DASHDOT)


def _runs(mask: Bool) -> list[slice]:
    """Maximal runs of True in `mask`."""
    out, start = [], None
    for i, m in enumerate([*mask.tolist(), False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append(slice(start, i))
            start = None
    return out


def band_line(ax: Axes, c: Curve, look: Look, *, mask: Bool | None = None, z: float = 2) -> None:
    """In-range points only (and only where `mask`), with the 95% band."""
    m = c.ok if mask is None else c.ok & mask
    x = np.array(c.x)
    for r in _runs(m):
        ax.fill_between(
            x[r], c.lo[r], c.hi[r], color=look.color, alpha=look.band, lw=0, zorder=z - 0.5
        )
        ax.plot(x[r], c.est[r], color=look.color, ls=look.ls, lw=look.lw, zorder=z)


def highlight(ax: Axes, c: Curve, *, mask: Bool | None = None, band: bool = False) -> None:
    """The best-budget envelope as a wide light stroke, one stroke per run of the same chosen
    budget, so a switch from B0 to B64 is not drawn as a vertical jump."""
    assert c.pick_b0 is not None
    x = np.array(c.x)
    m = c.ok if mask is None else c.ok & mask
    for pick in (c.pick_b0, ~c.pick_b0):
        for r in _runs(m & pick):
            if band:
                ax.fill_between(x[r], c.lo[r], c.hi[r], color=GRAY, alpha=BAND, lw=0, zorder=1)
            ax.plot(
                x[r],
                c.est[r],
                color=GRAY,
                lw=HL_LW,
                alpha=HL_ALPHA,
                solid_capstyle="butt",
                zorder=1,
            )


def start_marker(ax: Axes, r: Result, key: str, look: Look) -> None:
    """The k = 1 point: one attempt's tokens and pass@1 (where the curve starts)."""
    if look.marker is None:
        return
    ax.plot(
        [r.tpa[key]],
        [r.pass1[key].est],
        ls="none",
        marker=look.marker,
        ms=3.4,
        mfc=look.color,
        mec=S.WHITE,
        mew=0.5,
        zorder=4,
    )


def tokens_axis(ax: Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(GRID[0] / 1.08, GRID[-1] * 1.08)
    ax.xaxis.set_major_locator(FixedLocator(list(X_TICKS)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    )
    ax.xaxis.set_minor_formatter(NullFormatter())


def _pp(v: float, _pos: int | None = None) -> str:
    if abs(v) < 1e-9:
        return "0"
    return f"{100 * v:+.0f}".replace("-", "−")


def pp_axis(ax: Axes) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(_pp))
    S.light_grid(ax)


def pct_axis(ax: Axes) -> None:
    S.pct_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    S.light_grid(ax)


def draw(results: Sequence[Result], out: Path, name: str, *, height: float) -> None:
    n = len(results)
    letters = "abcdefghij"
    has_b0 = any("b0" in r.curves for r in results)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, height, layout="constrained")
        axs = fig.subplots(
            2, n, sharex=True, sharey="row", height_ratios=[1.25, 1.0], squeeze=False
        )
        for j, r in enumerate(results):
            top, bot = axs[0][j], axs[1][j]
            if "env" in r.curves:
                highlight(top, r.curves["env"])
            for key, look in (("base", BASE), ("b0", THINK_B0), ("think", THINK)):
                if key in r.curves:
                    band_line(top, r.curves[key], look, z=3 if key == "think" else 2)
                    start_marker(top, r, key, look)
            bot.axhline(0, color=S.INK, lw=0.6, zorder=1.5)
            if "env" in r.diffs:
                # where the best budget is B64 the envelope difference is the blue line itself
                env = r.diffs["env"]
                assert env.pick_b0 is not None
                highlight(bot, env, mask=env.pick_b0, band=True)
            band_line(bot, r.diffs["think"], THINK, z=3)
            title = r.pair.title if r.pair.set == "t1320" else r.pair.title.replace(", ", "\n", 1)
            S.panel_title(top, letters[j], title)
            for ax in (top, bot):
                tokens_axis(ax)
            pct_axis(top)
            pp_axis(bot)
        top_hi = max(float(c.hi[c.ok].max()) for r in results for c in r.curves.values())
        axs[0][0].set_ylim(0, min(1.0, np.ceil(top_hi * 10 + 0.2) / 10))
        shown = [
            (d, d.ok & (d.pick_b0 if k == "env" and d.pick_b0 is not None else True))
            for r in results
            for k, d in r.diffs.items()
            if k != "b0"
        ]
        lo = min(float(d.lo[m].min()) for d, m in shown if m.any())
        hi = max(float(d.hi[m].max()) for d, m in shown if m.any())
        axs[1][0].set_ylim(min(lo, -0.02) - 0.01, max(hi, 0.02) + 0.01)
        axs[0][0].set_ylabel("pass@k at matched tokens")
        axs[1][0].set_ylabel("thinking − baseline (pp)")
        fig.supxlabel("sampled tokens per problem (log scale)", fontsize=S.FS_LABEL)
        handles = [
            Line2D([], [], color=c.color, ls=c.ls, lw=c.lw, label=c.label) for c in (THINK, BASE)
        ]
        if has_b0:
            handles += [
                Line2D([], [], color=AQUA, ls=DOT, lw=1.3, label=THINK_B0.label),
                Line2D([], [], color=GRAY, lw=HL_LW, alpha=HL_ALPHA, label=ENV.label),
            ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        S.save(fig, out / name)
        S.close(fig)


# ------------------------------------------------------------------------------------- tables


def _num(v: float) -> str:
    return f"{100 * v:+.1f}".replace("-", "−")


def _cell(est: float, lo: float, hi: float, ok: bool = True) -> str:
    return f"{_num(est)} [{_num(lo)}, {_num(hi)}]" + ("" if ok else " ‡")


def _diff_cells(c: Curve, xs: Sequence[float] = TABLE_X) -> list[str]:
    return [_cell(*c.at(x)) for x in xs]


def _oor_note(results: Sequence[Result], xs: Sequence[float] = TABLE_X) -> list[str]:
    """Which arm is out of range at the flagged targets."""
    notes = []
    for r in results:
        for x in xs:
            for key, c in r.curves.items():
                if key == "env" or c.at(x)[3]:
                    continue
                a = r.arms[key]
                side = "above" if x > a.kmax * r.tpa[key] else "below"
                notes.append(
                    f"{SET_NAMES[r.pair.set]} {r.pair.key}, {x / 1000:g}k: {key} is {side} its "
                    f"sampled range ({r.tpa[key]:.0f}–{a.kmax * r.tpa[key]:.0f} tokens)"
                )
    return notes


def tables(res: Sequence[Result], sens: Sequence[Result], draws: int) -> str:
    xs = " | ".join(f"{x / 1000:g}k" for x in TABLE_X)
    lines = [
        "<!-- generated by matched_eval/tts_figs.py; do not edit by hand -->",
        "",
        "### Arms",
        "",
        f"pass@1 with 95% bootstrap intervals over problems ({draws} draws, one seed per arm).",
        "",
        "| set | pair | arm | budget | k | tokens / attempt | pass@1 | pass@k (all k) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {"think": "thinking", "base": "baseline", "b0": "thinking"}
    budget = {"think": 64, "base": 0, "b0": 0}
    for r in res:
        for k, a in r.arms.items():
            e = r.pass1[k]
            lines.append(
                f"| {SET_NAMES[r.pair.set]} | {r.pair.key} | {names[k]} | {budget[k]} | {a.kmax} | "
                f"{r.tpa[k]:.0f} | {100 * e.est:.1f}% [{100 * e.lo:.1f}, {100 * e.hi:.1f}] | "
                f"{100 * a.values(None).mean():.1f}% |"
            )
    lines += [
        "",
        "### Thinking − baseline: per attempt and at matched sampled tokens per problem (pp)",
        "",
        "Per attempt: pass@1 difference. Matched tokens: pass@k difference at X sampled tokens per "
        "problem (think + `</think>` + action + rejected redraws; prompt prefill excluded). "
        "‡ = X outside an arm's sampled range (value clamped to pass@1 or pass@k; see notes).",
        "",
        f"| set | pair | comparison | tokens ratio | per attempt | {xs} |",
        "|---|---|---|---|---|" + "---|" * len(TABLE_X),
    ]
    for r in res:
        rows = [("think", "thinking (B64) − baseline")]
        if "env" in r.diffs:
            rows += [("env", "best budget − baseline"), ("b0", "thinking at B0 − baseline")]
        for k, label in rows:
            if k == "env":
                p1 = "–"
                ratio = "–"
            else:
                e = r.pass1_diff[k]
                p1 = _cell(e.est, e.lo, e.hi)
                ratio = f"{r.tpa[k] / r.tpa['base']:.2f}×"
            lines.append(
                f"| {SET_NAMES[r.pair.set]} | {r.pair.key} | {label} | {ratio} | {p1} | "
                + " | ".join(_diff_cells(r.diffs[k]))
                + " |"
            )
    notes = _oor_note(res)
    if notes:
        lines += ["", "Out-of-range notes (‡):", ""] + [f"- {n}" for n in notes]
    lines += [
        "",
        "### Sensitivity: compute = forward tokens (sampled + prompt prefill per attempt)",
        "",
        "Each environment step re-prefills the state prompt, so prefill is most of the forward "
        "tokens. Same pairs, draws and items; only the cost per attempt changes.",
        "",
        "| set | pair | comparison | forward tokens ratio | "
        + " | ".join(f"{x / 1000:g}k" for x in FWD_X)
        + " |",
        "|---|---|---|---|" + "---|" * len(TABLE_X),
    ]
    for r in sens:
        lines.append(
            f"| {SET_NAMES[r.pair.set]} | {r.pair.key} | thinking (B64) − baseline | "
            f"{r.tpa['think'] / r.tpa['base']:.2f}× | "
            + " | ".join(_diff_cells(r.diffs["think"], FWD_X))
            + " |"
        )
    notes = _oor_note(sens, FWD_X)
    if notes:
        lines += ["", "Out-of-range notes (‡), forward tokens:", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def dump(results: Sequence[Result]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        out.append(
            {
                "set": r.pair.set,
                "pair": r.pair.key,
                "paths": {
                    k: str(p)
                    for k, p in (
                        ("think", r.pair.think),
                        ("base", r.pair.base),
                        ("b0", r.pair.think_b0),
                    )
                    if p is not None and k in r.arms
                },
                "n_problems": r.arms["think"].T,
                "k": {k: a.kmax for k, a in r.arms.items()},
                "tokens_per_attempt": r.tpa,
                "pass1": {k: e.as_dict() for k, e in r.pass1.items()},
                "pass1_diff": {k: e.as_dict() for k, e in r.pass1_diff.items()},
                "curves": {k: c.as_dict() for k, c in r.curves.items()},
                "diffs": {k: c.as_dict() for k, c in r.diffs.items()},
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    global _verbose
    here = Path("code/experiments/current/thinking_traces_20261005/matched_eval")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--table", default=str(here / "tts_uplift_tables.md"))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--verbose", action="store_true", help="print every arm and table value")
    a = ap.parse_args(argv)
    _verbose = a.verbose
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as td:
        loader = Loader(Path(td))
        res: dict[str, list[Result]] = {}
        sens: list[Result] = []
        for name, pairs in (("fig_tts_uplift", T1320), ("fig_tts_uplift_other", OTHER)):
            res[name] = []
            for p in pairs:
                r = analyse(p, loader, a.draws)
                if r is None:
                    print(f"skipped {p.set} {p.key} (rows not there yet)")
                    continue
                log_result(r)
                res[name].append(r)
                s = analyse(p, loader, a.draws, cost="forward")
                assert s is not None
                sens.append(s)
        draw(res["fig_tts_uplift"], out, "fig_tts_uplift", height=3.5)
        draw(res["fig_tts_uplift_other"], out, "fig_tts_uplift_other", height=3.6)
        for name, rs in res.items():
            (out / f"{name}.json").write_text(json.dumps(dump(rs), indent=1) + "\n")
        allres = res["fig_tts_uplift"] + res["fig_tts_uplift_other"]
        Path(a.table).write_text(tables(allres, sens, a.draws))
    print(f"wrote figures to {out} and tables to {a.table}")


if __name__ == "__main__":
    main()
