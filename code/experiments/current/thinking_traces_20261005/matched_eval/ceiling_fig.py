"""Coverage ceilings at the maximum attempt budget: thinking vs the no-think baseline, k = 512.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.ceiling_fig

Leon (2026-10-09): "what if you allow both methods their maximum budget". The round-2 seed-0 pair
(thinking at think budget 64, the no-think EI baseline at B0) read with 512 attempts per problem on
a random 250 of the 986 valid 13-20 problems (`t1320_ceil250`, seed 512; the thinking read ran in
two halves, concatenated into `t1320c_think_r2_k512`). Panel (a) is pass@k at equal attempts,
k = 1, 2, 4, ..., 512; panel (b) is pass@k at equal sampled tokens per problem (`stats.at_tokens`,
drawn only inside each read's sampled range). Bands are 95% bootstrap intervals over problems (one
training seed per arm). Writes fig_ceiling (PDF/SVG/PNG) and a JSON of the plotted values. CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from code.experiments.current.thinking_traces_20261005.matched_eval import tts_figs as T
from code.experiments.current.thinking_traces_20261005.matched_eval.figs import valid_1320
from code.experiments.current.thinking_traces_20261005.matched_eval.tts_pooled import BASE, THINK
from code.ndbench.figures import style as S
from code.thinking import stats

CEIL = T.E / "ceil"
ROWS = {
    "think": CEIL / "t1320c_think_r2_k512" / "rows_plain_B64.jsonl",
    "base": CEIL / "t1320c_nothink_r2_k512" / "rows_plain_B0.jsonl",
}
LOOKS = {"think": THINK, "base": BASE}
KS = [2**i for i in range(10)]  # 1 .. 512 attempts per problem
X_TOK = [float(x) for x in np.geomspace(250, 480_000, 61)]
CHECK_K = (1, 8, 32, 64, 128, 256, 512)
CHECK_X = (1000.0, 4000.0, 16000.0, 32000.0, 64000.0, 128000.0, 150000.0)


def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def _pp(e: stats.Estimate) -> str:
    return f"{100 * e.est:+.1f} [{100 * e.lo:+.1f}, {100 * e.hi:+.1f}]"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--draws", type=int, default=T.DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    arms = {k: stats.load_arm(k, [str(p)]) for k, p in ROWS.items()}
    keep = valid_1320()
    for k, arm in arms.items():
        bad = [n for n in arm.names if n not in keep]
        assert not bad, f"{k}: {len(bad)} problems outside the valid 13-20 items, e.g. {bad[:3]}"
        assert arm.T == 250 and arm.kmax == 512, (k, arm.T, arm.kmax)
    assert arms["think"].names == arms["base"].names
    tpa = {k: float(arm.tokens_per_attempt()[0]) for k, arm in arms.items()}
    by_k = {k: stats.ci_curve(arm, KS, draws=a.draws) for k, arm in arms.items()}
    # each curve starts at one attempt's tokens (nudged up so k* = 1 is not read as clamped)
    xs = sorted(set(X_TOK) | {v * 1.0001 for v in tpa.values()})
    by_x = {k: stats.at_tokens(arm, xs, draws=a.draws) for k, arm in arms.items()}

    print(f"tokens per attempt: thinking {tpa['think']:.0f}, baseline {tpa['base']:.0f}")
    for k in CHECK_K:
        e = {n: by_k[n][KS.index(k)] if k in KS else stats.ci(arms[n], k) for n in arms}
        d = stats.paired_diff(arms["think"], arms["base"], k, draws=a.draws)
        print(
            f"k {k:>3}: thinking {_pct(e['think'].est)} [{_pct(e['think'].lo)}, "
            f"{_pct(e['think'].hi)}]  baseline {_pct(e['base'].est)} [{_pct(e['base'].lo)}, "
            f"{_pct(e['base'].hi)}]  diff {_pp(d)}"
        )
    diffs = stats.paired_diff_at_tokens(arms["think"], arms["base"], CHECK_X, draws=a.draws)
    for x, d in zip(CHECK_X, diffs, strict=True):
        flag = "" if d.in_range else " (out of range)"
        print(f"{x / 1000:>5g}k tokens: diff {_pp(d)}{flag}")
    k512 = {n: arms[n].values(512) > 0 for n in arms}  # per problem: solved within 512 attempts
    only_t = int((k512["think"] & ~k512["base"]).sum())
    only_b = int((k512["base"] & ~k512["think"]).sum())
    both = int((k512["think"] & k512["base"]).sum())
    print(f"at 512: only thinking {only_t}, only baseline {only_b}, both {both}")

    draw(by_k, by_x, xs, tpa, out / "fig_ceiling")
    dump: dict[str, Any] = {
        "rows": {k: str(p) for k, p in ROWS.items()},
        "n_problems": 250,
        "tokens_per_attempt": tpa,
        "equal_attempts": {
            n: [{"k": k, "est": e.est, "lo": e.lo, "hi": e.hi} for k, e in zip(KS, es, strict=True)]
            for n, es in by_k.items()
        },
        "equal_tokens": {
            n: [
                {"tokens": e.tokens, "est": e.est, "lo": e.lo, "hi": e.hi, "in_range": e.in_range}
                for e in es
            ]
            for n, es in by_x.items()
        },
        "solved_at_512": {"only_thinking": only_t, "only_baseline": only_b, "both": both},
    }
    (out / "fig_ceiling.json").write_text(json.dumps(dump, indent=1) + "\n")
    print(f"wrote {out / 'fig_ceiling'}.{{pdf,svg,png,json}}")


def _band(
    ax: Any, x: Sequence[float], es: Sequence[stats.Estimate], look: T.Look, mk: bool
) -> None:
    ok = [e.in_range is not False for e in es]
    xs = [v for v, o in zip(x, ok, strict=True) if o]
    ee = [e for e, o in zip(es, ok, strict=True) if o]
    ax.fill_between(
        xs, [e.lo for e in ee], [e.hi for e in ee], color=look.color, alpha=look.band, lw=0
    )
    kw: dict[str, Any] = {}
    if mk:
        kw = {"marker": look.marker, "ms": 3.0, "mfc": look.color, "mec": S.WHITE, "mew": 0.5}
    ax.plot(xs, [e.est for e in ee], color=look.color, ls=look.ls, lw=look.lw, **kw)


def draw(
    by_k: dict[str, list[stats.Estimate]],
    by_x: dict[str, list[stats.Estimate]],
    xs: Sequence[float],
    tpa: dict[str, float],
    base: Path,
) -> None:
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35, layout="constrained")
        ax_k, ax_x = fig.subplots(1, 2, sharey=True)
        for n in ("base", "think"):
            look = LOOKS[n]
            _band(ax_k, [float(k) for k in KS], by_k[n], look, mk=True)
            _band(ax_x, xs, by_x[n], look, mk=False)
            ax_x.plot(
                [tpa[n]],
                [by_k[n][0].est],
                ls="none",
                marker=look.marker,
                ms=3.4,
                mfc=look.color,
                mec=S.WHITE,
                mew=0.5,
                zorder=4,
            )
        # direct labels for the 512-attempt ceilings (text in ink, the marker carries identity)
        for n, dy in (("think", 4), ("base", -9)):
            e = by_k[n][-1]
            ax_k.annotate(
                f"{100 * e.est:.0f}%",
                (512, e.est),
                xytext=(5, dy),
                textcoords="offset points",
                fontsize=S.FS_TICK,
                color=S.INK2,
                va="center",
            )
        ax_k.set_xscale("log", base=2)
        ax_k.set_xlim(0.8, 900)
        ax_k.xaxis.set_major_locator(FixedLocator([1, 8, 64, 512]))
        ax_k.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax_k.xaxis.set_minor_formatter(NullFormatter())
        ax_k.set_xlabel("attempts per problem (log scale)")
        ax_x.set_xscale("log")
        ax_x.set_xlim(220, 6.0e5)
        ax_x.xaxis.set_major_locator(FixedLocator([1000, 10000, 100000]))
        ax_x.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:g}k"))
        ax_x.xaxis.set_minor_formatter(NullFormatter())
        ax_x.set_xlabel("tokens generated per problem (log scale)")
        for ax in (ax_k, ax_x):
            T.pct_axis(ax)
        ax_k.set_ylim(0, 1.0)
        S.panel_title(ax_k, "a", "equal attempts (up to 512 each)")
        S.panel_title(ax_x, "b", "equal tokens generated")
        ax_k.set_ylabel("problems solved (250 of 13–20 lines)")
        handles = [
            Line2D([], [], color=c.color, ls=c.ls, lw=c.lw, marker=c.marker, ms=3.4, label=c.label)
            for c in (THINK, BASE)
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        S.save(fig, base)
        S.close(fig)


if __name__ == "__main__":
    main()
