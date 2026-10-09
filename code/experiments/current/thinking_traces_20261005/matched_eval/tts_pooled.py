"""Thinking vs no-think baseline at matched sampled tokens on 13-20, averaged over Stage-1 seeds.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.tts_pooled

The meeting version of `fig_tts_uplift` (Leon, 2026-10-09: "just the first row, not split by
seed"). One panel per EI round that has more than one seed pair; each curve is the mean over seeds
of `tts_figs.analyse`'s pass@k-at-matched-tokens curve. Every `at_tokens` call with the same number
of problems and draws resamples the same problems, so the seeds' bootstrap draws are averaged draw
by draw. The 95% bands are over problems only (two seeds cannot give a seed interval), and a point
is drawn only where every seed's read covers it. Round 1 has one seed and is left out.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from code.experiments.current.thinking_traces_20261005.matched_eval import tts_figs as T
from code.ndbench.figures import style as S

ROUNDS = {"r2": "after EI round 2", "r3": "after EI round 3"}
THINK = T.Look("thinking traces, then EI", T.BLUE, "-", "o")
BASE = T.Look("no thinking, then the same EI (baseline)", T.ORANGE, T.DASH, "s")


def mean_curve(cs: Sequence[T.Curve]) -> T.Curve:
    ok = np.logical_and.reduce([c.ok for c in cs])
    est = np.mean([c.est for c in cs], axis=0)
    samples = np.mean([c.samples for c in cs], axis=0)
    return T._from_samples(cs[0].x, est, samples, ok)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--draws", type=int, default=T.DRAWS)
    ap.add_argument("--set", choices=("t1320", "tb72"), default="t1320")
    a = ap.parse_args(argv)
    out = Path(a.out)
    pooled: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory() as td:
        loader = T.Loader(Path(td))
        for rnd in ROUNDS:
            pairs = T.T1320 if a.set == "t1320" else [p for p in T.OTHER if p.set == a.set]
            rs = [T.analyse(p, loader, a.draws) for p in pairs if p.key.startswith(rnd)]
            rs = [r for r in rs if r is not None]
            if not rs:
                print(f"skipped {rnd}: {len(rs)} seed pair(s)")
                continue
            curves = {k: mean_curve([r.curves[k] for r in rs]) for k in ("think", "base")}
            diff = T.difference(curves["think"], curves["base"])
            pooled[rnd] = {
                "seeds": [r.pair.key for r in rs],
                "curves": curves,
                "diff": diff,
                "tpa": {k: float(np.mean([r.tpa[k] for r in rs])) for k in ("think", "base")},
                "pass1": {
                    k: float(np.mean([r.pass1[k].est for r in rs])) for k in ("think", "base")
                },
            }
            print(f"{rnd} ({', '.join(str(s) for s in pooled[rnd]['seeds'])})")
            for x in T.TABLE_X:
                th, ba = curves["think"].at(x), curves["base"].at(x)
                d = diff.at(x)
                flag = "" if d[3] else " (out of range)"
                print(
                    f"  {x / 1000:>4g}k  think {100 * th[0]:.1f}  base {100 * ba[0]:.1f}"
                    f"  diff {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]{flag}"
                )
    name = "fig_tts_pooled" if a.set == "t1320" else f"fig_tts_pooled_{a.set}"
    draw(pooled, out / name, a.set)
    dump = {
        rnd: {
            "seeds": p["seeds"],
            "tokens_per_attempt": p["tpa"],
            "pass1": p["pass1"],
            "curves": {k: c.as_dict() for k, c in p["curves"].items()},  # type: ignore[union-attr]
            "diff": p["diff"].as_dict(),  # type: ignore[union-attr]
        }
        for rnd, p in pooled.items()
    }
    (out / f"{name}.json").write_text(json.dumps(dump, indent=1) + "\n")
    print(f"wrote {out / name}.{{pdf,svg,png,json}}")


def draw(pooled: dict[str, dict[str, object]], base: Path, set_: str = "t1320") -> None:
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35, layout="constrained")
        axs = fig.subplots(1, len(pooled), sharey=True, squeeze=False)[0]
        for j, (rnd, p) in enumerate(pooled.items()):
            ax = axs[j]
            curves: dict[str, T.Curve] = p["curves"]  # type: ignore[assignment]
            tpa: dict[str, float] = p["tpa"]  # type: ignore[assignment]
            pass1: dict[str, float] = p["pass1"]  # type: ignore[assignment]
            for key, look in (("base", BASE), ("think", THINK)):
                T.band_line(ax, curves[key], look, z=3 if key == "think" else 2)
                ax.plot(
                    [tpa[key]],
                    [pass1[key]],
                    ls="none",
                    marker=look.marker,
                    ms=3.4,
                    mfc=look.color,
                    mec=S.WHITE,
                    mew=0.5,
                    zorder=4,
                )
            seeds: list[str] = p["seeds"]  # type: ignore[assignment]
            who = f"mean of {len(seeds)} seeds" if len(seeds) > 1 else f"seed {seeds[0][-1]} only"
            S.panel_title(ax, "ab"[j], f"{ROUNDS[rnd]} ({who})")
            T.tokens_axis(ax)
            T.pct_axis(ax)
        axs[0].set_ylim(0, 0.8 if set_ == "t1320" else 1.0)
        axs[0].set_ylabel(
            "problems solved (13–20 lines)" if set_ == "t1320" else "problems solved (textbook72)"
        )
        fig.supxlabel("tokens generated per problem (log scale)", fontsize=S.FS_LABEL)
        handles = [
            Line2D([], [], color=c.color, ls=c.ls, lw=c.lw, marker=c.marker, ms=3.4, label=c.label)
            for c in (THINK, BASE)
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        S.save(fig, base)
        S.close(fig)


if __name__ == "__main__":
    main()
