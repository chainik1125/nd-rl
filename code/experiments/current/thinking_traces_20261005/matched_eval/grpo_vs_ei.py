"""GRPO-trained thinking vs EI-trained thinking vs the EI no-think baseline, at matched tokens.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.grpo_vs_ei [--stage g2]

Leon, 2026-10-09: "train blue without EI and with GRPO". Same Stage-1 inits (S1g3 seeds 0 and 1 for
thinking, T0S for the baseline) and the same training problems at matched sampled attempts: GRPO
stage g2 (78 iterations: rl_hard12, then hard712 a) <-> EI round 2; g3 (117) <-> round 3. Thinking
models are read at B64, the baseline at B0. Curves are the mean over the two seeds of
`tts_figs.analyse` (problem bootstrap, shared resample, so differences are paired); the bands leave
out seed variance. Pairs whose rows are missing are skipped.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from code.experiments.current.thinking_traces_20261005.matched_eval import tts_figs as T
from code.experiments.current.thinking_traces_20261005.matched_eval.tts_pooled import mean_curve
from code.ndbench.figures import style as S
from code.thinking import stats

TTS, M = T.TTS, T.MATCHED
GRPO = T.Look("GRPO + thinking", T.AQUA, "-", "D")
EI_T = T.Look("EI + thinking", T.BLUE, "-", "o")
BASE = T.Look("EI, no thinking (baseline)", T.ORANGE, T.DASH, "s")
ROUND = {"g2": "r2", "g3": "r3"}


def rows(d: Path, b: int) -> Path:
    return d / f"rows_plain_B{b}.jsonl"


def pairs(stage: str) -> dict[str, list[tuple[Path, Path, Path]]]:
    """set -> per seed (GRPO thinking rows, EI thinking rows, EI no-think rows)."""
    r = ROUND[stage]
    s1 = f"{r}s1"
    return {
        "t1320": [
            (
                rows(TTS / f"t1320_grpo_{stage}s0_k32", 64),
                rows(TTS / f"t1320_think_{r}_k32", 64),
                rows(TTS / f"t1320_nothink_{r}_k64", 0),
            ),
            (
                rows(TTS / f"t1320_grpo_{stage}s1_k32", 64),
                rows(TTS / f"t1320_think_{s1}_k32", 64),
                rows(TTS / f"t1320_nothink_{s1}_k64", 0),
            ),
        ],
        "tb72": [
            (
                rows(TTS / f"tb72_grpo_{stage}s0_k64", 64),
                rows(M / "tb72_eiS1g3_h712_r2", 64)
                if r == "r2"
                else rows(TTS / "tb72_think_r3_k64", 64),
                rows(M / "tb72_eiT0S_h712_r2", 0)
                if r == "r2"
                else rows(TTS / "tb72_nothink_r3_k64", 0),
            ),
            (
                rows(TTS / f"tb72_grpo_{stage}s1_k64", 64),
                rows(TTS / f"tb72_think_{s1}_k64", 64),
                rows(TTS / f"tb72_nothink_{s1}_k64", 0),
            ),
        ],
    }


def pp(e: float, lo: float, hi: float) -> str:
    return f"{100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(ROUND), default="g2")
    ap.add_argument("--out", default="experiment-summaries/2026-10-05-thinking-traces/figures")
    ap.add_argument("--draws", type=int, default=T.DRAWS)
    a = ap.parse_args(argv)
    lines: list[str] = []
    panels: list[tuple[str, dict[str, T.Curve], dict[str, float], dict[str, float]]] = []
    with tempfile.TemporaryDirectory() as td:
        loader = T.Loader(Path(td))
        for set_, seeds in pairs(a.stage).items():
            per: dict[str, list[T.Curve]] = {"grpo": [], "ei": [], "base": []}
            p1: dict[str, list[float]] = {k: [] for k in per}
            p5: dict[str, list[float]] = {k: [] for k in per}
            tpa: dict[str, list[float]] = {k: [] for k in per}
            used = 0
            for s, (g, e, b) in enumerate(seeds):
                if not all(p.exists() for p in (g, e, b)):
                    print(f"skipped {set_} seed {s}: rows missing")
                    continue
                used += 1
                for key, think in (("grpo", g), ("ei", e)):
                    r = T.analyse(T.Pair(f"s{s}", set_, "", think, b), loader, a.draws)
                    assert r is not None
                    per[key].append(r.curves["think"])
                    p1[key].append(r.pass1["think"].est)
                    p5[key].append(stats.ci(r.arms["think"], 5, draws=a.draws).est)
                    tpa[key].append(r.tpa["think"])
                    if key == "grpo":
                        per["base"].append(r.curves["base"])
                        p1["base"].append(r.pass1["base"].est)
                        p5["base"].append(stats.ci(r.arms["base"], 5, draws=a.draws).est)
                        tpa["base"].append(r.tpa["base"])
                q1 = " / ".join(f"{100 * p1[k][-1]:.1f}" for k in ("grpo", "ei", "base"))
                q5 = " / ".join(f"{100 * p5[k][-1]:.1f}" for k in ("grpo", "ei", "base"))
                lines.append(
                    f"{set_} seed {s}: GRPO / EI-think / baseline pass@1 {q1}; pass@5 {q5}"
                )
            if not used:
                continue
            curves = {k: mean_curve(v) for k, v in per.items()}
            for x in T.TABLE_X:
                d_base = T.difference(curves["grpo"], curves["base"]).at(x)
                d_ei = T.difference(curves["grpo"], curves["ei"]).at(x)
                flag = "" if d_base[3] and d_ei[3] else " (out of range)"
                v = " / ".join(f"{100 * curves[k].at(x)[0]:.1f}" for k in ("grpo", "ei", "base"))
                lines.append(
                    f"{set_} mean of {used} seeds, {x / 1000:g}k tokens: GRPO / EI-think / "
                    f"baseline {v}; GRPO - baseline {pp(*d_base[:3])}, "
                    f"GRPO - EI-think {pp(*d_ei[:3])}{flag}"
                )
            panels.append(
                (
                    set_,
                    curves,
                    {k: float(np.mean(v)) for k, v in tpa.items()},
                    {k: float(np.mean(v)) for k, v in p1.items()},
                )
            )
    print("\n".join(lines))
    draw(panels, Path(a.out) / f"fig_grpo_vs_ei_{a.stage}", a.stage)
    (Path(a.out) / f"fig_grpo_vs_ei_{a.stage}.txt").write_text("\n".join(lines) + "\n")


def draw(panels, base: Path, stage: str) -> None:  # noqa: ANN001
    titles = {"t1320": "13–20-line proofs (986)", "tb72": "textbook72 (72)"}
    looks = {"base": BASE, "ei": EI_T, "grpo": GRPO}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35, layout="constrained")
        axs = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
        for j, (set_, curves, tpa, p1) in enumerate(panels):
            ax = axs[j]
            for key in ("base", "ei", "grpo"):
                look = looks[key]
                T.band_line(ax, curves[key], look, z=3 if key == "grpo" else 2)
                ax.plot(
                    [tpa[key]],
                    [p1[key]],
                    ls="none",
                    marker=look.marker,
                    ms=3.4,
                    mfc=look.color,
                    mec=S.WHITE,
                    mew=0.5,
                    zorder=4,
                )
            S.panel_title(ax, "ab"[j], f"{titles[set_]}, mean of 2 seeds")
            T.tokens_axis(ax)
            T.pct_axis(ax)
        axs[0].set_ylim(0, 1.0)
        axs[0].set_ylabel("problems solved")
        n_it = {"g2": 78, "g3": 117}[stage]
        fig.supxlabel(
            f"tokens generated per problem (log scale); training: EI round {ROUND[stage][1]} = "
            f"GRPO iteration {n_it}, same problems and attempts",
            fontsize=S.FS_LABEL,
        )
        handles = [
            Line2D([], [], color=c.color, ls=c.ls, lw=c.lw, marker=c.marker, ms=3.4, label=c.label)
            for c in (GRPO, EI_T, BASE)
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        S.save(fig, base)
        S.close(fig)


if __name__ == "__main__":
    main()
