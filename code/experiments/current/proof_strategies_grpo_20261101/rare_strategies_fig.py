"""How much RL each proof strategy needs to double its share of correct proofs, against how much
of it pretraining contained: one panel, 5.5 in wide, laid out like Dan's knockout chart.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.rare_strategies_fig [--draws 4000]

Natural prevalence, not a knockout: nothing was removed from the corpus. Charles149, k 1,024,
seed 0, Dan's taxonomy v1.2 (`analyze.TRACK`).
- x: the strategy's share of the S1g3 Stage-1 corpus proofs (numbers.json `q1.corpus_s1g3`).
- Share of correct proofs: Bpool, the pooled share of all Lean-accepted attempts that use the
  strategy (`analyze.unit_Bpool`). The bar is Bpool >= 2 x the base's; 2 x is our choice.
- y: R = RL / Stage-1 FLOPs at the first checkpoint of the main run, in R order (mini L2 it2, it4,
  it8, L2 end, L3 end, L3 x2), where the bar holds. It is interpolated linearly in log10 R between
  that checkpoint and the one before it, on the fold change. A strategy at the bar by the first
  checkpoint is drawn there as "at or below" (down-triangle); one below it at L3 x2 (R 0.45) is an
  up-triangle above the axis.
- The two low-RL arms (kp16, kdense: R 7.6e-5 to 6.6e-3) are not used for y. The script reports
  which strategies meet the bar in them.
- Bars: 95% problem bootstrap of the crossing R; problems are resampled jointly in every read.
Reads the per-problem tables that `classify` wrote under `common.OUT`. Writes
figs/strategy_rl_needed.{pdf,svg,png} and rare_strategies_numbers.json here.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from typing import Any

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from matplotlib.transforms import offset_copy

from code.ndbench.figures import style as S

from ..high_rl_passk_20261030.passk_figs import RL_HUE, YUE_MEDIAN
from . import common as C
from .analyze import Read, load_read, unit_Bpool

K = 1024
ST = "charles149"
BAR = 2.0
BASE = "s1g3"
#: The main run's checkpoints in R order: the mini L2 run (the L2 command on a 1,000-row pool),
#: then the full L2 end, L3 end and L3 x2 (ki78).
MAIN = [
    "grpo_think_vs_mini20_L2it2",
    "grpo_think_vs_mini20_L2it4",
    "grpo_think_vs_mini20_L2it8",
    "grpo_think_vs_L2",
    "grpo_think_vs_mb_kkl0",
    "grpo_think_vs_mb_ki78",
]
MAIN_SHORT = ["mini L2 it2", "it4", "it8", "L2 end", "L3 end", "L3 ×2"]
LOWRL = [s for s in C.CHAIN if "kp16" in s or "kdense" in s]
#: The figure's strategies (key -> plain name), then the other distinct taxonomy keys (table only).
FIG = {
    "work_red": "proof by contradiction, working",
    "free_red": "proof by contradiction, free (working, no ¬¬ scaffold)",
    "templ_red": "proof by contradiction, idle only",
    "any_dn": "any classical step (contradiction, DNE, LEM)",
    "mt": "modus tollens",
    "ds": "disjunctive syllogism",
    "work_neg": "proof of a negation, working",
    "gen_cases": "proof by cases, genuine",
    "dne": "double-negation elimination",
    "efq": "ex falso",
    "lem": "excluded middle",
}
TABLE_ONLY = {
    "u:contradiction": "proof by contradiction, any (working or idle)",
    "u:cases": "proof by cases, any",
    "u:neg_intro": "proof of a negation, any",
    "u:direct": "direct reasoning (#1)",
    "u:and_intro": "conjunction introduction (#2)",
    "u:cond_proof": "conditional proof (#3)",
    "u:or_intro": "disjunction introduction (#4)",
    "u:lemma_reuse": "lemma reuse (#13)",
}
#: u:* keys that equal a tracked key on every problem of every read (checked below).
DUPLICATES = {
    "u:disj_syll": "ds",
    "u:mod_tollens": "mt",
    "u:ex_falso": "efq",
    "u:dne": "dne",
    "u:lem": "lem",
}
NOT_C = S.INK2
NEVER_C = S.REF


# ------------------------------------------------------------------------------ numbers
def crossing(fold: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First crossing of BAR along the main checkpoints, per row of `fold` (draws x checkpoints).
    Returns (log10 R, kind) with kind 0 = interpolated, 1 = at or below the first checkpoint,
    2 = not reached, 3 = undefined (the base and every checkpoint at 0)."""
    hit = np.nan_to_num(fold, nan=-1.0) >= BAR
    anyhit = hit.any(axis=1)
    i = hit.argmax(axis=1)
    rows = np.arange(len(fold))
    j = np.maximum(i - 1, 0)
    f0, f1 = fold[rows, j], fold[rows, i]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(np.isfinite(f1) & (f1 > f0), (BAR - f0) / (f1 - f0), 1.0)
    out = np.where(anyhit, lr[j] + np.clip(t, 0, 1) * (lr[i] - lr[j]), np.inf)
    kind = np.where(anyhit, np.where(i == 0, 1, 0), 2)
    out = np.where(kind == 1, lr[0], out)
    undefined = np.isnan(fold).all(axis=1)
    kind = np.where(undefined, 3, kind)
    return out, kind


def fold_of(base: np.ndarray, ck: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(base > 0, ck / base, np.where(ck > 0, np.inf, np.nan))


def compute(draws: int) -> dict[str, Any]:
    N = json.loads((C.HERE / "numbers.json").read_text())
    corpus, q2 = N["q1"]["corpus_s1g3"], N["q2_charles149"]
    ref = N["q1"]["charles149_reference"]
    keys = list(FIG) + list(TABLE_ONLY) + list(DUPLICATES)
    stems = [BASE] + MAIN + LOWRL
    reads = {s: Read(load_read(ST, s, 0, K), keys) for s in stems}
    names = reads[BASE].names
    assert len(names) == 149 and all(r.names == names for r in reads.values())
    for r in reads.values():
        for u, k in DUPLICATES.items():
            assert np.array_equal(r.c[u], r.c[k]), (u, k)
    R = C.compute_R()
    lr = np.log10([R[s] for s in MAIN])
    rng = np.random.default_rng(0)
    n = len(names)
    W = np.zeros((draws, n))
    for d in range(draws):
        W[d] = np.bincount(rng.integers(0, n, n), minlength=n)

    def bpool_draws(r: Read, key: str) -> np.ndarray:
        return 100 * (W @ r.c[key]) / (W @ r.n_ok)

    out: dict[str, Any] = {"bar": BAR, "draws": draws, "main": {}, "strategies": {}}
    for s, nm in zip(MAIN, MAIN_SHORT, strict=True):
        out["main"][s] = {"short": nm, "R": R[s]}
    for key in list(FIG) + list(TABLE_ONLY):
        full = {s: unit_Bpool(reads[s], key) for s in stems}
        for s in stems:  # the per-problem tables reproduce numbers.json
            assert math.isclose(full[s], q2[s]["Bpool"][key], abs_tol=1e-9), (key, s)
        fb = full[BASE]
        fold_main = np.array([[fold_of(np.array(fb), np.array(full[s])) for s in MAIN]])
        lr0, kind0 = crossing(fold_main.astype(float), lr)
        bb = bpool_draws(reads[BASE], key)
        bf = np.stack([fold_of(bb, bpool_draws(reads[s], key)) for s in MAIN], axis=1)
        lrb, kindb = crossing(bf, lr)
        end_b = bpool_draws(reads[C.END], key)
        fold_end = fold_of(bb, end_b)
        srt = np.sort(lrb)
        lo, hi = srt[int(0.025 * draws)], srt[int(math.ceil(0.975 * draws)) - 1]
        low_hits = [
            {"stem": s, "R": R[s], "fold": full[s] / fb}
            for s in LOWRL
            if fb > 0 and full[s] / fb >= BAR
        ]
        out["strategies"][key] = {
            "name": FIG.get(key) or TABLE_ONLY[key],
            "in_figure": key in FIG,
            "corpus_pct": corpus[key],
            "reference_pct": ref[key],
            "base_Bpool": fb,
            "base_Bpool_ci": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
            "end_Bpool": full[C.END],
            "end_Bpool_ci": [float(np.percentile(end_b, 2.5)), float(np.percentile(end_b, 97.5))],
            "fold_end": full[C.END] / fb if fb > 0 else None,
            "fold_end_ci": (
                [float(np.nanpercentile(fold_end, 2.5)), float(np.nanpercentile(fold_end, 97.5))]
                if fb > 0
                else None
            ),
            "Bpool_main": {s: full[s] for s in MAIN},
            "fold_main": {s: (full[s] / fb if fb > 0 else None) for s in MAIN},
            "max_fold_main": max(full[s] / fb for s in MAIN) if fb > 0 else None,
            "kind": ["interpolated", "at_or_below_first", "not_reached", "never_used"][
                int(kind0[0])
            ],
            "R_cross": float(10 ** lr0[0]) if np.isfinite(lr0[0]) else None,
            "R_cross_ci": [
                float(10**lo) if np.isfinite(lo) else None,
                float(10**hi) if np.isfinite(hi) else None,
            ],
            "draws_reached_pct": float(100 * np.mean(kindb < 2)),
            "draws_at_or_below_first_pct": float(100 * np.mean(kindb == 1)),
            "lowrl_at_bar": low_hits,
        }
    out["pass1"] = {s: q2[s]["attempt_success_pct"] for s in [BASE] + MAIN}
    return out


# ------------------------------------------------------------------------------ figure
def _pow10(v: float, _pos: Any = None) -> str:
    return f"$10^{{{int(round(math.log10(v)))}}}$"


def _pct(v: float, _pos: Any = None) -> str:
    return f"{v:g}%"


#: label placement: key -> (dx, dy) in points from the marker, ha, va
LABELS_MAIN: dict[str, tuple[float, float, str, str]] = {
    "work_red": (-6.0, 4.5, "right", "center"),
    "free_red": (-6.0, -4.5, "right", "center"),
    "any_dn": (5.0, 0.0, "left", "center"),
    "templ_red": (4.0, -4.0, "left", "top"),
}
#: not-reached band: key -> (row, label x in % or None for the marker's own x, ha). Row 0 is
#: beside the marker; a label on a higher row hangs on a leader (up, then across to its x).
#: A marker's leader never crosses a label on a lower row: the 4-6% cluster's labels stack
#: right to left from the bottom row up.
CLUSTER_X = 6.8
LABELS_BAND: dict[str, tuple[int, float | None, str]] = {
    "lem": (2, None, "left"),
    "mt": (0, None, "left"),
    "ds": (1, None, "right"),
    "gen_cases": (1, CLUSTER_X, "left"),
    "work_neg": (2, CLUSTER_X, "left"),
    "dne": (3, CLUSTER_X, "left"),
    "efq": (0, None, "left"),
}
BAND_ROWS = {0: 5.0, 1: 14.0, 2: 22.5, 3: 31.0}


def draw(num: dict[str, Any]) -> list[str]:
    st = num["strategies"]
    with S.paper_style():
        FW, FH = S.FULL_W, 3.25
        fig = S.new_figure(FW, FH)
        fig.set_layout_engine("none")
        L, W, B0 = 0.62, 4.34, 0.80
        ax = fig.add_axes((L / FW, B0 / FH, W / FW, 1.55 / FH))
        band = fig.add_axes((L / FW, (B0 + 1.61) / FH, W / FW, 0.50 / FH), sharex=ax)
        XL = (0.04, 45.0)
        ax.set_xscale("log")
        ax.set_xlim(*XL)
        ax.set_yscale("log")
        rmax = num["main"][MAIN[-1]]["R"]
        ax.set_ylim(7e-5, rmax * 1.25)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(_pct))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
        ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("share of the pretraining proofs that use the strategy (S1g3 corpus)")
        ax.set_ylabel("RL ÷ pretraining FLOPs\nto double its share")
        texts = []
        # the main run's checkpoints
        for i, s in enumerate(MAIN):
            r = num["main"][s]["R"]
            ax.axhline(r, color=S.GRID, lw=0.5, zorder=0)
            texts.append(
                ax.text(
                    1.012,
                    r,
                    num["main"][s]["short"],
                    transform=ax.get_yaxis_transform(),
                    fontsize=S.FS_SMALL,
                    color=S.REF,
                    ha="left",
                    va="bottom" if i == len(MAIN) - 1 else ("top" if i >= 3 else "center"),
                    clip_on=False,
                )
            )
        ax.axhline(YUE_MEDIAN, color=S.INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
        yue = ax.text(
            XL[0] * 1.12,
            YUE_MEDIAN * 1.18,
            r"Yue et al. median RL budget ($1.2\times10^{-4}$)",
            fontsize=S.FS_SMALL,
            color=S.INK2,
            ha="left",
            va="bottom",
        )
        texts.append(yue)
        # reached: dots, 95% bootstrap bars; work and free share x (1.47 vs 1.45%), dodged
        dodge = {"work_red": 2.2, "free_red": -2.2}
        for key in FIG:
            v = st[key]
            x = v["corpus_pct"]
            if v["kind"] not in ("interpolated", "at_or_below_first"):
                continue
            tr = offset_copy(ax.transData, fig=fig, x=dodge.get(key, 0.0), y=0, units="points")
            y = v["R_cross"]
            lo, hi = v["R_cross_ci"]
            hi_plot = hi if hi is not None else rmax
            ax.plot([x, x], [lo, hi_plot], color=RL_HUE, lw=0.7, transform=tr, zorder=3)
            hollow = key == "free_red"
            ax.plot(
                [x],
                [y],
                ls="none",
                marker="v" if v["kind"] == "at_or_below_first" else "o",
                ms=4.0,
                mfc=S.WHITE if hollow else RL_HUE,
                mec=RL_HUE,
                mew=0.8,
                transform=tr,
                zorder=4,
            )
            dx, dy, ha, va = LABELS_MAIN[key]
            name = v["name"]
            if v["kind"] == "at_or_below_first":
                name += ": at the bar by the first checkpoint"
            if key == "any_dn":
                name = name.replace(" (", "\n(")
            texts.append(
                ax.annotate(
                    name,
                    (x, y),
                    xytext=(dx + dodge.get(key, 0.0), dy),
                    textcoords="offset points",
                    fontsize=S.FS_SMALL,
                    color=S.INK,
                    ha=ha,
                    va=va,
                    zorder=5,
                )
            )
        S.light_grid(ax, "x")
        # the not-reached band: up-triangles on thin lines from the largest budget
        band.set_ylim(0, 36)
        band.yaxis.set_major_locator(NullLocator())
        band.yaxis.set_minor_locator(NullLocator())
        band.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for sp in ("left", "bottom"):
            band.spines[sp].set_visible(False)
        texts.append(
            band.text(
                -0.012,
                5.0,
                "not reached\nby 0.45×",
                transform=band.get_yaxis_transform(),
                fontsize=S.FS_TICK,
                ha="right",
                va="center",
                color=S.INK,
            )
        )
        y0 = BAND_ROWS[0]
        for key, (row, lx, ha) in LABELS_BAND.items():
            v = st[key]
            x = v["corpus_pct"]
            never = v["kind"] == "never_used"
            col = NEVER_C if never else NOT_C
            if never:
                band.plot([x], [y0], ls="none", marker="x", ms=4.0, mec=col, mew=0.9, zorder=4)
            else:
                band.plot([x, x], [0.0, y0 - 1.8], color=col, lw=0.6, zorder=3)
                band.plot(
                    [x], [y0], ls="none", marker="^", ms=4.0, mfc=col, mec=col, mew=0.5, zorder=4
                )
            name = v["name"]
            if never:
                name += ": never used, before or after RL (textbooks: 0%)"
            y = BAND_ROWS[row]
            if row == 0:
                xy, off = (x, y), (4.0 if ha == "left" else -4.0, 0.0)
            elif lx is None:
                band.plot([x, x], [y0 + 2.6, y - 3.2], color=S.REF, lw=0.4, zorder=2)
                xy, off = (x, y), (-1.0, 0.0)
            else:
                band.plot([x, x, lx], [y0 + 2.6, y, y], color=S.REF, lw=0.4, zorder=2)
                xy, off = (lx, y), (1.5, 0.0)
            texts.append(
                band.annotate(
                    name,
                    xy,
                    xytext=off,
                    textcoords="offset points",
                    fontsize=S.FS_SMALL,
                    color=S.INK,
                    ha=ha,
                    va="center",
                    zorder=5,
                )
            )
        fig.text(
            0.02 / FW,
            (FH - 0.06) / FH,
            "RL needed to double a strategy's share of correct proofs, "
            "by how much of it pretraining contained",
            fontsize=S.FS_LABEL,
            ha="left",
            va="top",
        )
        small = textwrap.fill(
            "Natural prevalence, not a knockout. Seed 0, Charles149 (149 problems), k = 1,024. "
            "Bar: the strategy's share of all Lean-accepted attempts reaches 2× the base's (2× "
            "is our choice). y: R at the first checkpoint of the main run (grey lines) at the "
            "bar, interpolated in log R; bars: 95% problem bootstrap; down-triangle: at the bar "
            "by the first checkpoint. The low-RL arms (kp16, kdense) are not used. Hollow: "
            "free\u00a0⊂\u00a0working, shifted sideways (1.45 vs 1.47%). The GRPO pools are 40–45% "
            "classical-only theorems.",
            width=137,
        )
        texts.append(
            fig.text(
                0.02 / FW,
                0.03 / FH,
                small,
                fontsize=S.FS_SMALL,
                color=S.INK2,
                ha="left",
                va="bottom",
                linespacing=1.25,
            )
        )
        check_overlaps(fig, texts)
        base = C.HERE / "figs" / "strategy_rl_needed"
        paths = [str(p) for p in S.save(fig, base)]
        S.close(fig)
    return paths


def check_overlaps(fig: Any, texts: list[Any]) -> None:
    """Print any pair of direct labels whose boxes overlap, or a label outside the figure."""
    rend = fig.canvas.get_renderer()
    boxes = [(t.get_text()[:28], t.get_window_extent(rend)) for t in texts]
    fb = fig.bbox
    for i, (a, ba) in enumerate(boxes):
        if ba.x0 < fb.x0 or ba.x1 > fb.x1 or ba.y0 < fb.y0 or ba.y1 > fb.y1:
            print(f"label outside the figure: {a!r}")
        for b, bb in boxes[i + 1 :]:
            if ba.overlaps(bb):
                print(f"labels overlap: {a!r} / {b!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args(argv)
    num = compute(a.draws)
    num["paths"] = draw(num)
    (C.HERE / "rare_strategies_numbers.json").write_text(json.dumps(num, indent=1))
    for key, v in num["strategies"].items():
        ci = v["R_cross_ci"]
        print(
            f"{key:16s} x {v['corpus_pct']:6.3f}  base {v['base_Bpool']:6.2f}  "
            f"end {v['end_Bpool']:6.2f}  fold {v['fold_end'] or float('nan'):5.2f}  "
            f"{v['kind']:18s} R {v['R_cross']}  ci {ci}  reached {v['draws_reached_pct']:.1f}%  "
            f"lowrl {[(h['stem'][-8:], round(h['fold'], 2)) for h in v['lowrl_at_bar']]}"
        )
    print("wrote", num["paths"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
