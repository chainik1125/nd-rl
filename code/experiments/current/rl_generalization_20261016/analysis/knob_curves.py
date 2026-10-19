"""Knob curves: how changing one GRPO setting changed the run over L3 training, for the full-scale
seed-0 knob pilots (thinking_traces_20261005/EXPERIMENT_QUEUE.md, "Knob pilots"). Eval reads come
through mini_results.py's loaders, so every pass@1 matches the existing reports; training curves
come from each L3 run's metrics.jsonl (nothing in results/ is read or written).

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python \
        code/experiments/current/rl_generalization_20261016/analysis/knob_curves.py [--out DIR]

Writes `fig_knob_ood`, `fig_knob_id`, `fig_knob_train` and `fig_knob_entropy` (.pdf, .svg, .png)
into `figures/` next to `analysis/` by default, and prints every plotted number and each figure's
caption.

Runs: `grpo_think_vs_mb_k<KNOB>` seed 0, each the merged recipe's L3 (`grpo_think_vs_mb`, KL anchor
0.02, 39 iterations, from `grpo_think_vs_s0_L2_it39.pt`) with one setting changed; `krr` changes
nothing (the noise control). x is the L3 iteration as logged. A rollback (`--rollback_on_alarm`)
restores an earlier checkpoint's weights and continues with the next iteration number, so after one
the x-axis counts more iterations than were kept; the printed tables give the kept updates.

1. `fig_knob_ood`: 13–20 pass@1 (the 986 valid items of t1320_n1000) at it0 (the shared L2
   checkpoint, read as `grpo_think_vs_L2` t1320 k4, exactly as mini_results.g3_view), the rolling
   checkpoints read (`<arm>_it<IT>`, k4) and the final (`<arm>`, k16). Panel titles give the final
   Δ against the merged recipe's final (seed 0, k16) in points; the merged recipe's seed 0 has no
   intermediate reads (its rolling checkpoints were deleted), so it is two black markers.
2. `fig_knob_id`: the same for L3 test_1k pass@1 (k8 everywhere). The merged recipe's seed 0 has
   no B64 L3 test read of its final (only the B0 one, a different budget), so its only marker is
   the shared it0 and the titles give each knob's final level instead of a Δ.
3. `fig_knob_train`: training pass@1 (`reward_pass1`: the share of attempts that succeed, not the
   penalised reward), a centred 5-iteration mean (shorter at the ends) over faint raw points.
4. `fig_knob_entropy`: policy entropy during the update (`entropy`, nats per scored token, at the
   training temperature; solid) and over think-block tokens only (`entropy_think`; dashed), centred
   5-iteration means, on a log axis.

Never reads textbook72. CPU only, one process, a few seconds.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from code.experiments.current.rl_generalization_20261016.analysis import mini_results as mr
from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as v2
from code.ndbench.figures import style as st

OUT = mr.STUDY / "figures"
EV = mr.R / "eval/v2"
BASE = v2.KNOB_BASE  # grpo_think_vs_mb, the merged recipe
SEED = 0
# (knob, panel label, the change; EXPERIMENT_QUEUE.md "Knob pilots" table), in panel order
KNOBS = (
    ("kl0", "KL anchor 0", "KL anchor 0 (0.02)"),
    ("kl1", "KL anchor 0.1", "KL anchor 0.1 (0.02)"),
    ("g64", "64 × 64 groups", "64 targets × 64 attempts (256 × 16)"),
    ("t12", "train T 1.2", "training temperature 1.2 (1.0)"),
    ("ch4", "clip-high 0.4", "clip-higher 0.4 (0.28)"),
    ("i78", "78 iterations", "78 L3 iterations (39)"),
    ("x66", "3-premise 0.66", "3-premise rows 0.66 of each batch (0.33)"),
    ("p3", "step penalty 0.3", "invalid-step penalty 0.3 (0.1)"),
    ("rr", "rerun", "nothing changed (noise control)"),
)
# 13–20 pass@1 Δ (points) from the "Knob pilots" table, checked against the reads
QUEUE_D1320 = {
    "kl0": 1.6,
    "kl1": -2.7,
    "g64": -5.5,
    "t12": -2.2,
    "i78": -1.5,
    "ch4": -1.7,
    "x66": 0.5,
    "p3": 0.0,
    "rr": -1.2,
}
# (sub, title, k at it0 = G3's, k at the final): the rolling reads' k is taken from their names
SETS = {
    "t1320": ("13–20 pass@1", 4, mr.REP_K["t1320"]),
    "l3test": ("L3 test_1k pass@1", 8, mr.FINAL_K["l3test"]),
}
HI = st.OKABE_ITO["vermillion"]  # the panel's knob
CTX = st.mix(st.REF, st.WHITE, 0.55)  # the other knobs
BLK = st.INK  # the merged recipe
HI_RAW = st.mix(HI, st.WHITE, 0.6)
BLK_RAW = st.mix(BLK, st.WHITE, 0.62)
VLINE = st.mix(st.REF, st.WHITE, 0.45)
TRI = 4.0  # rollback marker size
# the training curves whose level is not comparable with the merged recipe's (args.json)
NONCOMP = {"fig_knob_train": {"t12", "g64", "x66", "i78"}, "fig_knob_entropy": {"t12"}}
END = (35, 39)  # the training-curve titles compare these iterations' means


def arm_of(knob: str) -> str:
    return f"{BASE}_k{knob}"


def sgn(d: float) -> str:
    s = f"{d:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "−")


# ------------------------------------------------------------------------------------- training


@dataclass
class Train:
    run: str
    iters: int  # planned (args.json)
    rows: list[dict[str, Any]]
    rows_written: int
    restarts: int
    final_ckpt: bool
    args: dict[str, Any]
    # (alarm iteration, restored iteration, lr scale after)
    rollbacks: list[tuple[int, int, float]] = field(default_factory=list)
    ban_alarms: list[int] = field(default_factory=list)  # think-ban alarm iterations

    def series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([r["iter"] for r in self.rows], dtype=float)
        y = np.array([float(r.get(key, np.nan)) for r in self.rows])
        return x, y

    def kept(self, it: int) -> int:
        """Updates the weights at the end of iteration `it` carry: a rollback at iteration a to
        the checkpoint of iteration t restores t's weights, dropping every update since t."""
        back = {a: t for a, t, _ in self.rollbacks}
        n = [0]  # n[i]: the updates kept at the end of iteration i
        for i in range(1, it + 1):
            n.append(n[back[i]] if i in back else n[i - 1] + 1)
        return n[it]

    def mean(self, key: str, lo: int, hi: int) -> float:
        v = [float(r[key]) for r in self.rows if lo <= r["iter"] <= hi and key in r]
        return float(np.mean(v)) if v else float("nan")


def load_train(arm: str) -> Train:
    d = mr.GV / f"{arm}_s{SEED}_L3"
    rows, written, restarts = mr.metrics_rows(d / "metrics.jsonl")  # a rerun's rows win
    args = json.loads((d / "args.json").read_text())
    iters = int(args["iters"])
    t = Train(
        d.name,
        iters,
        rows,
        written,
        restarts,
        (d / "ckpts" / f"{d.name}_it{iters}.pt").exists(),
        args,
    )
    for r in rows:
        if r.get("rollback"):
            rb = r["rollback"]
            t.rollbacks.append((r["iter"], int(rb["to_iter"]), float(rb["lr_scale"])))
        if any("think banned" in a for a in r.get("alarms", [])):
            t.ban_alarms.append(r["iter"])
    return t


def smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Centred w-point mean, the window shortened at the ends."""
    h = w // 2
    return np.array([np.nanmean(y[max(0, i - h) : i + h + 1]) for i in range(len(y))])


# ---------------------------------------------------------------------------------------- reads


@dataclass
class Point:
    run: str  # the arm the curve belongs to
    it: int
    read: str  # the arm read
    k: int
    p: float  # set-mean pass@1, a fraction


def rolling_reads(arm: str, sub: str) -> list[tuple[int, int]]:
    """(iteration, k) of the `<sub>_<arm>_it<IT>_s0_plain_k<k>` reads on disk (it0 excluded: the
    shared L2 read stands for every run's it0)."""
    pat = re.compile(rf"{sub}_{re.escape(arm)}_it(\d+)_s{SEED}_plain_k(\d+)")
    out = []
    for x in EV.iterdir():
        m = pat.fullmatch(x.name)
        if m and int(m.group(1)) > 0:
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def load_reads(
    ld: mr.Loader, train: dict[str, Train], notes: list[str]
) -> dict[str, dict[str, list[Point]]]:
    """Per set, per run (knob arms and BASE): its points it0 … final."""
    out: dict[str, dict[str, list[Point]]] = {}
    for sub, (_, k0, kf) in SETS.items():
        p0 = mr.p_of(ld, mr.G3_IT0, sub, SEED, 1, k0)
        if p0 is None:
            notes.append(f"{sub}: no it0 read ({mr.G3_IT0} k{k0}); curves start at it10")
        per: dict[str, list[Point]] = {}
        for arm in (BASE, *(arm_of(k) for k, *_ in KNOBS)):
            pts = [] if p0 is None else [Point(arm, 0, mr.G3_IT0, k0, p0)]
            for it, k in rolling_reads(arm, sub):
                if mr.skey(sub, k) not in v2.SET:
                    notes.append(f"{sub}_{arm}_it{it} k{k}: k not registered, skipped")
                    continue
                p = mr.p_of(ld, f"{arm}_it{it}", sub, SEED, 1, k)
                if p is None:
                    notes.append(f"{sub}_{arm}_it{it}_s{SEED}_plain_k{k}: no rows/summary yet")
                    continue
                pts.append(Point(arm, it, f"{arm}_it{it}", k, p))
            fin = train[arm].iters
            p = mr.p_of(ld, arm, sub, SEED, 1, kf)
            if p is None:
                notes.append(f"{sub}_{arm}_s{SEED}_plain_k{kf}: missing (final, it{fin})")
            else:
                pts.append(Point(arm, fin, arm, kf, p))
            per[arm] = pts
        out[sub] = per
    return out


def load() -> mr.Loader:
    keep = mr.register_sets()  # also registers G3's it0 and kkl0 rolling arms
    for k, *_ in KNOBS:
        for sub in SETS:
            for it, _ in rolling_reads(arm_of(k), sub):
                mr.add_spec(f"{arm_of(k)}_it{it}", (SEED,), tuple(SETS))
    mr.add_spec(f"{arm_of('kl0')}_it0", (SEED,), tuple(SETS))  # cross-check of it0 only
    return mr.Loader(EV, keep)


# -------------------------------------------------------------------------------------- figures


def grid(height: float) -> tuple[Figure, list[Axes]]:
    fig = st.new_figure(st.FULL_W, height)
    axes = fig.subplots(3, 3, sharey=True)
    return fig, list(axes.ravel())


def x_axis(ax: Axes, iters: int) -> None:
    if iters == 39:
        ax.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 39]))
        ax.set_xlim(-2.5, 41.5)
    else:
        ax.xaxis.set_major_locator(FixedLocator([0, 20, 39, 60, iters]))
        ax.set_xlim(-4, iters + 3)
        ax.axvline(39, color=VLINE, lw=0.6, ls=(0, (2, 1.5)), zorder=0)
    ax.xaxis.set_minor_locator(NullLocator())


def rollback_marks(ax: Axes, t: Train) -> None:
    for a, _, _ in t.rollbacks:
        ax.plot(
            [a],
            [0.985],
            transform=ax.get_xaxis_transform(),
            marker="v",
            ms=TRI,
            mfc=HI,
            mec=st.WHITE,
            mew=0.4,
            ls="none",
            clip_on=False,
            zorder=5,
        )


def title(ax: Axes, knob: str, label: str, tail: str) -> None:
    st.panel_title(ax, None, f"{knob}: {label}, {tail}", size=st.FS_TICK)


def pct_y(ax: Axes) -> None:
    """st.pct_axis with whole-point steps (its 2.5-point steps print as rounded labels)."""
    st.pct_axis(ax, "y")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))


def tri() -> Line2D:
    return Line2D([], [], ls="none", marker="v", ms=TRI, mfc=HI, mec=st.WHITE, mew=0.4)


def finish(
    fig: Figure, axes: list[Axes], ylabel: str, handles: list[Any], labels: list[str]
) -> None:
    for i, ax in enumerate(axes):
        st.light_grid(ax, "y")
        ax.tick_params(labelsize=st.FS_SMALL)
        if i % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=st.FS_TICK)
        if i >= 6:
            ax.set_xlabel("L3 iteration", fontsize=st.FS_TICK)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=st.FS_TICK,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
    )


def fig_eval(
    sub: str, reads: dict[str, list[Point]], train: dict[str, Train], out: Path, name: str
) -> None:
    """Figures 1 and 2."""
    ylabel = SETS[sub][0]
    base = reads[BASE]
    b0 = next((p for p in base if p.it == 0), None)
    bfin = next((p for p in base if p.it == train[BASE].iters), None)
    with st.paper_style():
        fig, axes = grid(4.6)
        for ax, (knob, label, _) in zip(axes, KNOBS, strict=True):
            arm = arm_of(knob)
            top = train[arm].iters
            for other, *_ in KNOBS:  # context, cut at this panel's last iteration
                if other != knob:
                    o = [p for p in reads[arm_of(other)] if p.it <= top]
                    ax.plot([p.it for p in o], [p.p for p in o], color=CTX, lw=0.6, zorder=1)
            pts = reads[arm]
            ax.plot(
                [p.it for p in pts],
                [p.p for p in pts],
                color=HI,
                lw=1.2,
                marker="o",
                ms=3.4,
                mfc=HI,
                mec=st.WHITE,
                mew=0.4,
                zorder=3,
            )
            for p in (b0, bfin):
                if p is not None:
                    ax.plot(
                        [p.it], [p.p], ls="none", marker="s", ms=3.8, mfc=BLK, mec=st.WHITE,
                        mew=0.5, zorder=4,
                    )  # fmt: skip
            rollback_marks(ax, train[arm])
            x_axis(ax, top)
            fin = pts[-1] if pts and pts[-1].it == top else None
            if fin is not None and bfin is not None:
                tail = sgn(100 * (fin.p - bfin.p))
            elif fin is not None:
                tail = f"final {100 * fin.p:.1f}%"
            else:
                tail = "no final read"
            title(ax, knob, label, tail)
            pct_y(ax)
        handles = [
            Line2D([], [], color=HI, lw=1.2, marker="o", ms=3.4, mfc=HI, mec=st.WHITE),
            Line2D([], [], color=CTX, lw=0.6),
            Line2D([], [], ls="none", marker="s", ms=3.8, mfc=BLK, mec=st.WHITE),
            tri(),
        ]
        base_lab = (
            "merged recipe s0: it0 (shared L2 start) and final (no reads between)"
            if bfin is not None
            else "it0, the shared L2 start (merged recipe s0: no B64 final read)"
        )
        labels = [
            "the panel's knob, seed 0 (rolling checkpoints and final)",
            "the other knobs",
            base_lab,
            "rollback (weights restored, lr halved)",
        ]
        finish(fig, axes, ylabel, handles, labels)
        st.save(fig, out / name)
        st.close(fig)


def fig_train(
    key: str,
    train: dict[str, Train],
    out: Path,
    name: str,
    ylabel: str,
    *,
    think_key: str | None = None,
    log: bool = False,
) -> None:
    """Figures 3 and 4. Titles: the knob's mean over END minus (pass@1, points) or over
    (entropy, ratio) the merged recipe's; † marks a level that is not comparable."""
    base = train[BASE]
    with st.paper_style():
        fig, axes = grid(4.6)
        for ax, (knob, label, _) in zip(axes, KNOBS, strict=True):
            t = train[arm_of(knob)]
            for tr, col, raw, lw, z in ((base, BLK, BLK_RAW, 1.0, 3), (t, HI, HI_RAW, 1.2, 4)):
                x, y = tr.series(key)
                ax.plot(x, y, ls="none", marker="o", ms=3.0, mfc=raw, mec="none", zorder=1)
                ax.plot(x, smooth(y), color=col, lw=lw, zorder=z)
                if think_key:
                    xt, yt = tr.series(think_key)
                    ax.plot(xt, smooth(yt), color=col, lw=lw - 0.1, ls=(0, (3, 1.6)), zorder=z)
            rollback_marks(ax, t)
            x_axis(ax, t.iters)
            a, b = t.mean(key, *END), base.mean(key, *END)
            tail = f"×{a / b:.2f}" if log else sgn(100 * (a - b))
            title(ax, knob, label, tail + ("†" if knob in NONCOMP[name] else ""))
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                pct_y(ax)
        dots = (
            Line2D([], [], ls="none", marker="o", ms=3.0, mfc=HI_RAW, mec="none"),
            Line2D([], [], ls="none", marker="o", ms=3.0, mfc=BLK_RAW, mec="none"),
        )
        if think_key:
            handles = [
                Line2D([], [], color=HI, lw=1.2),
                Line2D([], [], color=BLK, lw=1.0),
                dots,
                tri(),
                Line2D([], [], color=st.INK2, lw=1.0),
                Line2D([], [], color=st.INK2, lw=0.9, ls=(0, (3, 1.6))),
            ]
            labels = [
                "the panel's knob, seed 0",
                "merged recipe, seed 0",
                "single iterations, all scored tokens",
                "rollback (weights restored, lr halved)",
                "all scored tokens (centred 5-iteration mean)",
                "think-block tokens only (centred 5-iteration mean)",
            ]
        else:
            handles = [Line2D([], [], color=HI, lw=1.2), Line2D([], [], color=BLK, lw=1.0)]
            handles += [dots, tri()]
            labels = [
                "the panel's knob, seed 0 (centred 5-iteration mean)",
                "merged recipe, seed 0 (centred 5-iteration mean)",
                "single iterations",
                "rollback (weights restored, lr halved)",
            ]
        finish(fig, axes, ylabel, handles, labels)
        st.save(fig, out / name)
        st.close(fig)


# ---------------------------------------------------------------------------------------- report


def comparability(train: dict[str, Train]) -> list[str]:
    """args.json differences from the merged recipe that change what the training curves
    measure."""
    b = train[BASE].args
    out = []
    for knob, *_ in KNOBS:
        a = train[arm_of(knob)].args
        diff = {
            k: (b.get(k), a.get(k))
            for k in sorted(set(a) | set(b))
            if k not in ("_meta", "name", "out") and a.get(k) != b.get(k)
        }
        out.append(f"  {knob}: " + (", ".join(f"{k} {x} → {y}" for k, (x, y) in diff.items())))
    return out


CAPTIONS = {
    "fig_knob_ood": (
        "13–20 pass@1 (% of the 986 valid items) over L3 training for the full-scale seed-0 knob "
        "pilots, each the merged recipe's L3 (KL 0.02, 39 iterations, from the shared L2 "
        "checkpoint) with one setting changed. Each panel highlights one knob; the others are "
        "grey. it0 is the shared L2 checkpoint (read k4), rolling checkpoints k4, finals k16. The"
        " merged recipe's seed 0 has no intermediate reads (its rolling checkpoints were "
        "deleted), so it is two black markers, it0 and it39. Titles: the knob's final minus the "
        "merged recipe's final, in points. Triangles: rollbacks (weights restored to an earlier "
        "checkpoint, lr halved; the iteration count continues). p3's reads are at it5, 20 and 25 "
        "(it rolled back 3 times); i78 runs to 78 (dashed line: it39). One seed per knob; rerun "
        "noise (rr) is about 1–1.5 points."
    ),
    "fig_knob_id": (
        "L3 test_1k pass@1 (in-distribution), the same runs and layout, all reads k8. The merged "
        "recipe's seed 0 has no B64 L3 test read of its final (only a B0 read, a different "
        "budget), so the black marker is the shared it0 only and the titles give each knob's "
        "final level."
    ),
    "fig_knob_train": (
        "Training pass@1 (share of the iteration's attempts that succeed), knob vs the merged "
        "recipe seed 0; centred 5-iteration means over the single iterations. Titles: the knob's "
        "mean over it 35–39 minus the merged recipe's, in points. † marks a level that is not "
        "comparable: t12 samples at T 1.2; g64 draws 64 targets × 64 attempts (the same 4,096 "
        "attempts, a quarter of the targets); x66 trains on twice the share of 3-premise rows; "
        "i78's WSD cooldown runs over its last 16 iterations (63–78), so at it 35–39 it is at its"
        " plateau lr (halved by its it16 rollback) where the others are cooling down. p3 changes "
        "the reward, not this metric (it counts successes), but its 3 rollbacks drop 9 of its 39 "
        "updates (kl0 and ch4 drop 4 each)."
    ),
    "fig_knob_entropy": (
        "Policy entropy during the update (nats per scored token of the masked policy at the "
        "training temperature; log axis): all scored tokens (solid, with single iterations) and "
        "think-block tokens only (dashed), centred 5-iteration means. Titles: the ratio of the "
        "knob's all-token entropy over it 35–39 to the merged recipe's. † t12's is measured at T "
        "1.2, so its level is not comparable."
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ld = load()
    train = {arm: load_train(arm) for arm in (BASE, *(arm_of(k) for k, *_ in KNOBS))}
    notes: list[str] = []
    reads = load_reads(ld, train, notes)

    print("Eval points (run, iteration, kept updates, set, pass@1 %, k, read):")
    for sub, per in reads.items():
        for arm, pts in per.items():
            for p in pts:
                print(
                    f"  {arm:28s} it{p.it:<3d} kept {train[arm].kept(p.it):>2d} {sub:7s} "
                    f"{100 * p.p:5.1f}  k{p.k:<3d} {p.read}"
                )
    x0 = mr.p_of(ld, f"{arm_of('kl0')}_it0", "l3test", SEED, 1, 8)
    if x0 is not None:
        print(f"  check: kkl0_it0 (its start checkpoint) L3 test k8 {100 * x0:.1f}%")

    print("\nFinal Δ vs the merged recipe's final (points), 13–20 pass@1, against the queue table:")
    base_fin = {p.it: p for p in reads["t1320"][BASE]}.get(train[BASE].iters)
    for knob, *_ in KNOBS:
        fin = {p.it: p for p in reads["t1320"][arm_of(knob)]}.get(train[arm_of(knob)].iters)
        if fin is None or base_fin is None:
            print(f"  {knob}: no final pair")
            continue
        d = 100 * (fin.p - base_fin.p)
        q = QUEUE_D1320[knob]
        flag = "ok" if abs(round(d, 1) - q) < 0.05 else f"MISMATCH (queue {q:+.1f})"
        print(f"  {knob}: {sgn(d)} {flag}")

    print("\nTraining (first / last 5-iteration means; i78 also at it35–39):")
    for arm, t in train.items():
        n = t.rows[-1]["iter"]
        line = (
            f"  {arm:28s} pass@1 {100 * t.mean('reward_pass1', 1, 5):.1f} → "
            f"{100 * t.mean('reward_pass1', n - 4, n):.1f}%  H {t.mean('entropy', 1, 5):.3f} → "
            f"{t.mean('entropy', n - 4, n):.3f}  Hthink {t.mean('entropy_think', 1, 5):.3f} → "
            f"{t.mean('entropy_think', n - 4, n):.3f}"
        )
        if arm != BASE:
            d1 = 100 * (t.mean("reward_pass1", *END) - train[BASE].mean("reward_pass1", *END))
            rh = t.mean("entropy", *END) / train[BASE].mean("entropy", *END)
            line += f"  it35–39 vs merged: pass@1 {sgn(d1)}, H ×{rh:.2f}"
        if n != 39:
            line += (
                f"  [it35–39: pass@1 {100 * t.mean('reward_pass1', 35, 39):.1f}%, "
                f"H {t.mean('entropy', 35, 39):.3f}]"
            )
        print(line)
        print(
            f"    rows {len(t.rows)} (written {t.rows_written}, restarts {t.restarts}), "
            f"final ckpt it{t.iters} {'yes' if t.final_ckpt else 'NO'}, rollbacks "
            + (", ".join(f"it{x}→it{y} (lr ×{s:g})" for x, y, s in t.rollbacks) or "none")
            + f", think-ban alarms {len(t.ban_alarms)}"
            + (f" (it {', '.join(map(str, t.ban_alarms))})" if t.ban_alarms else "")
        )
    print("\nargs.json vs the merged recipe:")
    print("\n".join(comparability(train)))

    fig_eval("t1320", reads["t1320"], train, out, "fig_knob_ood")
    fig_eval("l3test", reads["l3test"], train, out, "fig_knob_id")
    fig_train("reward_pass1", train, out, "fig_knob_train", "training pass@1")
    fig_train(
        "entropy",
        train,
        out,
        "fig_knob_entropy",
        "entropy (nats / token)",
        think_key="entropy_think",
        log=True,
    )
    if notes:
        print("\nGaps:\n" + "\n".join(f"  {x}" for x in notes))
    print("\nCaptions:")
    for k, v in CAPTIONS.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {out}/fig_knob_ood.*, fig_knob_id.*, fig_knob_train.*, fig_knob_entropy.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
