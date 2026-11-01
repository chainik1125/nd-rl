"""Figure 1 (teaser): Charles149 pass@k up to k = 1024, base vs every RL checkpoint read so far.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \
        code.experiments.current.high_rl_passk_20261030.teaser [--draws 2000] [--compute TSV]

Picks up every finished k-1024 Charles149 read in runs/eval/v2
(`charles149_<checkpoint>_s0_plain_k1024/`, with summary.json; partial reads are skipped), so new
checkpoints appear on a rerun with no code change. Each RL checkpoint needs its RL / Stage-1 FLOP
ratio R in the compute table (`lowrl_compute.tsv`, a copy of
ynez:/scratch/cowifornia/nd-rl/think_hirl/lowrl/compute.tsv; column R, the README's definition).

(a) pass@k against k (log2), base grey dashed with its band, RL checkpoints coloured by R on one
sequential ramp (fixed log scale, so colours do not move as points arrive), band on the final
model only (the recipe's L3 end, else the largest R).  (b) RL - base in points against R at k = 1
(filled) and k = 1024 (open), with 95% problem-bootstrap bars, a zero line, the region below it
shaded (the elicitation signature at k 1024), and Yue et al.'s median FLOP ratio.  Writes
figures/teaser_passk.{pdf,png} and teaser_numbers.{md,json} next to this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from code.ndbench.figures import style as S
from code.thinking import stats

from .passk_figs import (
    BASE_C,
    FIGDIR,
    HERE,
    KEEP,
    RL_HUE,
    V2,
    YUE_MEDIAN,
    band,
    diff_curve,
    mean_curve,
    pow2,
    save,
)

K = 1024
BASE_STEM = "s1g3"
FINAL = "grpo_think_vs_mb_kkl0"  # the recipe's L3 end
R_MIN, R_MAX = 5e-5, 0.5  # fixed colour scale (covers the low-RL arm and ki78)
COMPUTE = HERE / "lowrl_compute.tsv"


def reads() -> dict[str, Path]:
    """checkpoint stem -> rows file, finished reads only."""
    out = {}
    for d in sorted(V2.glob(f"charles149_*_s0_plain_k{K}")):
        if not (d / "summary.json").exists():
            continue
        stem = d.name[len("charles149_") : -len(f"_s0_plain_k{K}")]
        out[stem] = d / "rows_plain_B64.jsonl"
    return out


def load(stem: str, path: Path) -> stats.Arm:
    a = stats.load_arm(stem, [path])
    keep = KEEP["charles149"]
    assert keep is not None
    a = a.subset(sorted(keep & set(a.names)), stem)
    if len(a.names) != len(keep) or a.kmax < K:
        raise ValueError(f"{stem}: {len(a.names)} problems, k {a.kmax}")
    return a


def ratios(path: Path) -> dict[str, float]:
    with open(path, newline="") as f:
        return {r["checkpoint"]: float(r["R"]) for r in csv.DictReader(f, delimiter="\t")}


def cmap() -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "rl_ramp", [S.shade(RL_HUE, 0.55), RL_HUE, S.shade(RL_HUE, -0.62)]
    )


def short(stem: str) -> str:
    s = stem.removeprefix("grpo_think_vs_")
    return {"mb_kkl0": "L3 end (recipe)", "mb_ki78": "L3 x2 (ki78)", "L2": "L2 end"}.get(s, s)


def make(draws: int, compute: Path, figdir: Path) -> dict[str, Any]:
    rd = reads()
    if BASE_STEM not in rd:
        raise SystemExit(f"no finished base read charles149_{BASE_STEM}_s0_plain_k{K}")
    R = ratios(compute)
    base = load(BASE_STEM, rd[BASE_STEM])
    rl: list[tuple[float, str, stats.Arm]] = []
    skipped = []
    for stem, p in rd.items():
        if stem == BASE_STEM:
            continue
        if stem not in R:
            skipped.append(stem)
            continue
        try:
            rl.append((R[stem], stem, load(stem, p)))
        except (ValueError, OSError) as e:  # a malformed or short read: report, do not plot
            skipped.append(f"{stem} ({e})")
    rl.sort()
    final = FINAL if any(s == FINAL for _, s, _ in rl) else (rl[-1][1] if rl else None)
    norm = mcolors.LogNorm(vmin=R_MIN, vmax=R_MAX)
    cmp = cmap()
    N: dict[str, Any] = {"k": K, "skipped_no_R": skipped, "checkpoints": []}
    ks = pow2(K)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.05)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
        ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        # (a) curves
        cb = 100 * mean_curve(base)
        kk = np.arange(1, K + 1)
        lo, hi = band(base, ks, draws)
        ax.fill_between(ks, 100 * lo, 100 * hi, color=BASE_C, alpha=0.16, lw=0, zorder=2)
        ax.plot(kk, cb, color=BASE_C, ls=(0, (3, 1.5)), lw=1.2, zorder=5)
        N["base"] = {
            "pass": {k: float(cb[k - 1]) for k in ks},
            "pass5": float(cb[4]),
            "lo": (100 * lo).tolist(),
            "hi": (100 * hi).tolist(),
        }
        for r, stem, a in rl:
            c = 100 * mean_curve(a)[:K]
            col = mcolors.to_hex(cmp(norm(min(max(r, R_MIN), R_MAX))))
            is_final = stem == final
            ax.plot(kk, c, color=col, lw=1.3 if is_final else 0.8, zorder=4 if is_final else 3)
            if is_final:
                flo, fhi = band(a, ks, draws)
                ax.fill_between(ks, 100 * flo, 100 * fhi, color=col, alpha=0.18, lw=0, zorder=2)
            d = diff_curve(a, base, [1, K], max(draws, 4000))
            N["checkpoints"].append(
                {
                    "checkpoint": stem,
                    "R": r,
                    "final": is_final,
                    "pass": {k: float(c[k - 1]) for k in ks},
                    "pass5": float(c[4]),
                    "gap1": [float(d["est"][0]), float(d["lo"][0]), float(d["hi"][0])],
                    "gapK": [float(d["est"][1]), float(d["lo"][1]), float(d["hi"][1])],
                }
            )
        S.log2_k_axis(ax, K, every=4)
        ax.set_ylim(30, 100)
        ax.set_xlabel("k (attempts per problem)")
        ax.set_ylabel("pass@k (% of Charles149)")
        S.light_grid(ax)
        S.panel_title(ax, "a", "Charles149, base vs RL checkpoints")
        ax.legend(
            handles=[
                Line2D([], [], color=BASE_C, ls=(0, (3, 1.5)), lw=1.2, label="base (Stage 1)"),
                Line2D([], [], color=S.shade(RL_HUE, -0.3), lw=1.0, label="RL checkpoints"),
            ],
            loc="lower right",
            fontsize=S.FS_SMALL,
        )
        sm = cm.ScalarMappable(norm=norm, cmap=cmp)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015, aspect=30)
        cbar.set_label("RL / pretraining FLOPs", fontsize=S.FS_SMALL, labelpad=1.0)
        cbar.ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.tick_params(labelsize=S.FS_SMALL, length=2, width=0.5)
        cbar.outline.set_linewidth(0.5)
        # (b) gap against R
        bx.axhspan(-100, 0, color=S.SHADE, lw=0, zorder=0)
        bx.axhline(0, color=S.REF, lw=0.7, zorder=1)
        bx.axvline(YUE_MEDIAN, color=S.INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
        pts = N["checkpoints"]
        for key, filled in (("gap1", True), ("gapK", False)):
            for p in pts:
                col = mcolors.to_hex(cmp(norm(min(max(p["R"], R_MIN), R_MAX))))
                e, l_, h_ = p[key]
                bx.plot([p["R"]] * 2, [l_, h_], color=col, lw=0.6, zorder=3)
                bx.plot(
                    p["R"],
                    e,
                    ls="none",
                    marker="o" if filled else "s",
                    ms=3.6,
                    mfc=col if filled else "white",
                    mec="white" if filled else col,
                    mew=0.4 if filled else 0.9,
                    zorder=4,
                )
        allv = [v for p in pts for key in ("gap1", "gapK") for v in p[key]] or [0.0]
        bx.set_ylim(min(-8.0, min(allv) - 2), max(40.0, max(allv) + 3))
        bx.set_xscale("log")
        bx.set_xlim(3e-5, 1.2)
        bx.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1]))
        bx.xaxis.set_major_formatter(FuncFormatter(_pow10))
        bx.xaxis.set_minor_formatter(NullFormatter())
        bx.text(
            YUE_MEDIAN * 1.2,
            0.97,
            "Yue et al.\nmedian",
            transform=bx.get_xaxis_transform(),
            fontsize=S.FS_SMALL,
            va="top",
            ha="left",
            color=S.INK2,
        )
        bx.text(
            0.98,
            0.03,
            "RL below base",
            transform=bx.transAxes,
            fontsize=S.FS_SMALL,
            ha="right",
            va="bottom",
            color=S.INK2,
        )
        bx.set_xlabel("RL / pretraining FLOPs")
        bx.set_ylabel("RL − base (points)")
        S.light_grid(bx)
        S.panel_title(bx, "b", f"RL − base: k = 1 (filled), k = {K} (open)")
        N["paths"] = [str(p) for p in save(fig, figdir / "teaser_passk")]
        S.close(fig)
    return N


def _pow10(v: float, _pos: Any = None) -> str:
    return f"$10^{{{int(round(math.log10(v)))}}}$"


def write_numbers(N: dict[str, Any]) -> None:
    (HERE / "teaser_numbers.json").write_text(json.dumps(N, indent=1) + "\n")
    b = N["base"]["pass"]
    L = [
        "# Teaser (Figure 1) numbers: Charles149 at k up to 1024 (generated by `teaser.py`)",
        "",
        f"Base S1g3_s0: pass@1 {b[1]:.1f}, pass@5 {N['base']['pass5']:.1f}, pass@1024 {b[K]:.1f} "
        "(% of 149). Gaps in points, 95% problem-bootstrap intervals.",
        "",
        "| checkpoint | R | pass@1 | pass@5 | pass@1024 | gap pass@1 [95%] | gap pass@1024 [95%] |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in N["checkpoints"]:
        g1, gk = p["gap1"], p["gapK"]
        L.append(
            f"| {short(p['checkpoint'])} | {p['R']:.3g} | {p['pass'][1]:.1f} | {p['pass5']:.1f} | "
            f"{p['pass'][K]:.1f} | "
            f"{g1[0]:+.1f} [{g1[1]:+.1f}, {g1[2]:+.1f}] | "
            f"{gk[0]:+.1f} [{gk[1]:+.1f}, {gk[2]:+.1f}] |"
        )
    if N["skipped_no_R"]:
        L += ["", f"Skipped (no R in the compute table): {', '.join(N['skipped_no_R'])}"]
    (HERE / "teaser_numbers.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--compute", type=Path, default=COMPUTE)
    ap.add_argument("--figdir", type=Path, default=FIGDIR)
    a = ap.parse_args(argv)
    N = make(a.draws, a.compute, a.figdir)
    write_numbers(N)
    for p in N["paths"]:
        print(p)
    print((HERE / "teaser_numbers.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
