#!/usr/bin/env python3
"""Tables and figures from results/.

  python analyze.py          # writes figures/*.png and numbers.md

pass@k uses the Chen et al. estimator per problem (1 - C(n-c, k) / C(n, k) from c successes in n samples), averaged
over problems. Intervals are 95 % bootstraps over problems, paired where a gap is reported.
"""
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8f8e88"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 12,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK,
    "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 220,
})
ROUND_COLOR = {1: "#9cc3ee", 2: "#7cafe8", 4: "#4f93df", 8: "#2a78d6", 16: "#123f7a"}
FIG_ROUNDS = (1, 2, 4, 8, 16)  # the rounds every seed has; 3, 6, 12 stay in numbers.md
NS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
KS = (1, 4, 16, 32)
BOOT = 2000


def per_problem(res, k):
    return np.array([1 - comb(r["n_tried"] - r["n_ok"], k) / comb(r["n_tried"], k) for r in res["rows"]])


def ci(vals, rng):
    idx = rng.integers(len(vals), size=(BOOT, len(vals)))
    return np.percentile(vals[idx].mean(1), [2.5, 97.5])


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")} if Path(d).exists() else {}


def svd_table(seed_dir, others="full"):
    """{round: {n: result}} plus references: 'pre' is the pretrain and 'r' the full RL model. With others='full', n = 0
    moves only the 1-D parameters, and n = 384 would equal the RL model because every matrix has rank at most 384."""
    res = load(seed_dir)
    rounds = sorted(v["round"] for k, v in res.items() if k.startswith("ref_r") and v["round"] > 0)
    out = {}
    for r in rounds:
        out[r] = {n: res[f"r{r}_n{n}_{others}"] for n in [*NS, 384] if f"r{r}_n{n}_{others}" in res}
        out[r]["pre"], out[r]["r"] = res["ref_r0"], res[f"ref_r{r}"]
    return out


def fig_svd(table, k, path, title, foot):
    """x: components kept per matrix. y: pass@k. One line per RL round, each ending at that round's own full model."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = {"pre": 0, **{n: i + 1 for i, n in enumerate(NS)}, "r": len(NS) + 1}
    table = {r: row for r, row in table.items() if r in FIG_ROUNDS}
    for r, row in sorted(table.items()):
        keys = [n for n in ["pre", *NS, "r"] if n in row]
        ys = [per_problem(row[n], k).mean() for n in keys]
        ax.plot([xs[n] for n in keys], ys, color=ROUND_COLOR.get(r, INK), marker="o", mec=SURFACE, mew=1.2, label=f"{r}")
        if r == max(table):
            lo, hi = zip(*(ci(per_problem(row[n], k), rng) for n in keys))
            ax.fill_between([xs[n] for n in keys], lo, hi, color=ROUND_COLOR.get(r, INK), alpha=0.12, lw=0)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["pre-\ntrain", *map(str, NS), "RL\nmodel"], fontsize=9)
    ax.set_xlabel("top singular components of the weight change kept per matrix")
    ax.set_ylabel(f"pass@{k}")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(title="RL rounds", ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=INK)
    fig.subplots_adjust(bottom=0.27, left=0.09, right=0.86, top=0.93)
    fig.text(0.01, 0.01, foot, fontsize=8, color=MUTED, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def gap(row_edit, r0, r1, k, rng, removed=False):
    """Share of the pass@k gap between r0 and r1 recovered (or, for an edit of r1, lost)."""
    e, a, b = (per_problem(x, k) for x in (row_edit, r0, r1))
    num = (b - e) if removed else (e - a)
    den = b - a
    idx = rng.integers(len(e), size=(BOOT, len(e)))
    boot = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), *np.percentile(boot, [2.5, 97.5])


def fig_sae(rows, k, path):
    """x: SAE features edited. y: gap recovered. One line per RL round, one panel per arm."""
    arms = [("steer_mean", "Steer pretrain: mean shift", False), ("steer_map", "Steer pretrain: per-feature map", False),
            ("ablate_map", "Remove from RL model: gap lost", True)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    ns = [1, 4, 16, 64, 256, 1024, 8192]
    for ax, (arm, title, removed) in zip(axes, arms):
        ax.axhline(0, color=MUTED, lw=1)
        for r, res in sorted(rows.items()):
            keys = [n for n in ns if f"{arm}_n{n}" in res]
            if not keys:
                continue
            ys = [gap(res[f"{arm}_n{n}"], res["ref_r0"], res["ref_r"], k, rng, removed)[0] for n in keys]
            ax.plot(keys, ys, color=ROUND_COLOR.get(r, INK), marker="o", mec=SURFACE, mew=1.2, label=f"{r}")
            rand = [n for n in (64, 1024) if f"{arm}_rand_n{n}" in res]
            ctrl = [gap(res[f"{arm}_rand_n{n}"], res["ref_r0"], res["ref_r"], k, rng, removed)[0] for n in rand]
            ax.plot(rand, ctrl, ls="", marker="x", ms=7, mew=1.6, color=BASE, zorder=5, label="random" if r == max(rows) else None)
        ax.set_xscale("log", base=4)
        ax.set_xticks(ns[:-1] + [8192])
        ax.set_xticklabels(["1", "4", "16", "64", "256", "1k", "all"], fontsize=9)
        ax.minorticks_off()
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("SAE features edited")
    axes[0].set_ylabel(f"share of pass@{k} gap")
    axes[1].legend(*[x[::1] for x in axes[1].get_legend_handles_labels()], title="RL rounds", loc="center left",
                   bbox_to_anchor=(2.08, 0.5), labelcolor=INK)
    fig.subplots_adjust(bottom=0.26, left=0.07, right=0.89, top=0.9, wspace=0.08)
    fig.text(0.01, 0.01, "255 textbook problems, Lean-judged, 32 samples per problem. Seed 0, block 4, BatchTopK SAE (8192 latents, k = 32). "
             "Grey x: same edit on random latents.\n95 % intervals: about ±0.13 per point, ±0.25 at round 1 (small gap), so wiggles under that size are noise.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    lines = ["# Numbers (generated by analyze.py)\n"]
    rng = np.random.default_rng(0)
    n_problems = None
    dirs = sorted((RES / "svd").glob("s*")) if (RES / "svd").exists() else []
    for d, others in [(d, o) for d in dirs for o in ("full", "base")]:
        table = svd_table(d, others)
        table = {r: row for r, row in table.items() if len(row) > 2}
        if not table:
            continue
        lines.append(f"\n## SVD {d.name}, 1-D parameters: {'moved to round r' if others == 'full' else 'held at pretrain'}\n")
        lines.append("| round | n | energy kept | " + " | ".join(f"pass@{k}" for k in KS) + " | gap@32 recovered [95 %] |")
        lines.append("|---|---|---|" + "---|" * (len(KS) + 1))
        for r, row in sorted(table.items()):
            n_problems = len(row["pre"]["rows"])
            for n in ["pre", *NS, 384, "r"]:
                if n not in row:
                    continue
                g = gap(row[n], row["pre"], row["r"], 32, rng) if n not in ("pre", "r") else None
                gs = f"{g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}]" if g else ""
                e = row[n].get("energy")
                vals = " | ".join(f"{per_problem(row[n], k).mean():.3f}" for k in KS)
                lines.append(f"| {r} | {n} | {'' if e is None else f'{e:.2f}'} | {vals} | {gs} |")
    s0 = svd_table(RES / "svd" / "s0") if (RES / "svd" / "s0").exists() else {}
    s0 = {r: row for r, row in s0.items() if len(row) > 2}
    if s0:
        foot = (f"{n_problems} textbook problems, Lean-judged, 32 samples per problem. Seed 0; band: 95 % bootstrap over problems, round 16.\n"
                "x = 0 moves only the 1-D parameters to round r; every other point adds the top n components of each matrix.")
        for k in (32, 1):
            fig_svd(s0, k, FIG / f"svd_passk{k}.png", "Weight change truncated to n components, by RL round", foot)
    rows = {}
    for d in sorted((RES / "sae").glob("s0_r*_b4")) if (RES / "sae").exists() else []:
        rows[int(d.name.split("_r")[1].split("_")[0])] = load(d)
    all_sae = sorted((RES / "sae").glob("s*")) if (RES / "sae").exists() else []
    for d in all_sae:
        res = load(d)
        if "ref_r" not in res or "ref_r0" not in res:
            continue
        lines.append(f"\n## SAE {d.name}: {json.dumps(res.get('sae_info', {}))[:200]}\n")
        lines.append(f"pass@32: r0 {per_problem(res['ref_r0'], 32).mean():.3f}, r {per_problem(res['ref_r'], 32).mean():.3f}\n")
        lines.append("| edit | gap share@32 [95 %] | gap share@1 |")
        lines.append("|---|---|---|")
        for tag, v in sorted(res.items()):
            if "arm" not in v:
                continue
            removed = v["arm"].startswith("ablate")
            g, g1 = gap(v, res["ref_r0"], res["ref_r"], 32, rng, removed), gap(v, res["ref_r0"], res["ref_r"], 1, rng, removed)
            lines.append(f"| {tag} | {g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}] | {g1[0]:.2f} |")
    if rows:
        fig_sae(rows, 32, FIG / "sae_gap32.png")
    (HERE / "numbers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
