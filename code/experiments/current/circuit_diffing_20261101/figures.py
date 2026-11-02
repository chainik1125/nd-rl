"""Figures for the circuit-diffing experiment (code/ndbench/figures/style.py: 5.5 in wide, serif
7-8 pt, PDF / SVG / PNG).

    python -m code.experiments.current.circuit_diffing_20261101.figures [--only stage0,stage1]

Inputs: `stage0.json`, `stage1.json`, `stage2.json` in this folder. Outputs: `figs/`.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from code.ndbench.figures import style as S

from .common import HERE

FIG = HERE / "figs"
C = S.OKABE_ITO
INK = S.INK
ARM = {  # label prefix -> (legend, colour, marker)
    "kp16": ("low-RL, 16 prompts/step", C["sky_blue"], "o"),
    "kdense": ("low-RL, dense", C["blue"], "s"),
    "mini": ("mini20 L2", C["bluish_green"], "D"),
    "L2": ("L2 / L3 chain", C["vermillion"], "^"),
    "cpt": ("weight-drift control (CPT)", C["black"], "x"),
}


def arm_of(label: str) -> str:
    if label.startswith(("L2", "L3", "ki78")):
        return "L2"
    return label.split("_")[0]


def adamw_steps(label: str) -> int:
    if label.startswith(("kp16_st", "kdense_st")):
        return int(label.split("st")[-1])
    if label.startswith("mini_it"):
        return 4 * int(label[7:])
    if label == "L2":
        return 156
    if label.startswith("L3_it"):
        return 156 + 4 * int(label[5:])
    if label == "L3":
        return 312
    if label == "ki78":
        return 468
    if label.startswith("cpt"):
        return 312
    raise ValueError(label)


def stage0() -> None:
    res = json.loads((HERE / "stage0.json").read_text())
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 1.9, layout="constrained")
        axs = fig.subplots(1, 3)
        seen: set[str] = set()
        for v in res.values():
            arm = arm_of(v["label"])
            name, col, mk = ARM[arm]
            other_seed = v["seed"] != 0
            kw = dict(
                color=col,
                marker=mk,
                s=12,
                facecolors="none" if other_seed else col,
                linewidths=0.8,
                zorder=3,
            )
            lab = None if arm in seen or other_seed else name
            seen.add(arm)
            md = v["median"]
            if math.isfinite(v["R"]):
                axs[0].scatter(v["R"], 100 * md["rel"], label=lab, **kw)
                axs[2].scatter(v["R"], md["align"], **kw)
            axs[1].scatter(adamw_steps(v["label"]), 100 * md["rel"], **kw)
        for ax in axs:
            ax.set_xscale("log")
            S.light_grid(ax, "both")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[0].set_xlabel("RL / Stage-1 FLOPs (R)")
        axs[1].set_xlabel("AdamW steps")
        axs[2].set_xlabel("RL / Stage-1 FLOPs (R)")
        axs[0].set_ylabel("median |ΔW| / |W| (%)")
        axs[1].set_ylabel("median |ΔW| / |W| (%)")
        axs[2].set_ylabel("top-10% subspace share / random")
        axs[2].axhline(1, color=S.REF, lw=0.7, ls=":")
        axs[2].set_ylim(0, 10)
        S.panel_title(axs[0], "a", "weight change vs R")
        S.panel_title(axs[1], "b", "weight change vs steps")
        S.panel_title(axs[2], "c", "alignment with base")
        axs[0].legend(loc="upper left", fontsize=S.FS_TICK - 1, frameon=False, handletextpad=0.2)
        S.save(fig, FIG / "stage0_weights")
        S.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="stage0,stage1,stage2")
    a = ap.parse_args(argv)
    todo = a.only.split(",")
    if "stage0" in todo:
        stage0()
    if "stage1" in todo and (HERE / "stage1.json").exists():
        stage1()
    if "stage2" in todo and (HERE / "stage2.json").exists():
        stage2()
        stage2("L3_s0_L5_n8192_k32_seed0", "L5")
    return 0


def stage1(layer: int = 3) -> None:
    import torch

    from .stage1_analyze import CC, arrays, classify

    rows = json.loads((HERE / "stage1.json").read_text())
    rows = [r for r in rows if r["layer"] == layer and r["cc_seed"] == 0]
    main = [r for r in rows if not r["planted"]]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.0, layout="constrained")
        axs = fig.subplots(1, 3)
        # (a) Δnorm histograms
        bins = np.linspace(0, 1, 21)
        mids = 0.5 * (bins[1:] + bins[:-1])
        pick = [
            ("kp16_st1", False, "null (R 7.6e-5)", C["sky_blue"], "-"),
            ("L3", False, "L3 end (R 0.29)", C["vermillion"], "-"),
            ("L3", True, "L3 end + 6 planted", INK, ":"),
        ]
        for b, pl, lab, col, ls in pick:
            for r in rows:
                if r["b"].split(" ")[0] == b and r["planted"] == pl and r["seed"] == 0:
                    h = np.array(r["dnorm_hist"], dtype=float)
                    axs[0].step(
                        mids, np.maximum(h, 0.3), where="mid", color=col, ls=ls, lw=1.0, label=lab
                    )
        axs[0].set_yscale("log")
        axs[0].set_ylim(0.3, 2e4)
        axs[0].set_xlabel("Δnorm (0 = RL-only, 1 = base-only)")
        axs[0].set_ylabel("live latents")
        axs[0].axvspan(0, 0.1, color=S.SHADE, zorder=0)
        axs[0].axvspan(0.9, 1, color=S.SHADE, zorder=0)
        S.panel_title(axs[0], "a", "decoder-norm balance")
        axs[0].legend(loc="upper right", fontsize=S.FS_TICK - 1.5, frameon=False)
        axs[0].set_ylim(0.3, 2e5)
        # (b) Δnorm vs decoder cosine, L3 end and the planted control
        for name, col, mk, sz in (
            (f"L3_s0_L{layer}_n8192_k32_seed0", S.REF, ".", 2),
            (f"L3_s0_L{layer}_n8192_k32_seed0_plant", INK, "x", 14),
        ):
            p = CC / name / "latents.pt"
            if not p.exists():
                continue
            L = arrays(torch.load(p, weights_only=False))
            c = classify(L)
            live = c["live"]
            if mk == "x":  # only the latents that track a planted feature
                ev = json.loads((CC / name / "eval.json").read_text())
                ids = sorted(
                    {
                        cand["latent"]
                        for pl in ev["planted"]
                        for cand in pl["candidates"]
                        if (cand["precision"] or 0) > 0.25 and (cand["recall"] or 0) > 0.25
                    }
                )
                axs[1].scatter(
                    c["dnorm"][ids],
                    L["cos"][ids],
                    marker="x",
                    s=sz,
                    color=col,
                    linewidths=0.8,
                    label="planted features",
                    zorder=4,
                )
            else:
                axs[1].scatter(
                    c["dnorm"][live],
                    L["cos"][live],
                    marker=".",
                    s=sz,
                    color=col,
                    linewidths=0,
                    label="L3 end latents",
                    rasterized=True,
                )
        axs[1].axvspan(0, 0.1, color=S.SHADE, zorder=0)
        axs[1].set_xlim(0, 1)
        axs[1].set_xlabel("Δnorm")
        axs[1].set_ylabel("decoder cosine (base, RL)")
        S.panel_title(axs[1], "b", "every RL latent has a base twin")
        axs[1].legend(loc="lower right", fontsize=S.FS_TICK - 1.5, frameon=False)
        # (c) rotation vs R
        seen: set[str] = set()
        for r in main:
            arm = arm_of(r["b"])
            name, col, mk = ARM[arm]
            other = r["seed"] != 0
            kw = dict(color=col, marker=mk, s=13, linewidths=0.8, zorder=3)
            kw["facecolors"] = "none" if other else col
            lab = None if arm in seen or other else name
            seen.add(arm)
            y = 1 - r["ewt_cos"]
            if arm == "cpt":
                axs[2].axhline(max(y, 1e-5), color=col, ls="--", lw=0.8)
                continue
            axs[2].scatter(r["R"], max(y, 1e-5), label=lab, **kw)
        cpu = HERE / "stage1_cpu.json"
        cpt_cpu = [
            r
            for r in (json.loads(cpu.read_text()) if cpu.exists() else [])
            if r["b"].startswith("cpt") and r["layer"] == layer
        ]
        for r in cpt_cpu:  # reduced CPU crosscoder (its L3-end twin matches the GPU index)
            axs[2].axhline(1 - r["ewt_cos"], color=INK, ls="--", lw=0.8)
        if cpt_cpu or any(arm_of(r["b"]) == "cpt" for r in main):
            axs[2].plot([], [], color=INK, ls="--", lw=0.8, label="CPT, no RL (matched ΔW)")
        axs[2].set_xscale("log")
        axs[2].set_yscale("log")
        axs[2].set_xlabel("RL / Stage-1 FLOPs (R)")
        axs[2].set_ylabel("1 − energy-weighted decoder cos")
        S.light_grid(axs[2], "both")
        S.panel_title(axs[2], "c", "decoder rotation vs R")
        axs[2].legend(loc="lower right", fontsize=S.FS_TICK - 1.5, frameon=False)
        S.save(fig, FIG / f"stage1_L{layer}")
        S.close(fig)


S2_LABEL = {
    "base_clean": "base",
    "base_rescale": "base, shared latents at RL norm",
    "base_transplant": "base, latents switched to RL version",
    "base_replace": "base, residual replaced by RL's",
    "rl_clean": "RL (L3 end)",
    "rl_unscale": "RL, shared latents at base norm",
    "rl_recon": "RL, reconstruction only",
    "rl_revert": "RL, latents switched to base version",
    "rl_revertperm": "RL, the same edits permuted (control)",
}
S2_ORDER = [
    "base_clean",
    "base_rescale",
    "base_transplant",
    "base_replace",
    "rl_clean",
    "rl_unscale",
    "rl_recon",
    "rl_revert",
    "rl_revertperm",
]


def stage2(name: str = "L3_s0_L3_n8192_k32_seed0", tag: str = "L3") -> None:
    allres = json.loads((HERE / "stage2.json").read_text())
    if name not in allres:
        return
    res = allres[name]["conditions"]
    conds = [c for c in S2_ORDER if c in res]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.5, layout="constrained")
        ax = fig.subplots(1, 1)
        ys = np.arange(len(conds))[::-1].astype(float)
        nb = sum(c.startswith("base") for c in conds)
        ys[nb:] -= 0.6  # a gap between the base-driven and the RL-driven conditions
        for y, cnd in zip(ys, conds, strict=True):
            r = res[cnd]
            for cat, col, mk, dy, lab in (
                ("G", C["vermillion"], "o", 0.14, "RL-gained problems (G, 80)"),
                ("S", C["blue"], "s", -0.14, "shared problems (S, 121)"),
            ):
                v = r["solve_pct"][cat]
                lo, hi = r["solve_ci"][cat]
                ax.errorbar(
                    v,
                    y + dy,
                    xerr=[[v - lo], [hi - v]],
                    fmt=mk,
                    color=col,
                    ms=3.2,
                    lw=0.8,
                    capsize=1.5,
                    label=lab if cnd == conds[0] else None,
                )
        for ref, ls in (("base_clean", ":"), ("rl_clean", "--")):
            if ref in res:
                for cat, col in (("G", C["vermillion"]), ("S", C["blue"])):
                    ax.axvline(res[ref]["solve_pct"][cat], color=col, ls=ls, lw=0.6, alpha=0.6)
        ax.axhline(ys[nb - 1] - 0.8 if nb else 0, color=S.GRID, lw=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([S2_LABEL[c] for c in conds])
        ax.set_xlim(0, 100)
        ax.set_xlabel("solve rate (% of attempts, mean over problems; 95% problem bootstrap)")
        S.light_grid(ax, "x")
        ax.legend(loc="upper center", fontsize=S.FS_TICK - 1, frameon=True, framealpha=0.9)
        S.save(fig, FIG / f"stage2_{tag}")
        S.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
