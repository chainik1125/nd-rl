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
    return 0


def stage1(layer: int = 3) -> None:
    rows = json.loads((HERE / "stage1.json").read_text())
    rows = [r for r in rows if r["layer"] == layer and r["cc_seed"] == 0]
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 1.95, layout="constrained")
        axs = fig.subplots(1, 3)
        seen: set[str] = set()
        cpt = [r for r in rows if r["b"].startswith("cpt")]
        for r in rows:
            if r["b"].startswith("cpt"):
                continue
            arm = arm_of(r["b"])
            name, col, mk = ARM[arm]
            other = r["seed"] != 0
            kw = dict(color=col, marker=mk, s=13, linewidths=0.8, zorder=3)
            kw["facecolors"] = "none" if other else col
            lab = None if arm in seen or other else name
            seen.add(arm)
            axs[0].scatter(r["R"], r["N_RL"] + 0.5, label=lab, **kw)
            axs[1].scatter(r["R"], 100 * r["E_RL"]["all"] + 1e-3, **kw)
        for r in cpt:  # no RL compute: drawn as a horizontal reference
            axs[0].axhline(r["N_RL"] + 0.5, color=ARM["cpt"][1], ls="--", lw=0.8)
            axs[1].axhline(100 * r["E_RL"]["all"] + 1e-3, color=ARM["cpt"][1], ls="--", lw=0.8)
        for ax in axs[:2]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("RL / Stage-1 FLOPs (R)")
            S.light_grid(ax, "both")
        axs[0].set_ylabel("strict RL-only latents (+0.5)")
        axs[1].set_ylabel("RL variance in RL-only latents (%)")
        S.panel_title(axs[0], "a", "RL-only latents")
        S.panel_title(axs[1], "b", "their share of variance")
        if cpt:
            axs[0].plot([], [], color=ARM["cpt"][1], ls="--", lw=0.8, label=ARM["cpt"][0])
        axs[0].legend(loc="upper left", fontsize=S.FS_TICK - 1.5, frameon=False)
        bins = np.linspace(0, 1, 21)
        mids = 0.5 * (bins[1:] + bins[:-1])
        pick = {
            "kp16_st1": ("null (R 7.6e-5)", C["sky_blue"], "-"),
            "mini_it8": ("mini20 L2 it8", C["bluish_green"], "-"),
            "L3": ("L3 end", C["vermillion"], "-"),
        }
        for r in rows:
            if r["seed"] == 0 and r["b"] in pick:
                lab, col, ls = pick[r["b"]]
                h = np.array(r["dnorm_hist"], dtype=float)
                axs[2].step(
                    mids, np.maximum(h, 0.5), where="mid", color=col, ls=ls, lw=1.0, label=lab
                )
        axs[2].set_yscale("log")
        axs[2].set_xlabel("Δnorm (0 = RL-only, 1 = base-only)")
        axs[2].set_ylabel("live latents")
        axs[2].axvspan(0, 0.1, color=S.SHADE, zorder=0)
        axs[2].axvspan(0.9, 1, color=S.SHADE, zorder=0)
        S.panel_title(axs[2], "c", "decoder-norm balance")
        axs[2].legend(loc="upper left", fontsize=S.FS_TICK - 1.5, frameon=False)
        S.save(fig, FIG / f"stage1_L{layer}")
        S.close(fig)


def stage2() -> None:  # filled in once Stage 2 has numbers
    pass


if __name__ == "__main__":
    raise SystemExit(main())
