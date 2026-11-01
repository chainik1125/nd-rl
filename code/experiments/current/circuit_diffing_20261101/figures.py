"""Figures for the circuit-diffing experiment (code/ndbench/figures/style.py: 5.5 in wide, serif
7-8 pt, PDF / SVG / PNG).

    python -m code.experiments.current.circuit_diffing_20261101.figures [--only stage0,stage1]

Inputs: `stage0.json`, `stage1.json`, `stage2.json` in this folder. Outputs: `figs/`.
"""

from __future__ import annotations

import argparse
import json
import math

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


def stage1() -> None:  # filled in once Stage 1 has numbers
    pass


def stage2() -> None:  # filled in once Stage 2 has numbers
    pass


if __name__ == "__main__":
    raise SystemExit(main())
