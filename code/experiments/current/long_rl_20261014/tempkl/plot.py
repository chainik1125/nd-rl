#!/usr/bin/env python3
"""tempkl.png from summary.json (analyze.py): raw and residual KL and T* over the RL rounds of each run, the per-band
breakdown of the last checkpoint, and the accepted/rejected split.

  uv run --no-project --with matplotlib,numpy tempkl/plot.py        # matplotlib is not in code/.venv
"""

import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_ROUND = {"v4_untied-ei_s0": 36, "v3_untied-grpo_s0": 168}  # oracle calls at final.pt (summary.json of each run)
BANDS = ("7-9", "10-14", "15-19", "20-24", "25-34", "35-49", "50+")
C = {"raw": "#4C72B0", "residual": "#DD8452", "t_star": "#55A868"}


def run_round(tag):
    run, stage = tag.rsplit("_", 1)
    return run, 0 if stage == "pretrain" else FINAL_ROUND[run] if stage == "final" else int(re.sub(r"\D", "", stage))


def main():
    summ = json.load(open(os.path.join(HERE, "summary.json")))
    runs = {}
    for s in summ:
        run, rnd = run_round(s["tag"])
        runs.setdefault(run, []).append((rnd, s))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    # (a) raw and residual KL over RL rounds, (b) T* over rounds
    for run, pts in sorted(runs.items()):
        pts.sort()
        x = [r for r, _ in pts]
        ls = "-" if "ei" in run else "--"
        axes[0, 0].plot(x, [s["all"]["raw"] for _, s in pts], ls, marker="o", color=C["raw"], label=f"{run} raw (T = 1)")
        axes[0, 0].plot(x, [s["all"]["residual"] for _, s in pts], ls, marker="s", color=C["residual"], label=f"{run} residual (T*)")
        axes[0, 1].plot(x, [s["all"]["t_star"] for _, s in pts], ls, marker="o", color=C["t_star"], label=run)
    axes[0, 0].set(xlabel="RL round (expert-iteration oracle call; GRPO at its final round)", ylabel="nats per proof token",
                   title="KL(pi_RL || pi_pre^T) on pi_RL's own dev rollouts")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(1, color="grey", lw=0.8)
    axes[0, 1].set(xlabel="RL round", ylabel="T*", title="best-matching pretrain temperature", yscale="log")
    axes[0, 1].legend(fontsize=8)
    # (c) per-band breakdown of each run's final checkpoint
    w = 0.8 / (2 * len(runs))
    for i, (run, pts) in enumerate(sorted(runs.items())):
        s = max(pts)[1]
        for j, key in enumerate(("raw", "residual")):
            ys = [s["band"][b][key] if s["band"][b] else 0 for b in BANDS]
            axes[1, 0].bar([k + (2 * i + j) * w for k in range(len(BANDS))], ys, w, color=C[key],
                           alpha=1 if "ei" in run else 0.5, label=f"{run} {key}")
    axes[1, 0].set_xticks([k + 0.4 - w / 2 for k in range(len(BANDS))], BANDS)
    axes[1, 0].set(xlabel="target body lines", ylabel="nats per token", title="final checkpoint, by band")
    axes[1, 0].legend(fontsize=8)
    # (d) T* per band for every checkpoint of the EI run
    ei = sorted(runs.get("v4_untied-ei_s0", []))
    for rnd, s in ei:
        if rnd == 0:
            continue
        ys = [s["band"][b]["t_star"] if s["band"][b] else float("nan") for b in BANDS]
        axes[1, 1].plot(BANDS, ys, marker="o", color=plt.cm.viridis(rnd / max(r for r, _ in ei)), label=f"round {rnd}")
    axes[1, 1].axhline(1, color="grey", lw=0.8)
    axes[1, 1].set(xlabel="target body lines", ylabel="T*", title="v4_untied-ei_s0: T* by band and round", yscale="log")
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tempkl.png"), dpi=130)
    print("wrote", os.path.join(HERE, "tempkl.png"))


if __name__ == "__main__":
    main()
