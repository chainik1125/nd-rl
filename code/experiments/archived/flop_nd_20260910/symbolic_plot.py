"""Standalone figure for the frozen symbolic diagnostic."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

audit = json.loads((common.ROOT / "reports/symbolic_audit.json").read_text())
groups = audit["evaluations"]
depths = [5, 15, 31, 63]
lengths = np.array(depths) + 1
def values(tag, family):
    return np.array([100 * groups[tag]["groups"][f"{family}/d{d}"]["successes"] /
                     groups[tag]["groups"][f"{family}/d{d}"]["attempts"] for d in depths])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, family, title in zip(axes, ["spine", "compound_decoy"],
                            ["Ordinary conjunction spine", "Compound distractor branches"]):
    for mode, color, marker in [("full", "#1b6ca8", "o"), ("reset", "#d45c24", "s")]:
        data = np.array([values(f"{mode}_s{s}", family) for s in [4101, 4102]])
        ax.plot(lengths, data.mean(0), marker=marker, color=color, label=f"PPO {mode}: mean of 2 seeds")
        ax.fill_between(lengths, data.min(0), data.max(0), color=color, alpha=.12)
    initial = np.array([values(f"initial_s{s}", family) for s in [4101, 4102]])
    ax.plot(lengths, initial.mean(0), color="#777777", linestyle=":", label="Exact initial policies: mean")
    ax.fill_between(lengths, initial.min(0), initial.max(0), color="#999999", alpha=.13)
    ax.plot(lengths, values("random_minimal", family), marker="x", color="#773f9a", linestyle="--",
            label="Random: same d-action budget")
    ax.plot(lengths, values("random_loose", family), marker="+", color="#2a8c61", linestyle="--",
            label="Random: d+3 action budget")
    ax.set_title(title)
    ax.set_xlabel("Dependency-pruned certificate lines (d+1)")
    ax.set_xticks(lengths)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", alpha=.18)
axes[0].set_ylabel("Successful attempts (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(.5, -.08), frameon=False, fontsize=9)
fig.suptitle("Two six-line seeds enable long projection efficiency, without branch-strategy transfer", fontsize=12)
fig.subplots_adjust(bottom=.29, top=.84, wspace=.10)
out = common.ROOT / "results/symbolic_figures"
out.mkdir(exist_ok=True)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(out / f"symbolic_transfer.{ext}", dpi=180, bbox_inches="tight")
