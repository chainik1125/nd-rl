"""Family/length extrapolation curves from fully replay-audited results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import common
import plot_results as P


def plot(path, prefix):
    report = json.loads(Path(path).read_text())
    if report.get("status") != "complete":
        raise ValueError("Only complete audited analyses may enter a report figure")
    if report["population"] != "structured":
        raise ValueError("Family-length curves require the structured population")
    summaries = report["summaries"]
    families = [("projection_atomic", "Atomic distractors"), ("projection_compound", "Compound distractors"),
                ("implication_projection", "Projection + implication elimination"), ("scope_projection", "Projection inside assumption scope")]
    lengths = [7, 9, 13, 17, 25] if report["study"] == "minimal" else [7, 9, 13]
    P.style()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    colors = {"full": "#E69F00", "reset": "#009E73", "bc": "#0072B2", "initial": "#888888", "uniform": "#000000"}
    labels = {"full": "PPO full starts", "reset": "PPO backward starts", "bc": "Behavior cloning", "initial": "Initial network", "uniform": "Uniform actions"}
    metric = "first" if report["profile"] == "greedy" else "any"
    for ax, (family, title) in zip(axes, families):
        for mode in ("initial", "uniform", "full", "bc", "reset"):
            keys = ["uniform"] if mode == "uniform" else [f"{mode}_s{s}" for s in report["seeds"]]
            values = np.array([[summaries[key][f"family_length:{family}:{length}"][metric] /
                               summaries[key][f"family_length:{family}:{length}"]["theorems"] for length in lengths] for key in keys])
            ax.plot(lengths, values.mean(0), label=labels[mode], color=colors[mode],
                    ls=":" if mode in ("uniform", "initial") else "-", lw=1.5, zorder=2)
            for i, row in enumerate(values):
                ax.scatter(np.array(lengths) + (.13 if i else -.13), row, s=14,
                           marker="o" if i == 0 else "s", color=colors[mode], alpha=.7, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(lengths)
        ax.set_xlabel("Verified reference lines")
        ax.set_ylim(-.025, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Verified theorem coverage")
    title = "Training: two six-line proofs and their statements" if report["study"] == "minimal" else "104 proofs (≤6 lines); PPO also receives 512 theorem-only targets"
    title += " · " + report["split"] + (" · greedy" if report["profile"] == "greedy" else " · any of four sampled attempts")
    fig.suptitle("Policies choosing legal ND inferences\n" + title, fontsize=11)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=5, frameon=False, fontsize=9)
    counts = {summaries[key][f"family_length:{family}:{length}"]["theorems"]
              for key in summaries for family, _ in families for length in lengths}
    count_label = str(next(iter(counts))) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
    fig.text(.5, -.12,
        f"Each point: {count_label} statements. Circle = seed {report['seeds'][0]}; square = seed {report['seeds'][1]}; lines = mean. Uniform uses one fixed random stream.",
        ha="center", va="top", fontsize=8)
    fig.text(.5, -.18,
        "PPO: 131,072 sampled actions per run. Cloning: 1,000 full-batch updates on demonstrations. The environment supplies premises and proof syntax.",
        ha="center", va="top", fontsize=8)
    fig.tight_layout()
    P.save(fig, prefix, [Path(path), Path(__file__)],
        "All predeclared symbolic policies on each structured family and length. Lines average the two training seeds; circles/squares show the seeds separately. Uniform has one fixed random seed. BC fits the same supplied demonstrations with a separate, smaller compute budget. Legal actions and premise serialization are supplied by the environment. Reference lengths are verified witnesses, not unrestricted lower bounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(args.analysis, args.prefix)
