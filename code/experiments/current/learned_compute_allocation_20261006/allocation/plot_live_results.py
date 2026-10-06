"""Plot registered held-out estimates without fitting additional statistics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def interval(axis, positions, estimates, scale=1, **kwargs):
    means = np.array([item["mean"] for item in estimates]) * scale
    low = np.array([item["ci_low"] for item in estimates]) * scale
    high = np.array([item["ci_high"] for item in estimates]) * scale
    # Draw stored endpoints directly: constant metrics can have a bootstrap
    # endpoint a few ulps beyond their mean, making relative error bars negative.
    marker = kwargs.pop("fmt", "o")
    color = kwargs.get("color")
    axis.vlines(positions, low, high, color=color)
    axis.plot(positions, means, marker, linestyle="none", **kwargs)


def tradeoffs(result, stem):
    """Display frozen estimates and training-seed points; compute no new statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"no_cost": "#2563eb", "cost": "#b45309"}
    styles = {"learned": "o", "counter": "s", "fixed1": "^", "fixed4": "v"}
    for row, outcome in enumerate(("correct_probability", "utility_at_cost_0.03")):
        for col, work in enumerate(("attempts", "total_forward_calls")):
            axis = axes[row, col]
            for arm, color in colors.items():
                for method, marker in styles.items():
                    estimates = result["arms"][arm][method]
                    x, y = estimates[work], estimates[outcome]
                    axis.hlines(y["mean"] * 100, x["ci_low"], x["ci_high"], color=color)
                    axis.vlines(x["mean"], y["ci_low"] * 100, y["ci_high"] * 100, color=color)
                    axis.scatter(
                        x["mean"],
                        y["mean"] * 100,
                        marker=marker,
                        color=color,
                        label=f"{arm}: {method}" if row == col == 0 else None,
                    )
                    for seed in (101, 102, 103):
                        point = result["per_seed"][f"{arm}:{seed}"][method]
                        axis.scatter(
                            point[work],
                            point[outcome] * 100,
                            marker=marker,
                            color=color,
                            alpha=0.3,
                            s=18,
                        )
            for arm, method, marker in (
                ("fixed1_control", "fixed1", "X"),
                ("fixed4_control", "fixed4", "P"),
                ("format_only", "fixed1", "D"),
            ):
                point = result["arms"][arm][method]
                axis.hlines(
                    point[outcome]["mean"] * 100,
                    point[work]["ci_low"],
                    point[work]["ci_high"],
                    color="black",
                    linewidth=0.7,
                )
                axis.vlines(
                    point[work]["mean"],
                    point[outcome]["ci_low"] * 100,
                    point[outcome]["ci_high"] * 100,
                    color="black",
                    linewidth=0.7,
                )
                axis.scatter(
                    point[work]["mean"],
                    point[outcome]["mean"] * 100,
                    marker=marker,
                    color="black",
                    s=50,
                    label=arm if row == col == 0 else None,
                )
            axis.set_xlabel("Expected attempts" if col == 0 else "Expected charged model forwards")
            axis.set_ylabel(
                "Correct (%)" if row == 0 else "Utility at cost 0.03 (percentage points)"
            )
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Categorical policy: means, three training-seed points, theorem-bootstrap intervals"
    )
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_name(stem.name + "_tradeoffs").with_suffix("." + extension), dpi=200)
    plt.close(fig)


def readout_sensitivity(result, stem):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    methods = ("learned", "learned_greedy_index", "greedy_stop_greedy_index")
    for arm, offset, color in (("no_cost", -0.07, "#2563eb"), ("cost", 0.07, "#b45309")):
        for axis, metric, scale in zip(
            axes,
            ("correct_probability", "attempts", "utility_at_cost_0.03"),
            (100, 1, 100),
            strict=True,
        ):
            interval(
                axis,
                np.arange(3) + offset,
                [result["arms"][arm][method][metric] for method in methods],
                scale=scale,
                fmt="o",
                color=color,
                label=arm,
            )
    for axis, ylabel in zip(
        axes, ("Correct (%)", "Expected attempts", "Utility at cost 0.03 (pp)"), strict=True
    ):
        axis.set_xticks(np.arange(3), ["Categorical", "Greedy index", "Greedy stop + index"])
        axis.tick_params(axis="x", labelrotation=15)
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Prespecified readout sensitivity on identical generated paths")
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_name(stem.name + "_readouts").with_suffix("." + extension), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True, help="Output stem for PNG/PDF/provenance")
    args = parser.parse_args()
    source = Path(args.results)
    result = json.loads(source.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = ["Learned", "First", "Count", "Context"]
    methods = ["learned", "fixed1", "counter", "context_threshold"]
    for arm, offset, color, title in [
        ("no_cost", -0.08, "#2563eb", "No cost"),
        ("cost", 0.08, "#b45309", "Cost 0.03"),
    ]:
        for axis, metric, scale in [
            (axes[0], "correct_probability", 100),
            (axes[1], "attempts", 1),
        ]:
            interval(
                axis,
                np.arange(4) + offset,
                [result["arms"][arm][method][metric] for method in methods],
                scale=scale,
                fmt="o",
                color=color,
                label=title,
            )
        comparisons = ["fixed1", "counter", "context_threshold"]
        interval(
            axes[2],
            np.arange(3) + offset,
            [
                result["contrasts"][f"{arm}:learned-minus-{method}"]["reward_native"]
                for method in comparisons
            ],
            scale=100,
            fmt="o",
            color=color,
            label=title,
        )
    for axis in axes[:2]:
        axis.set_xticks(np.arange(4), labels)
    axes[0].set_ylabel("Selected proof correct (%)")
    axes[1].set_ylabel("Expected attempts")
    axes[2].set_xticks(np.arange(3), ["vs first", "vs count", "vs context"])
    axes[2].set_ylabel("Native utility difference (percentage points)")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    fig.suptitle("Joint generator and stopping policy: held-out theorem estimates")
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(stem.with_suffix("." + extension), dpi=200)
    tradeoffs(result, stem)
    readout_sensitivity(result, stem)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "condition", "method", "metric", "mean", "ci_low", "ci_high"])
        for arm, methods in result["arms"].items():
            for method, metrics in methods.items():
                for metric, values in metrics.items():
                    writer.writerow(
                        [
                            "arm",
                            arm,
                            method,
                            metric,
                            values["mean"],
                            values["ci_low"],
                            values["ci_high"],
                        ]
                    )
        for comparison, metrics in result["contrasts"].items():
            for metric, values in metrics.items():
                writer.writerow(
                    [
                        "paired_contrast",
                        comparison,
                        "",
                        metric,
                        values["mean"],
                        values["ci_low"],
                        values["ci_high"],
                    ]
                )
    provenance = {
        "figure_stems": [
            str(stem),
            str(stem.with_name(stem.name + "_tradeoffs")),
            str(stem.with_name(stem.name + "_readouts")),
        ],
        "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interval_scope": result["uncertainty_scope"],
        "source": str(source),
        "frozen_protocol_sha256": result["provenance"]["frozen_protocol_sha256"],
    }
    stem.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
