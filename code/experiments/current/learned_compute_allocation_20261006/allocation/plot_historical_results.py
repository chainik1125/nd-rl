"""Plot registered descriptive historical estimates, without new statistics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_live_results import interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, stem = Path(args.results), Path(args.out)
    result = json.loads(source.read_text())
    panels = [
        ("Combined (72)", result["combined"]),
        ("Historical textbook_dev (58)", result["by_source_role"]["textbook_dev"]),
        ("Historical rl_train role (14)", result["by_source_role"]["rl_train"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (title, panel) in zip(axes, panels, strict=True):
        for arm, offset, color in [("no_cost", -0.08, "#2563eb"), ("cost", 0.08, "#b45309")]:
            interval(
                axis,
                np.arange(3) + offset,
                [
                    panel["arms"][arm][method]["correct_probability"]
                    for method in ("learned", "fixed1", "fixed4")
                ],
                scale=100,
                fmt="o",
                color=color,
                label=arm,
            )
        for position, (arm, method) in enumerate(
            [("fixed1_control", "fixed1"), ("fixed4_control", "fixed4"), ("format_only", "fixed1")],
            start=3,
        ):
            interval(
                axis,
                [position],
                [panel["arms"][arm][method]["correct_probability"]],
                scale=100,
                fmt="D",
                color="black",
            )
        axis.set_xticks(
            np.arange(6),
            ["Learned", "First", "Fixed4", "Fixed1 train", "Fixed4 train", "Format first"],
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_title(title)
        axis.set_ylabel("Selected proof correct (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Reused historical benchmark: native categorical readout; "
        "at least one known pretraining exposure"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=200)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "arm", "method", "metric", "mean", "ci_low", "ci_high"])
        for title, panel in panels:
            for arm, methods in panel["arms"].items():
                for method, metrics in methods.items():
                    for metric, estimate in metrics.items():
                        writer.writerow(
                            [
                                title,
                                arm,
                                method,
                                metric,
                                estimate["mean"],
                                estimate["ci_low"],
                                estimate["ci_high"],
                            ]
                        )
    provenance = {
        "source": str(source),
        "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interval_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("plot_live_results.py").read_bytes()
        ).hexdigest(),
        "frozen_protocol_sha256": result["provenance"]["frozen_protocol_sha256"],
        "scope": result["scope"],
        "uncertainty_scope": result["uncertainty_scope"],
    }
    stem.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
