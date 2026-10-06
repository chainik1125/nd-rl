"""Reproduce the development-only context/interface figure from private outcomes.

This analysis is downstream of generation; no checker information is returned to
the model. Four continuation attempts are first averaged within each theorem.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
ARTIFACTS = ROOT / "artifacts/learned_compute_allocation_20261006/context"
CONDITIONS = {
    "base_0": ("base_dev48_v1/history_0", "No prefix"),
    "base_1": ("base_dev48_v1/history_1", "One prior attempt"),
    "base_4": ("base_dev48_v1/history_4", "Four prior attempts"),
    "boundary": ("base_dev48_boundary/history_1", "3-token attempt marker"),
    "eos": ("base_dev48_eos/history_1", "3 EOS tokens"),
    "unrelated": ("base_dev48_unrelated/history_1", "Unrelated history"),
    "adapt_0": ("format200_dev48_v1/history_0", "Adapted, no history"),
    "adapt_1": ("format200_dev48_v1/history_1", "Adapted, one history"),
    "adapt_4": ("format200_dev48_v1/history_4", "Adapted, four histories"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    inputs = {}
    for name, (relative, _) in CONDITIONS.items():
        path = ARTIFACTS / relative / "private.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mapped = {
            row["theorem_id"]: {
                label["candidate_id"]: float(label["valid"]) for label in row["labels"]
            }
            for row in rows
        }
        if len(mapped) != len(rows):
            raise ValueError("Duplicate theorem ID")
        outcomes[name] = mapped
        inputs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    theorem_ids = sorted(outcomes["base_0"])
    for mapped in outcomes.values():
        if sorted(mapped) != theorem_ids or any(len(x) != 4 for x in mapped.values()):
            raise ValueError("Conditions require the same theorem panel and four attempts")
    values = {
        name: np.array([np.mean(list(mapped[tid].values())) for tid in theorem_ids])
        for name, mapped in outcomes.items()
    }
    draws = np.random.default_rng(61006).integers(0, len(theorem_ids), (50000, len(theorem_ids)))

    def interval(array):
        lo, hi = np.quantile(array[draws].mean(axis=1), [0.025, 0.975])
        return {"mean": float(array.mean()), "low": float(lo), "high": float(hi)}

    stats = {name: interval(array) for name, array in values.items()}
    pairs = [
        ("adapt_0", "base_0"),
        ("adapt_1", "base_1"),
        ("adapt_4", "base_4"),
        ("adapt_1", "adapt_0"),
        ("adapt_4", "adapt_0"),
        ("eos", "base_0"),
    ]
    differences = {f"{a}_minus_{b}": interval(values[a] - values[b]) for a, b in pairs}
    report = {
        "scope": "development; 48 reused theorems; one generation seed; one adaptation seed",
        "sampling_unit": "theorem, with four attempts averaged within each theorem",
        "bootstrap": {"draws": 50000, "seed": 61006, "confidence": 0.95},
        "conditions": stats,
        "paired_differences": differences,
        "inputs": inputs,
    }
    (output / "context_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    order = ["base_0", "eos", "boundary", "base_1", "unrelated", "base_4"]
    means = np.array([stats[name]["mean"] for name in order]) * 100
    error = (
        np.array(
            [
                [stats[n]["mean"] - stats[n]["low"] for n in order],
                [stats[n]["high"] - stats[n]["mean"] for n in order],
            ]
        )
        * 100
    )
    axes[0].barh(range(len(order)), means, xerr=error, color="#547c96", capsize=3)
    axes[0].set_yticks(range(len(order)), [CONDITIONS[name][1] for name in order])
    axes[0].invert_yaxis()
    axes[0].set(
        xlim=(0, 103),
        xlabel="Valid continuations (%)",
        title="Original model: marker choice matters",
    )
    for prefix, label, color in [
        ("base", "Original r16", "#547c96"),
        ("adapt", "200-step format adaptation", "#b66630"),
    ]:
        points = [stats[f"{prefix}_{i}"] for i in [0, 1, 4]]
        y = np.array([point["mean"] for point in points]) * 100
        err = (
            np.array(
                [[p["mean"] - p["low"] for p in points], [p["high"] - p["mean"] for p in points]]
            )
            * 100
        )
        axes[1].errorbar([0, 1, 4], y, yerr=err, label=label, color=color, marker="o", capsize=3)
    axes[1].set(
        xticks=[0, 1, 4],
        ylim=(-3, 103),
        xlabel="Prior attempts in context",
        ylabel="Valid continuations (%)",
        title="Adaptation restores much of the loss",
    )
    axes[1].legend(loc="center right", fontsize=9)
    fig.suptitle(
        "Development interface audit — 48 theorems, 4 continuations per condition", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(output / "context_interface.png", dpi=200)
    fig.savefig(output / "context_interface.svg")
    plt.close(fig)
    print(json.dumps({"conditions": stats, "paired_differences": differences}, indent=2))


if __name__ == "__main__":
    main()
