"""Render observed training diagnostics; no held-out outcomes or new inferential fit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifacts)
    stem = Path(args.out)
    rows, inputs = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    colors = {101: "#2563eb", 102: "#c2410c", 103: "#15803d"}
    for arm, style in (("cost", "--"), ("no_cost", "-")):
        for seed in (101, 102, 103):
            source = artifact / arm / f"main_s{seed}/metrics.jsonl"
            records = [json.loads(line) for line in source.read_text().splitlines()]
            inputs[str(source)] = sha(source)
            for row in records:
                rows.append(dict(arm=arm, seed=seed, **row))
            for axis, key in zip(axes, ("attempts", "accuracy"), strict=True):
                values = np.array([row[key] for row in records])
                smooth = np.convolve(values, np.ones(20) / 20, mode="valid")
                axis.plot(
                    np.arange(20, len(values) + 1),
                    smooth,
                    style,
                    color=colors[seed],
                    label=f"{arm}, seed {seed}",
                )
    axes[0].set_ylabel("Mean attempts per training episode")
    axes[1].set_ylabel("Selected-proof training correctness")
    axes[0].set_ylim(0.9, 5)
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("Completed updates")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Training diagnostics: 20-update trailing means")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + ext), dpi=200)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        fields = ("arm", "seed", "completed_updates", "attempts", "accuracy", "reward", "seconds")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(
                scope="Training diagnostics only; not held-out evidence or confidence intervals.",
                trailing_window_updates=20,
                input_hashes=inputs,
                plot_script_sha256=sha(__file__),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
