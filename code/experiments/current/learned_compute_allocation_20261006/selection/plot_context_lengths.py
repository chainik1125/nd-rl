"""Plot measured main development chooser input lengths, without test outcomes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    for name, label in [
        ("pooled", "Eight-proof pool (320 inputs)"),
        ("factorized", "One proof at a time (2560 inputs)"),
    ]:
        x = np.array(data[name]["sorted_lengths"])
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label)
    ax.axvline(1024, color="gray", linestyle="--", linewidth=1)
    ax.axvline(8192, color="gray", linestyle=":", linewidth=1)
    ax.text(1024, 0.04, "Checkpoint config: 1024", rotation=90, ha="right", va="bottom", fontsize=8)
    ax.text(8192, 0.04, "Declared cap: 8192", rotation=90, ha="right", va="bottom", fontsize=8)
    ax.set(
        xscale="log",
        xlabel="Serialized chooser input length (tokens)",
        ylabel="Fraction of inputs at or below length",
        ylim=(0, 1.01),
        title="Chooser input lengths on frozen main development pools",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.15)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
