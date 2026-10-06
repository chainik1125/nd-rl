"""Development-only interface quality and retained-answer availability figure."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def statistics(folder):
    labels = folder / "private_labels.jsonl"
    config = json.loads((folder / "config.json").read_text())
    data = [json.loads(line) for line in Path(config["data"]).read_text().splitlines()]
    if any(row["split"] != "dev" for row in data):
        raise ValueError("this interface figure is development-only")
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(row["theorem_id"], []).append(row["validities"])
    values = np.asarray(list(groups.values()), dtype=float)
    availability = np.maximum.accumulate(values, axis=-1)
    rng = np.random.default_rng(61)
    indices = rng.integers(0, len(groups), size=(3000, len(groups)))
    result = {}
    for name, observations in ("validity", values), ("oracle", availability):
        means = observations.mean(axis=1)
        boot = means[indices].mean(axis=1)
        result[name] = {
            "mean": means.mean(axis=0),
            "low": np.quantile(boot, 0.025, axis=0),
            "high": np.quantile(boot, 0.975, axis=0),
        }
    return result, hashlib.sha256(labels.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run500", required=True)
    parser.add_argument("--run1000", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    rows, provenance = [], {}
    for label, folder, color in (
        ("500 steps", Path(args.run500), "#757575"),
        ("1,000 steps (chosen)", Path(args.run1000), "#1764ab"),
    ):
        result, digest = statistics(folder)
        provenance[label] = {"run": str(folder), "private_labels_sha256": digest}
        x = np.arange(1, 9)
        for axis, key in zip(axes, ("validity", "oracle"), strict=True):
            s = result[key]
            axis.plot(x, s["mean"], marker="o", label=label, color=color, linewidth=1.6)
            axis.fill_between(x, s["low"], s["high"], color=color, alpha=0.12)
            for i in range(8):
                rows.append(
                    {
                        "condition": label,
                        "measure": key,
                        "attempt": i + 1,
                        "mean": s["mean"][i],
                        "cluster_bootstrap_low": s["low"][i],
                        "cluster_bootstrap_high": s["high"][i],
                    }
                )
    axes[0].set_title("Correctness of each new attempt")
    axes[1].set_title("At least one correct retained proof")
    axes[0].set_ylabel("Probability")
    for axis in axes:
        axis.set_xlabel("Attempt count")
        axis.set_xticks(range(1, 9))
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Format adaptation audit: 16 development theorems, two paths each", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "format_interface_audit.png", dpi=180)
    fig.savefig(out / "format_interface_audit.pdf")
    with (out / "format_interface_audit.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance["uncertainty"] = "3000 bootstrap resamples of theorem clusters, seed61"
    provenance["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "format_interface_audit.provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
