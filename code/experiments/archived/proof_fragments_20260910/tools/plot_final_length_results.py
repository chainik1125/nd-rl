"""Render prespecified reference-length outcomes from saved final aggregates."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/proof-fragments-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/final_seed1_v1.json"
NAMES = ["adapt", "selected_full_s1_f40", "selected_reset_s1_f40", "uniform_full_s1_f40", "uniform_reset_s1_f40"]
LABELS = ["Adapted starting model", "Selected, full starts", "Selected, reset curriculum", "Uniform, full starts", "Uniform, reset curriculum"]
COLORS = ["#444444", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    data = json.loads(SOURCE.read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    bins = ["2-6", "7-9", "10-12"]
    for ax, metric, ylabel in zip(axes, ["coverage_at_k", "attempt_success"], ["Theorems solved in eight attempts (%)", "Successful attempts (%)"]):
        for i, (name, label, color) in enumerate(zip(NAMES, LABELS, COLORS)):
            vals = [data["models"][name][metric][b] for b in bins]
            means = np.array([v["mean"] for v in vals]) * 100
            low = np.array([v["context_bootstrap95"][0] for v in vals]) * 100
            high = np.array([v["context_bootstrap95"][1] for v in vals]) * 100
            x = np.arange(3) + (i - 2) * .13
            ax.errorbar(x, means, yerr=[means-low, high-means], fmt="o", label=label, color=color, capsize=3, markersize=5)
        ax.set(xticks=range(3), xticklabels=["2–6\n256 targets", "7–9\n256 targets", "10–12\n256 targets"], ylabel=ylabel, xlabel="Reference proof lines, including premises and assumptions", ylim=(0, 100))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Final reference-length bins · primary RL endpoints · one online training seed", fontsize=13)
    fig.tight_layout()
    outputs = {}
    for ext in ["png", "svg", "pdf"]:
        path = ROOT / "figures" / f"final_main_reference_length_v1.{ext}"
        fig.savefig(path, bbox_inches="tight")
        outputs[str(path)] = sha(path)
    receipt = {"input_sha256": {str(SOURCE): sha(SOURCE), str(Path(__file__).resolve()): sha(__file__)}, "outputs": outputs, "scope": "All three frozen reference-length bins; 256 targets/bin, 8 attempts/target. Context percentile bootstrap conditional on one online training seed. Reference length is a witness upper bound, not minimum proof difficulty."}
    target = ROOT / "figures/final_main_reference_length_manifest_v1.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(target)

if __name__ == "__main__":
    main()
