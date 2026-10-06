"""Presentation-only plot of already registered paired contrasts."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
source = ART / "allocation/final_main_summary_fp32_v2.json"
result = json.loads(source.read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
for axis, metric, scale, title in zip(
    axes,
    ("correct_probability", "attempts", "total_forward_tokens"),
    (100, 1, 1),
    ("Returned correctness (percentage points)", "Expected attempts", "Forward-input tokens"),
    strict=True,
):
    axis.axvline(0, color="#777777", linewidth=0.8)
    for y, arm, color in ((1, "cost", "#b45309"), (0, "no_cost", "#2563eb")):
        estimate = result["contrasts"][f"{arm}:learned-minus-fixed1"][metric]
        axis.hlines(
            y, estimate["ci_low"] * scale, estimate["ci_high"] * scale, color=color, linewidth=2
        )
        axis.scatter(estimate["mean"] * scale, y, color=color, zorder=3)
        for seed in (101, 102, 103):
            row = result["per_seed"][f"{arm}:{seed}"]
            axis.scatter(
                (row["learned"][metric] - row["fixed1"][metric]) * scale,
                y,
                color=color,
                alpha=0.25,
                s=15,
            )
    axis.set_yticks((0, 1), ("Correctness-only", "Cost-aware"))
    axis.set_ylim(-0.5, 1.5)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Learned policy minus own first proof")
fig.suptitle("Extra allocation does not improve returned correctness", fontsize=13)
folders = (
    ART / "report_figures/live_main_v2",
    ROOT
    / "experiment-summaries/2026-10-06-learned-compute-allocation/allocation/figures/live_main_v2",
)
provenance = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "plot_source": str(Path(__file__).resolve()),
    "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "original_registry_sha256": hashlib.sha256(
        (ART / "final_evaluation/main_protocol.json").read_bytes()
    ).hexdigest(),
    "caption": (
        "400 fresh test theorems, two paths per theorem, three fixed training seeds per arm. "
        "Points show seed differences and arm means; bars are saved paired family-stratified "
        "theorem-bootstrap 95% intervals, conditional on the trained models. Forward-input "
        "tokens are expected charged policy work, not forced-eight-path measurement expense. "
        "No new statistics or fits."
    ),
}
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"learned_minus_first.{extension}", dpi=180)
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
