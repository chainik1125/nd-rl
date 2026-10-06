"""Render saved registered causal estimates with explicit populations and endpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(axis, rows, title):
    means = np.array([row["mean"] for row in rows]) * 100
    lows = np.array([row["ci95"][0] for row in rows]) * 100
    highs = np.array([row["ci95"][1] for row in rows]) * 100
    positions = np.arange(len(rows))
    axis.axvline(0, color="#a0a0a0", linewidth=1, zorder=0)
    axis.errorbar(
        means,
        positions,
        xerr=np.array([means - lows, highs - means]),
        fmt="none",
        ecolor="#414141",
        capsize=4,
        linewidth=1.5,
    )
    for index, row in enumerate(rows):
        axis.scatter(means[index], positions[index], color=row["color"], s=65, zorder=3)
    axis.set_yticks(positions, [row["label"] for row in rows])
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=12, weight="bold", pad=16)
    axis.set_xlabel("Accuracy change (percentage points)")
    axis.grid(axis="x", color="#e7e7e7", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.margins(y=0.23, x=0.12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--split", choices=("Development", "Held-out test"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.metrics).read_text())
    if not data["primary"]["available"]:
        raise ValueError("Registered fixed-weight primary endpoint is unavailable")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    primary = data["primary"]
    first_rows = []
    for key, label, color in (
        ("old_mass_change", "Change in mass on\nold valid proofs", "#c66b35"),
        ("new_correct_mass", "Mass on\nnew valid proofs", "#318a70"),
        ("delta_continuation", "Net change", "#345f8c"),
    ):
        first_rows.append(
            {"metric": key, "label": label, "color": color, **primary["metrics"][key]}
        )
    second_rows = []
    for key, label in (
        ("one_extra64", "One extra, then select"),
        ("policy_forced1_32", "Force one, then resume policy"),
        ("policy_forced2_16", "Force two, then resume policy"),
        ("policy_forced4_16", "Force four, then resume policy"),
    ):
        group = data["common_first16"][key]
        if not group["available"] or group["planned_theorems"] != 16:
            raise ValueError("Require the complete registered common 16-theorem comparison")
        second_rows.append(
            {
                "condition": key,
                "label": label,
                "color": "#345f8c",
                **group["metrics"]["delta_continuation"],
            }
        )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), gridspec_kw={"wspace": 1.02})
    draw(
        axes[0],
        first_rows,
        f"One extra attempt, then select\nPrimary: {primary['planned_theorems']} theorems",
    )
    draw(axes[1], second_rows, "Continuation horizons\nSecondary: same 16 theorems")
    figure.suptitle(f"{args.split}: continuation under categorical proof selection", fontsize=15)
    caption = (
        "Changes compare continuation with selecting from the proofs already available. "
        "Correctness integrates categorical proof selection; mass is selection probability on valid proofs.\n"
        "Force one/two/four means a minimum number of additional attempts before resuming the policy. "
        "All old candidates remain available.\n"
        "Equal weights for two checkpoints and forced states after attempts 1 and 2. "
        "Bars: 95% family-stratified theorem-bootstrap intervals, conditional on these models and draws.\n"
        "The right panel uses the same 16 theorems from the original 64/32/16/16 panels. "
        "Intervals do not test differences between horizons."
    )
    figure.text(0.02, 0.025, caption, fontsize=9, color="#454545", va="bottom")
    figure.subplots_adjust(left=0.17, right=0.985, top=0.77, bottom=0.32)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"continuation.{suffix}", dpi=180, facecolor="white")
    plt.close(figure)
    provenance = {
        "input": str(Path(args.metrics).resolve()),
        "input_sha256": sha(args.metrics),
        "renderer_sha256": sha(__file__),
        "split_label": args.split,
        "primary_rows": first_rows,
        "common16_rows": second_rows,
        "transform": "Saved means and CI endpoints multiplied by 100; no statistics refitted.",
        "caption": caption,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "renderer.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
