"""Presentation only: render previously registered MAIN selection estimates."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.metrics).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    methods = [
        ("pooled", "Whole-pool selector\nPrimary", "#38658b"),
        ("factorized", "Per-proof selector\nControl", "#34856f"),
        ("bce", "Binary critic\nSecondary", "#b9773d"),
    ]
    if data["split"] != "test" or any(
        data["groups"][key]["primary"]["theorems"] != 400 for key, _, _ in methods
    ):
        raise ValueError("This presentation is specific to the registered400-theorem MAIN test")
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 5.8))
    figure.suptitle(
        "Selecting a proof from eight attempts: 400 held-out theorems", fontsize=19, y=0.98
    )
    labels = [label for _, label, _ in methods]
    for y, (key, _, color) in enumerate(methods):
        record = data["groups"][key]
        seed_values = record["seed_values"]
        for offset, seed in zip((-0.12, 0, 0.12), seed_values, strict=True):
            left.scatter(seed["accuracy"] * 100, y + offset, s=54, color=color, alpha=0.55)
        left.scatter(
            record["primary"]["accuracy"] * 100, y, marker="D", s=100, color=color,
            edgecolor="white", linewidth=1.0, zorder=4,
        )
        contrast = record["paired_vs_feature"]
        center, low, high = [contrast[k] * 100 for k in ("mean", "low", "high")]
        right.errorbar(
            center, y, xerr=[[center - low], [high - center]], fmt="o", color=color,
            ecolor=color, capsize=5, markersize=9, linewidth=2,
        )
    for key, name, style, color in [
        ("random", "Random candidate", ":", "#777777"),
        ("feature", "Feature baseline", "--", "#a05c42"),
        ("oracle", "Oracle availability", "-.", "#777d40"),
    ]:
        left.axvline(data["baselines"][key] * 100, color=color, linestyle=style, label=name)
    left.scatter([], [], s=54, color="#555555", alpha=0.55, label="Individual training seeds")
    left.scatter([], [], s=100, color="#555555", marker="D", label="Three-seed mean")
    left.set_title("Selected proof correctness", pad=14)
    left.set_xlabel("Selected proof correct (%)")
    left.set_xlim(68, 92)
    left.set_yticks(range(3), labels)
    left.legend(loc="lower left", fontsize=10, frameon=False, bbox_to_anchor=(0.0, -0.02))
    right.set_title("Paired difference from feature baseline", pad=14)
    right.axvline(0, color="#777777", linewidth=1)
    right.set_xlabel("Accuracy difference (percentage points)")
    right.set_xlim(-9, 1)
    right.set_yticks(range(3), labels)
    for ax in (left, right):
        ax.set_ylim(3.25, -0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    figure.text(
        0.04, 0.075,
        "400 theorems, equally split by family. Seeds 0/1/2 averaged within theorem; "
        "prespecified candidate order and exact score ties retained.", fontsize=10.5,
    )
    figure.text(
        0.04, 0.04,
        "Bars: 95% paired theorem-bootstrap intervals, stratified by family and conditional "
        "on these trained models. No method beats the feature baseline.", fontsize=10.5,
    )
    figure.text(
        0.04, 0.005,
        "Oracle availability uses held-back verifier labels, unavailable at inference.",
        fontsize=10.5,
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.81, bottom=0.23, wspace=0.68)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out / ("selection." + suffix), dpi=170, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(__file__, out / "renderer.py")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Presentation-only rendering of saved registered estimates; no new analysis"
                ),
                "input": str(Path(args.metrics).resolve()),
                "input_sha256": sha(args.metrics),
                "renderer_sha256": sha(__file__),
                "outputs": {p.name: sha(p) for p in out.glob("selection.*")},
                "unit_conversion": "saved probabilities multiplied by100 only",
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
