"""Presentation only: historical source-role metrics from frozen evaluations."""

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
    parser.add_argument("--selection", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    head = json.loads(Path(args.head).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    roles = [("textbook_dev", 58), ("rl_train", 14)]
    methods = [
        ("pooled", "Whole-pool selector (primary)", "#38658b"),
        ("factorized", "Per-proof selector (control)", "#34856f"),
        ("bce", "Binary critic (secondary)", "#b9773d"),
        ("frozen_head", "Head on frozen original model", "#735585"),
    ]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    figure.suptitle(
        "Historical source roles: selecting a proof from eight attempts", fontsize=18, y=1.03
    )
    for ax, (role, n) in zip(axes, roles, strict=True):
        head_role = next(
            row for row in head["strata"] if row["axis"] == "family" and row["value"] == role
        )
        assert head_role["theorems"] == n
        role_rows = {}
        for group, record in selection["groups"].items():
            role_rows[group] = next(
                row for row in record["strata"] if row["axis"] == "family" and row["value"] == role
            )
            assert role_rows[group]["metrics"]["theorems"] == n
        for y, (key, _, color) in enumerate(methods):
            value = (
                head_role["metrics"]["frozen_head"]["accuracy"]
                if key == "frozen_head"
                else role_rows[key]["metrics"]["accuracy"]
            )
            ax.scatter(value * 100, y, color=color, s=95, marker="D", zorder=4)
            ax.text(value * 100 + 0.8, y - 0.07, f"{value * 100:.2f}%", fontsize=10.5, color=color)
        references = head_role["metrics"]
        for key, label, style, color in [
            ("random", "Random candidate", ":", "#777777"),
            ("public_features", "Feature baseline", "--", "#a05c42"),
        ]:
            ax.axvline(references[key]["accuracy"] * 100, color=color, linestyle=style, label=label)
        ax.axvline(
            references["frozen_head"]["oracle_accuracy"] * 100,
            color="#777d40",
            linestyle="-.",
            label="Oracle availability",
        )
        ax.set_title(f"Source role: {role}\n{n} theorems", fontsize=14, pad=12)
        ax.set_xlim(20, 72)
        ax.set_ylim(3.6, -0.5)
        ax.set_yticks(range(4), [row[1] for row in methods])
        ax.set_xlabel("Selected proof correct (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_axisbelow(True)
    figure.legend(
        *axes[1].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.64, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    figure.text(
        0.04,
        0.08,
        "72 reused benchmark theorems; source-role labels are not exposure counts. "
        "All models and policies were chosen on MAIN development.",
        fontsize=10.5,
    )
    figure.text(
        0.04,
        0.04,
        "Points: three-seed means, except the single frozen head. Role panels are descriptive; "
        "paired 72-theorem uncertainty is reported separately.",
        fontsize=10.5,
    )
    figure.text(
        0.04,
        0.005,
        "Oracle uses held-back verifier labels, unavailable at inference. No historical tuning. "
        "All 72 syntax-then-likelihood: 59.72% (not plotted by role).",
        fontsize=10.5,
    )
    figure.subplots_adjust(left=0.27, right=0.98, top=0.77, bottom=0.25, wspace=0.15)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out / ("roles." + suffix), dpi=170, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(__file__, out / "renderer.py")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "scope": "Presentation only; saved role accuracies, no new estimates",
                "inputs": {str(p): sha(p) for p in (Path(args.selection), Path(args.head))},
                "renderer_sha256": sha(__file__),
                "outputs": {p.name: sha(p) for p in out.glob("roles.*")},
                "unit_conversion": "Multiply saved probability values by100 only",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
