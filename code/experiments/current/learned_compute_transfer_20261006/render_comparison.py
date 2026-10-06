"""Render audited transfer aggregates without selecting models or changing estimators."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
OUT = ROOT / "experiment-summaries/2026-10-06-learned-compute-transfer"
ARM_ORDER = ("phase1_fixed8", "fixed4_control", "cost", "no_cost")
ARM_LABELS = {
    "phase1_fixed8": "Frozen generator\nfixed 8 selector",
    "fixed4_control": "Joint model\ntrained at fixed 4",
    "cost": "Joint model\ncompute penalty",
    "no_cost": "Joint model\ncorrectness only",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results():
    sources = {
        "historical": ART / "historical_comparison.json",
        "selection": ART / "analysis/phase1_summary/metrics.json",
        "live": ART / "live/summary.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in sources.items()}
    scopes = {"textbook72": data["historical"]}
    for scope, selection_scope in (
        ("ordinary187", "ordinary187"),
        ("derived225projection", "derived225"),
    ):
        selection = data["selection"][selection_scope]
        live = data["live"]["scopes"][scope]
        assert selection["theorems"] == live["theorems"]
        metrics = selection["metrics"]
        arms = {
            "phase1_fixed8": {
                "pass1": metrics["pass1_mean8"]["accuracy"],
                "first_candidate": metrics["first_candidate"]["accuracy"],
                "oracle8": metrics["oracle8"]["accuracy"],
                "returned": metrics["pooled_transitions"]["accuracy"],
                "attempts": 8.0,
                "training_seeds": 3,
                "intervals": {
                    key: {
                        "mean": metrics[source]["accuracy"],
                        "ci_low": metrics[source]["ci95"][0],
                        "ci_high": metrics[source]["ci95"][1],
                    }
                    for key, source in (
                        ("pass1", "pass1_mean8"),
                        ("oracle8", "oracle8"),
                        ("returned", "pooled_transitions"),
                    )
                },
            }
        }
        for arm, policy in (
            ("fixed4_control", "fixed4"),
            ("cost", "learned"),
            ("no_cost", "learned"),
        ):
            readouts = live["arms"][arm]
            arms[arm] = {
                "pass1": readouts["fixed1"]["correct_probability"]["mean"],
                "oracle4": readouts["oracle4"]["correct_probability"]["mean"],
                "oracle8": readouts["oracle8"]["correct_probability"]["mean"],
                "returned": readouts[policy]["correct_probability"]["mean"],
                "attempts": readouts[policy]["attempts"]["mean"],
                "training_seeds": 1 if arm == "fixed4_control" else 3,
                "return_policy": policy,
                "intervals": {
                    "pass1": readouts["fixed1"]["correct_probability"],
                    "oracle4": readouts["oracle4"]["correct_probability"],
                    "oracle8": readouts["oracle8"]["correct_probability"],
                    "returned": readouts[policy]["correct_probability"],
                    "attempts": readouts[policy]["attempts"],
                },
                "all_readouts": readouts,
            }
            if arm != "fixed4_control":
                arms[arm]["paired_vs_pass1"] = live["contrasts"][f"{arm}:native-minus-ownfirst"][
                    "correct_probability"
                ]
        scopes[scope] = {
            "scope": scope,
            "rows": live["theorems"],
            "arms": arms,
            "selection_comparisons": {name: metric["accuracy"] for name, metric in metrics.items()},
            "live_per_seed": live["per_model"],
        }
    for scope in scopes.values():
        for arm in scope["arms"].values():
            assert 0 <= arm["pass1"] <= arm["oracle8"] + 1e-6 <= 1 + 1e-6
            assert 0 <= arm["returned"] <= arm["oracle8"] + 1e-6
            assert 1 - 1e-6 <= arm["attempts"] <= 8 + 1e-6
    return {
        "status": "complete",
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in sources.values()},
        "render_source_sha256": sha(Path(__file__)),
        "scopes": scopes,
    }


def draw(scopes, names, stem, title, projection=False):
    fig, axes = plt.subplots(1, len(names), figsize=(7.2 * len(names), 5.9), squeeze=False)
    colors = {"pass1": "#547aa5", "oracle8": "#c5cbd0", "returned": "#21866b"}
    labels = {
        "pass1": "Pass@1",
        "oracle8": "Any correct in 8 (oracle)",
        "returned": "Returned proof correct",
    }
    titles = {
        "textbook72": "Textbook 72",
        "ordinary187": "Charles: 187 ordinary problems",
        "derived225projection": "Charles: 225 sequent projections",
    }
    for ax, name in zip(axes[0], names, strict=True):
        scope = scopes[name]
        for j, metric in enumerate(("pass1", "oracle8", "returned")):
            values = [scope["arms"][arm][metric] * 100 for arm in ARM_ORDER]
            bars = ax.bar(
                np.arange(4) + (j - 1) * 0.24,
                values,
                0.23,
                color=colors[metric],
                label=labels[metric],
            )
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(
            np.arange(4),
            [ARM_LABELS[arm] + f"\nN={scope['arms'][arm]['attempts']:.2f}" for arm in ARM_ORDER],
            fontsize=9,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct proofs / problems (%)")
        ax.set_title(titles[name], fontsize=13, pad=13)
        ax.yaxis.grid(True, alpha=0.16)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(title, fontsize=16, y=0.98)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        fontsize=9,
    )
    footer = (
        "N = expected attempts. Each method uses its own generator. "
        "Fixed-four returns at 4; oracle@8 is diagnostic.\n"
        "Pass@1: original = mean of 8 exchangeable attempts; joint = first proof. "
        "Joint retries use history.\n"
        "Returned: native categorical expectation for joint policies. "
        "Point estimates; 3 training seeds (fixed-four: 1)."
    )
    if projection:
        footer = (
            "Sequent projections omit given prefixes and extra requirements; "
            "not native task scores.\n"
            "Rows share source bases. N = expected attempts. "
            "Fixed-four returns at 4.\n"
            "Oracle@8 is diagnostic; joint retries use history. "
            "Joint returns use native categorical probabilities.\n"
            "Point estimates; 3 training seeds (fixed-four: 1). "
            "Pass@1: original mean-8; joint first."
        )
    fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.subplots_adjust(
        left=0.065 if len(names) > 1 else 0.1,
        right=0.985,
        bottom=0.27 if projection else 0.24,
        top=0.78,
        wspace=0.2,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    result = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    (ART / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "scope",
        "problems",
        "method",
        "pass1",
        "oracle4",
        "oracle8",
        "returned",
        "expected_attempts",
        "training_seeds",
    ]
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scope, cohort in result["scopes"].items():
            for name in ARM_ORDER:
                arm = cohort["arms"][name]
                writer.writerow(
                    {
                        "scope": scope,
                        "problems": cohort["rows"],
                        "method": name,
                        "pass1": arm["pass1"],
                        "oracle4": arm.get("oracle4", ""),
                        "oracle8": arm["oracle8"],
                        "returned": arm["returned"],
                        "expected_attempts": arm["attempts"],
                        "training_seeds": arm["training_seeds"],
                    }
                )
    draw(
        result["scopes"],
        ["textbook72", "ordinary187"],
        "comparison",
        "Does more test-time compute improve the returned proof?",
    )
    draw(
        result["scopes"],
        ["derived225projection"],
        "derived_projections",
        "Derived sequents: diagnostic transfer",
        projection=True,
    )
    print(json.dumps({"status": "complete", "scopes": list(result["scopes"]), "out": str(OUT)}))


if __name__ == "__main__":
    main()
