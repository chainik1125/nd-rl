"""Summarize fixed-action precision probes; no model or held-out inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("head_fp32", "reduction_fp32", "sdpa_math", "head_attention_fp32",
         "lora_fp32", "all_fp32", "all_fp32_efficient")
LABELS = ("FP32 output head", "Reduction flags", "Math attention", "FP32 head + attention",
          "FP32 LoRA", "Full FP32 / math", "Full FP32 / efficient")


def collect():
    receipts, variants = {}, {}
    for name in ("precision_probe_v3", "precision_probe_full_v3", "all_fp32_efficient_probe_v3"):
        path = ROOT / "runs" / name / "summary.json"
        raw = path.read_bytes()
        receipts[str(path)] = hashlib.sha256(raw).hexdigest()
        summary = json.loads(raw)
        if (summary["status"] != "complete" or summary["optimizer_steps"] != 0
                or summary["adapter_before"] != summary["adapter_after"]
                or summary["frozen_tolerance_changed"]):
            raise ValueError("Incomplete, mutating, or differently guarded diagnostic")
        values = summary.get("results", {"all_fp32_efficient": summary})
        for variant, result in values.items():
            metric = result["same_candidate_cached_vs_teacher"]
            gradient = result["gradient_audit"]
            if (metric["tokens"] != 1805 or metric["frozen_tolerance"] != .1
                    or not gradient["gradient_enabled"] or not gradient["backward_exercised"]
                    or not gradient["gradient_checkpointing"] or gradient["optimizer_steps"] != 0
                    or gradient["tensors_with_gradients"] != 896
                    or not math.isfinite(gradient["gradient_norm"]) or gradient["gradient_norm"] <= 0):
                raise ValueError("Missing fixed-action/gradient evidence")
            passed = metric["max_abs_error"] <= .1
            if passed != result["passes_existing_guard_on_recorded_actions"]:
                raise ValueError("Diagnostic pass declaration does not match its metric")
            variants[variant] = {k: metric[k] for k in ("tokens", "max_abs_error", "mean_abs_error",
                "p99_abs_error", "tokens_above_frozen_tolerance")}
            variants[variant].update(passed=passed, seconds=result["seconds"],
                original_behavior_drift_max=result["candidate_cached_vs_original_behavior"]["max_abs_error"],
                gradient_norm=gradient["gradient_norm"], source=str(path))
    if set(variants) != set(ORDER):
        raise ValueError("Unexpected precision candidate set")
    for name, expected in receipts.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Diagnostic changed during report construction")
    return {"kind": "training_action_numerical_diagnostics", "tokens_per_candidate": 1805,
        "optimizer_steps": 0, "threshold": .1, "sha256": receipts,
        "variants": variants, "order": list(ORDER),
        "interpretation": "Each candidate's own cached probabilities versus its own differentiable teacher scorer on the same recorded training actions. This is internal numerical consistency, not proof accuracy or a universal bound. Original-BF16 policy drift is separate and never replaces historical behavior probabilities. Fresh rollouts remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "reports") or args.out.exists():
        parser.error("Use a fresh sprint report directory")
    report = collect()
    args.out.mkdir()
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    lines = ["# Fixed-action numerical consistency", "", report["interpretation"], "",
        "All seven candidates preserve the exact adapter, use no optimizer, and exercise finite nonzero checkpointed gradients. Each evaluates 1,805 tokens. The frozen maximum absolute log-probability threshold is 0.1.", "",
        "| Candidate | Max absolute error | Tokens above 0.1 | Guard |", "|---|---:|---:|---|"]
    for name, label in zip(ORDER, LABELS, strict=True):
        result = report["variants"][name]
        lines.append(f"| {label} | {result['max_abs_error']:.9g} | {result['tokens_above_frozen_tolerance']} | {'pass' if result['passed'] else 'fail'} |")
    lines += ["", "The math-attention-only value of approximately 0.106 is a failure, not rounded into acceptance. These observations motivated the uniform v4 FP32 policy; they do not imply that BF16 RL in general is invalid.", ""]
    with (args.out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [report["variants"][name]["max_abs_error"] for name in ORDER]
    fig, ax = plt.subplots(figsize=(8.4, 4.1), layout="constrained")
    ax.scatter(values, range(len(ORDER)), c=["#167d8d" if value <= .1 else "#bb6040" for value in values], s=50)
    ax.axvline(.1, color="#555555", linestyle="--", linewidth=1, label="Frozen guard: 0.1")
    ax.set_xscale("log")
    ax.set_xlim(2e-5, 1)
    ax.set_yticks(range(len(ORDER)), LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Maximum absolute log-probability discrepancy (log scale)")
    ax.set_title("Same 1,805 training tokens; each candidate's own behavior/scorer")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower right", frameon=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out / f"numerical_consistency.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps({"out": str(args.out), "candidates": len(ORDER), "held_out_inputs_opened": False}))


if __name__ == "__main__":
    main()
