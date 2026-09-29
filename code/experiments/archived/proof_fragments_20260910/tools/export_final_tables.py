"""Export all saved final results without changing estimands or selecting rows."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/final_seed1_v1.json"
CALIBRATION = ROOT / "reports/final_calibration_ranking_v1.json"
CERTIFICATES = ROOT / "reports/final_certificate_lengths_v1.json"
METRICS = {
    "coverage_at_k": ("coverage_at8_percent", 100),
    "attempt_success": ("attempt_success_percent", 100),
    "whole_proof_bits": ("reference_bits_per_proof", 1),
    "per_proof_bits_per_token": ("mean_per_proof_bits_per_token", 1),
    "mixed_group_fraction": ("mixed_group_percent", 100),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    data = json.loads(ANALYSIS.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    certificates = json.loads(CERTIFICATES.read_text())["models"]
    outputs = {}

    def save(name, rows):
        target = ROOT / "reports" / name
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs[str(target)] = {"rows": len(rows), "sha256": sha(target)}

    model_rows = []
    bin_rows = []
    for name, model in data["models"].items():
        for stratum in ["balanced", "2-6", "7-9", "10-12"]:
            n = model["coverage_at_k"][stratum]["n_sources"]
            row = {"model": name, "reference_length_bin": stratum, "targets": n,
                   "attempts_per_target": 8,
                   "covered_targets": round(model["coverage_at_k"][stratum]["mean"] * n),
                   "successful_attempts": round(model["attempt_success"][stratum]["mean"] * n * 8),
                   "total_attempts": n * 8}
            for metric, (label, scale) in METRICS.items():
                values = model[metric][stratum]
                row[label] = values["mean"] * scale
                row[label + "_context95_low"] = values["context_bootstrap95"][0] * scale
                row[label + "_context95_high"] = values["context_bootstrap95"][1] * scale
            if stratum == "balanced":
                assert row["covered_targets"] == data["receipts"][name]["covered_sources"]
                assert row["successful_attempts"] == data["receipts"][name]["successful_attempts"]
                certificate = certificates[name]
                assert certificate["covered"] == row["covered_targets"]
                for key in ["maximum_solved_reference_lines", "maximum_valid_written_lines",
                            "maximum_best_observed_pruned_lines", "covered_long_reference_targets",
                            "long_reference_targets_with_observed_cap6_certificate"]:
                    row[key] = certificate[key]
                model_rows.append(row)
            else:
                certificate = certificates[name]["by_reference_length_bin"][stratum]
                assert certificate["covered"] == row["covered_targets"]
                for key in ["covered_with_observed_certificate_at_most_six",
                            "covered_best_observed_certificate_above_six"]:
                    row[key] = certificate[key]
                bin_rows.append(row)
    save("final_models_v1.csv", model_rows)
    save("final_models_by_reference_length_v1.csv", bin_rows)

    contrasts = []
    for kind, objects in [("pairwise", data["comparisons"]), ("factorial", data["contrasts"])]:
        for name, obj in objects.items():
            values = obj if kind == "pairwise" else obj["metrics"]
            for metric, (label, scale) in METRICS.items():
                for stratum in ["balanced", "2-6", "7-9", "10-12"]:
                    value = values[metric][stratum]
                    contrasts.append({"contrast": name, "kind": kind, "metric": metric,
                                      "reference_length_bin": stratum,
                                      "unit": "percentage_points" if scale == 100 else label,
                                      "difference": value["mean"] * scale,
                                      "context95_low": value["context_bootstrap95"][0] * scale,
                                      "context95_high": value["context_bootstrap95"][1] * scale})
    save("final_paired_effects_v1.csv", contrasts)

    family_rows = []
    for name, row in calibration["families"].items():
        family_rows.append({"family": name, "frozen_selected": name in calibration["frozen_top3"],
                            "selection_utility_bits": row["selection_utility_bits"],
                            "final_utility_bits": row["final_utility_bits"],
                            "final_context95_low": row["final_context_bootstrap95"][0],
                            "final_context95_high": row["final_context_bootstrap95"][1]})
    save("final_family_transfer_v1.csv", family_rows)
    receipt = {"input_sha256": {str(p): sha(p) for p in [ANALYSIS, CALIBRATION, CERTIFICATES, Path(__file__).resolve()]},
               "outputs": outputs,
               "scope": "Complete export of saved results. Final balanced estimates use three equally sized bins, so aggregate counts reconcile exactly. All intervals condition on fixed trained models and decoding seed; no multiplicity correction. All proof-line counts include premises and assumptions. Long reference means above six lines. Best observed pruned length is an upper bound on minimum proof length, not a lower bound or difficulty certificate. No new statistics, filtering, training or checkpoint selection."}
    path = ROOT / "reports/final_tables_manifest_v1.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({Path(p).name: v["rows"] for p, v in outputs.items()}))

if __name__ == "__main__":
    main()
