"""Bind the completed reader-facing handoff and its supporting evidence."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    "../research/proof_fragment_selection_20260910.md",
    "summary.md", "start.md", "research_log.md", "manifest.json",
    "reports/methods_and_reproduction.md", "reports/claim_evidence_v1.json",
    "reports/report_review_closeout_v1.json",
    "reports/compute_ledger_final_v1.json", "reports/reproduction_dependencies_v1.json",
    "reports/final_seed1_v1.json", "reports/final_calibration_v1.json",
    "reports/final_calibration_ranking_v1.json", "reports/retention_development_v1.json",
    "reports/final_certificate_lengths_v1.json", "reports/final_proof_examples_v1.json",
    "reports/final_models_v1.csv", "reports/final_models_by_reference_length_v1.csv",
    "reports/final_paired_effects_v1.csv", "reports/final_family_transfer_v1.csv",
    "reports/final_tables_manifest_v1.json", "reports/training_closed_v1.json",
    "configs/training_closure_plan_v1.json", "configs/recipe_final_seed1_v1.json",
    "configs/release_final_seed1_v1.json", "configs/rl_calibration_design_v1.json",
    "theory/theory_memo.md", "theory/fisher_transfer_bound.md", "theory/coverage_gradients.md",
    "theory/theory_distillation_for_report.md", "theory/literature_evidence_map.md",
    "theory/final_results_interpretation.md", "theory/final_prefix_exposure_review.md",
    "theory/final_prefix_exposure_performance_v1.json", "theory/final_calibration_independent.json",
    "theory/final_primary_agreement_receipt.json", "theory/final_report_claim_review_v1.md",
    "theory/supplementary_mechanisms_review.md", "theory/supplementary_mechanisms_analysis.json",
    "code/README.md", "code/tests/final_validation_summary.md",
    "code/tests/final_provenance_independent_v1.json", "code/tests/final_gpu_observation_v1.json",
    "code/tests/final_seed1_independent_v2/receipt.json",
    "code/tests/retention_development_independent_receipt_v1.json",
    "code/tests/completed_diagnostics_independent_v1.json",
    "code/tests/final_report_graph_review_v1.md", "code/tests/final_report_graph_review_data_v1.json",
    "code/tests/main_runtime_selected_full_endpoint_complete_v1.json",
    "code/tests/main_runtime_selected_reset_endpoint.json",
    "code/tests/main_runtime_uniform_full_endpoint.json",
    "code/tests/main_runtime_uniform_reset_endpoint.json",
    "code/tests/retention_seed1_endpoint_independent_v1.json",
    "figures/final_calibration_transfer_manifest_v1.json",
    "figures/final_factorial_plot_manifest_v1.json",
    "figures/final_main_reference_length_manifest_v1.json",
    "tools/plot_final_calibration.py", "tools/plot_factorial_results.py",
    "tools/plot_final_length_results.py", "tools/export_final_tables.py",
    "tools/build_delivery_inventory.py",
]
for prefix in ["final_calibration_transfer", "final_main_learning_curves",
               "final_main_endpoint_contrasts", "final_controls", "final_main_reference_length"]:
    ARTIFACTS.extend(f"figures/{prefix}_v1.{ext}" for ext in ["png", "svg", "pdf"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/delivery_inventory_v1.json")
    parser.add_argument("--check-paths-only", action="store_true")
    args = parser.parse_args()
    missing = [p for p in ARTIFACTS if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.check_paths_only:
        print(json.dumps({"curated_artifacts_present": len(ARTIFACTS)}))
        return
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Close the research/report manifest before producing the final handoff identity")
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError("Preserve the previous delivery inventory and choose a new version")
    artifacts = {}
    for relative in ARTIFACTS:
        path = ROOT / relative
        artifacts[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
    inventory = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "status": "completed_handoff", "sprint": ROOT.name,
                 "scope": "Curated index of reader-facing artifacts and load-bearing evidence, not a complete archive of raw rollouts or weights. Frozen job/result manifests and the provenance receipt bind those larger artifacts. No listed artifact is modified by this inventory builder.",
                 "entry_point": "summary.md", "reproduction": "reports/methods_and_reproduction.md",
                 "resource_status": "All nine acquired sprint jobs completed and released; no new job was launched after the final evaluation-only job ended at 02:31:27 UTC.",
                 "artifacts": artifacts}
    target.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"inventory": str(target), "artifacts": len(artifacts),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__ == "__main__":
    main()
