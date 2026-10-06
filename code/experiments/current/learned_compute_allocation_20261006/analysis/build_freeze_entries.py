"""Describe frozen feature and staged analyses without opening TEST outcomes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import aggregate_staged
import analyze_candidates
import critic_thresholds
import threshold_baselines

from code.shared_code.evaluation_freeze import signature_sha256

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_new(path, value):
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to replace different freeze input: {path}")
    else:
        path.write_text(text)


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    feature_model = ART / "analysis/features1600_main_dev/feature_model.json"
    source_inventory = ART / "analysis/staged_dev_inventory.json"
    dev = json.loads(source_inventory.read_text())
    signatures = [analyze_candidates.analysis_signature(feature_model)]
    feature_policy = None
    for run in dev["runs"]:
        policy = json.loads(Path(run["policy"]).read_text())
        if analyze_candidates.sha256(run["policy"]) != run["policy_sha256"]:
            raise ValueError("A development policy changed")
        if run["group"] == "feature":
            feature_policy = run["policy"]
            signatures.append(threshold_baselines.analysis_signature(feature_model, feature_policy))
        else:
            signatures.append(
                critic_thresholds.analysis_signature(policy["critic_signature"], run["policy"])
            )
    if feature_policy is None:
        raise ValueError("Missing development feature policy")
    inventories = {}
    for dataset in ("main_test", "historical_test"):
        inventory = {"runs": []}
        for source in dev["runs"]:
            run = dict(source)
            tag = "feature" if run["group"] == "feature" else f"bce_s{run['training_seed']}"
            run["evaluation"] = relative(ART / f"analysis/{dataset}_{tag}_staged_v1")
            inventory["runs"].append(run)
        path = out / f"{dataset}_staged_inventory.json"
        write_new(path, inventory)
        args = SimpleNamespace(inventory=path, bootstrap=5000, seed=61006)
        signatures.append(aggregate_staged.signature(args))
        inventories[dataset] = {
            "inventory": relative(path),
            "inventory_sha256": analyze_candidates.sha256(path),
            "public": relative(ART / f"pools/{dataset}/public.jsonl"),
            "feature_evaluation": relative(ART / f"analysis/{dataset}_features_v1"),
            "staged_aggregate": relative(ART / f"analysis/{dataset}_staged_aggregate_v1"),
            "bootstrap_samples": 5000,
            "bootstrap_seed": 61006,
        }
    result = {
        "status": "entries_ready_not_test_authorization",
        "selection_split": "dev",
        "feature_model": relative(feature_model),
        "feature_model_sha256": analyze_candidates.sha256(feature_model),
        "feature_policy": feature_policy,
        "source_dev_inventory_sha256": analyze_candidates.sha256(source_inventory),
        "registered_analyses": [
            {"signature": signature, "signature_sha256": signature_sha256(signature)}
            for signature in signatures
        ],
        "inventories": inventories,
        "builder_sha256": analyze_candidates.sha256(__file__),
        "historical_scope": "descriptive transfer on reused textbook72; no historical tuning",
    }
    write_new(out / "analysis_freeze_entries.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build(Path(args.out))
    print(json.dumps({"entries": len(result["registered_analyses"]), "out": args.out}))
