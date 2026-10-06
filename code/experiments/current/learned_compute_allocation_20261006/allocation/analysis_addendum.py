"""Bind a reviewed numeric-domain analysis successor to unchanged generation registries."""

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_statistics(addendum_file, benchmark, registry_file, successor_file):
    addendum = json.loads(Path(addendum_file).read_text())
    if (
        addendum.get("status") != "frozen_analysis_only"
        or addendum.get("validation_helper_sha256") != sha(__file__)
        or addendum.get("numeric_change")
        != {
            "field": "correct_probability",
            "original_domain_tolerance": 1e-8,
            "successor_domain_tolerance": 1e-6,
            "clipping": False,
            "renormalization": False,
            "estimates_changed": False,
        }
    ):
        raise ValueError("Reviewed analysis-only addendum is required")
    registries = addendum["original_generation_registries"]
    for item in registries.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Original frozen generation registry changed")
    item = registries[benchmark]
    if Path(registry_file).resolve() != Path(item["path"]).resolve():
        raise ValueError("Analysis must use the exact original generation registry")
    entries = addendum["original_allocation_entries"]
    if sha(entries["path"]) != entries["sha256"]:
        raise ValueError("Original pre-outcome allocation entries changed")
    original_entries = json.loads(Path(entries["path"]).read_text())
    for manifest in original_entries["planned_run_manifests"].values():
        if sha(manifest["path"]) != manifest["sha256"]:
            raise ValueError("Original planned-run manifest changed")
    analyses = addendum["analyses"]
    for name, analysis in analyses.items():
        freeze = json.loads(Path(registries[name]["path"]).read_text())
        if analysis["original_statistical_analysis"] != freeze["statistical_analysis"]:
            raise ValueError("Original statistical registration differs")
        if sha(analysis["original_script_path"]) != freeze["statistical_analysis"]["script_sha256"]:
            raise ValueError("Original frozen statistical source changed")
        if sha(analysis["successor_script_path"]) != analysis["successor_script_sha256"]:
            raise ValueError("Reviewed successor statistical source changed")
        expected = dict(freeze["statistical_analysis"])
        expected["script_sha256"] = analysis["successor_script_sha256"]
        if name == "historical":
            expected["aggregate_live_dependency_sha256"] = analyses["main"][
                "successor_script_sha256"
            ]
        if analysis["statistical_analysis"] != expected:
            raise ValueError("Successor changes statistics beyond source identity")
    analysis = analyses[benchmark]
    if Path(successor_file).resolve() != Path(analysis["successor_script_path"]).resolve():
        raise ValueError("Unregistered statistical successor")
    return analysis["statistical_analysis"]
