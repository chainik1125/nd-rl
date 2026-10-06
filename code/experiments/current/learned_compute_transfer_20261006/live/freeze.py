"""Register unchanged selected checkpoints and new transfer data before generation."""

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OLD = ROOT / "code/experiments/current/learned_compute_allocation_20261006/allocation"
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(OLD))
    from evaluation_protocol import BEHAVIOR_FILES, fingerprint, signature, validate_test_freeze

    parent = (
        ROOT
        / "artifacts/learned_compute_allocation_20261006/final_evaluation/historical_protocol.json"
    )
    parent_expected = "d228e09e16fa86aa69160e84464eca5064fcecde3c057393e7cc0eecba1b2d7d"
    if sha(parent) != parent_expected:
        raise ValueError("Original frozen model choices changed")
    data = ART / "data/combined412/public.jsonl"
    expected_data = "969d72c5d861f91adcbf238577daf53ee129440010d4adcd2dea5cc99acf8e47"
    if sha(data) != expected_data:
        raise ValueError("Transfer dataset differs from declared public corpus")
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    if (
        len(rows) != 412
        or len({r["theorem_id"] for r in rows}) != 412
        or sum(r["component"] == "ordinary187" for r in rows) != 187
        or sum(r["component"] == "derived225" for r in rows) != 225
    ):
        raise ValueError("Unexpected transfer cohorts")
    original = json.loads(parent.read_text())
    entries = []
    configs = []
    parents = {
        str(parent): sha(parent),
        str(ART / "live/model_inventory.json"): sha(ART / "live/model_inventory.json"),
    }
    implementations = {str(OLD / name): sha(OLD / name) for name in BEHAVIOR_FILES}
    for name in ("aggregate_live_fp32_v2.py", "analysis_addendum.py"):
        implementations[str(OLD / name)] = sha(OLD / name)
    implementations[str(ROOT / "code/shared_code/learned_allocation.py")] = sha(
        ROOT / "code/shared_code/learned_allocation.py"
    )
    for original_entry in original["registered_models"]:
        entry = dict(original_entry)
        entry["model_key"] = entry["arm"] + (
            f"_s{entry['rl_seed']}" if entry["rl_seed"] is not None else ""
        )
        config_file = Path(entry["dev_evaluation_path"]) / "config.json"
        if sha(config_file) != entry["dev_config_sha256"]:
            raise ValueError("Original selected inference config changed")
        parents[str(config_file)] = sha(config_file)
        config = json.loads(config_file.read_text())
        if fingerprint(signature(config)) != entry["inference_sha256"]:
            raise ValueError("Selected model/inference signature changed")
        for name, expected in config["source_hashes"].items():
            path = ROOT / config["source"] / name
            if sha(path) != expected:
                raise ValueError("Original generator source changed")
            implementations[str(path)] = expected
        for key in ("base", "checkpoint"):
            if (
                entry.get(key + "_path")
                and sha(ROOT / entry[key + "_path"]) != entry[key + "_sha256"]
            ):
                raise ValueError("Selected checkpoint changed")
        entries.append(entry)
        configs.append(config)
    registry = {
        "status": "frozen",
        "selection_split": "dev",
        "selection_scope": "Inherited MAIN development choices; no transfer fitting or tuning",
        "sealed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "release_commit": "5d0084add453212e6bea8055d4d28cec790f268c",
        "data_path": str(data),
        "test_data_sha256": sha(data),
        "test_theorems": 412,
        "test_seed": 95006,
        "test_repeats": 2,
        "registered_models": entries,
        "results_root": str(ART / "live/results"),
        "launcher_sha256": sha(HERE / "launch.py"),
        "aggregation_sha256": sha(HERE / "aggregate.py"),
        "implementation_hashes": implementations,
        "parent_hashes": parents,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 192006,
        "confidence": 0.95,
        "uncertainty": (
            "Average two paths per row, then three fixed training seeds within primary arm; "
            "row-weighted cohort point mean. Bootstrap whole base_cluster groups within each "
            "scope and recompute row-weighted denominator per draw. Conditional on frozen "
            "models and saved paths, not uncertainty over new training seeds. "
        ),
        "scope": (
            "ordinary187 primary transfer benchmark; derived225 sequent-only projection "
            "diagnostic and individual source/nuance components. No pooled412 native-task "
            "result. Oracle4/8 is validity availability in capped contextual path prefixes, "
            "not an iid unbiased pass@k estimator or deployable oracle policy. Fixed4 control "
            "trained4/eval4 primary; eval8 secondary. Native zero-cost policy may stop and "
            "has no explicit coherence reward. "
        ),
        "return_policies": {
            "cost": "learned",
            "no_cost": "learned",
            "fixed4_control": "fixed4",
            "fixed1_control": "fixed1",
            "format_only": "fixed1",
        },
    }
    for config in configs:
        validate_test_freeze(registry, signature(config), sha(data), 95006, 2, 412)
    path = ART / "live/protocol.json"
    with path.open("x") as stream:
        json.dump(registry, stream, indent=2)
        stream.write("\n")
    print(str(path), sha(path))


if __name__ == "__main__":
    main()
