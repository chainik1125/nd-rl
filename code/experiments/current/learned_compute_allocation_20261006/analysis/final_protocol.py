"""Freeze decisions first, then seal public pool metadata without opening outcomes."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
sys.path.insert(0, str(ROOT / "code/shared_code"))
from evaluation_freeze import signature_sha256  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def entries(*groups):
    result = {}
    for group in groups:
        for entry in group:
            if signature_sha256(entry["signature"]) != entry["signature_sha256"]:
                raise ValueError("Inconsistent analysis signature")
            result.setdefault(entry["signature_sha256"], entry)
    return list(result.values())


def validate_models(models):
    identities = {(m["arm"], m["rl_seed"]) for m in models}
    expected = {(arm, seed) for arm in ("cost", "no_cost") for seed in (101, 102, 103)}
    expected |= {("fixed1_control", 101), ("fixed4_control", 101), ("format_only", None)}
    if identities != expected or len(models) != 9:
        raise ValueError("Require exactly nine prespecified live models")
    for model in models:
        required = {
            "base_path",
            "base_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "inference_sha256",
            "dev_evaluation_path",
            "dev_config_sha256",
            "native_cost",
        }
        if model["arm"] in ("cost", "no_cost"):
            required |= {
                "counter_hazards_path",
                "counter_hazards_sha256",
                "budget_baselines_path",
                "budget_baselines_sha256",
            }
        if not required <= model.keys():
            raise ValueError("Incomplete live model registration")
        if model["arm"] in ("cost", "no_cost") and any(
            not model[key]
            for key in (
                "counter_hazards_path",
                "counter_hazards_sha256",
                "budget_baselines_path",
                "budget_baselines_sha256",
            )
        ):
            raise ValueError("Missing primary live baseline fit")
        if not model["base_path"] or not model["inference_sha256"]:
            raise ValueError("Missing live base or inference identity")
        if model["arm"] == "format_only":
            if model["checkpoint_path"] is not None or model["checkpoint_sha256"] is not None:
                raise ValueError("Format baseline unexpectedly includes an RL checkpoint")
        elif not model["checkpoint_path"] or not model["checkpoint_sha256"]:
            raise ValueError("Missing trained live checkpoint")
        if sha(Path(model["dev_evaluation_path"]) / "config.json") != model["dev_config_sha256"]:
            raise ValueError("Selected development configuration changed")
        for stem in ("base", "checkpoint", "counter_hazards", "budget_baselines"):
            key = stem + "_path"
            if model.get(key) and sha(model[key]) != model[stem + "_sha256"]:
                raise ValueError(f"Registered model/policy changed: {key}")


def validate_parent(parent):
    if parent["status"] != "decisions_frozen_pending_public_seal":
        raise ValueError("Not an immutable decision parent")
    if parent["builder_sha256"] != sha(__file__):
        raise ValueError("Parent builder changed after decision freeze")
    for path, digest in parent["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen entry file changed: {path}")
    for path, digest in parent["collection_sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen collection source changed: {path}")
    validate_models(parent["registries"]["main"]["registered_models"])


def create_parent(args):
    selection = Path(args.selection_inputs)
    paths = {
        "live": Path(args.live_entries),
        "analysis": Path(args.analysis_entries),
        "main": selection / "main_entries.json",
        "historical": selection / "historical_entries.json",
        "causal": selection / "causal_entries.json",
        "selection_manifest": selection / "manifest.json",
    }
    loaded = {key: read(path) for key, path in paths.items()}
    live = loaded["live"]
    if live["status"] != "ready_for_root_seal" or live["pending"]:
        raise ValueError("Live model selection is incomplete")
    validate_models(live["registered_models"])
    registries = {}
    for dataset in ("main", "historical"):
        analysis = loaded["analysis"]
        inventory_hash = analysis["inventories"][dataset + "_test"]["inventory_sha256"]
        additional = [
            entry
            for entry in analysis["registered_analyses"]
            if entry["signature"]["method"] != "staged_policy_aggregation"
            or entry["signature"]["inventory_sha256"] == inventory_hash
        ]
        core = {
            "status": "decisions_frozen_pending_public_seal",
            "selection_split": "dev",
            "dataset_role": "fresh_main" if dataset == "main" else "historical_reused_benchmark",
            "registered_models": live["registered_models"],
            "postprocessing_hashes": live["postprocessing_hashes"],
            "launch_helper_sha256": live["launch_helper_sha256"],
            "causal_launch_helper_sha256": sha(WORK / "context/run_registered_causal.py"),
            "selection_launch_helper_sha256": sha(WORK / "selection/run_frozen_test_queue.py"),
            "statistical_analysis": live["statistical_analysis"],
            "registered_public_pool_sha256": [],
            "registered_analyses": entries(
                loaded[dataset]["registered_analyses"],
                additional,
                loaded["causal"]["registered_analyses"] if dataset == "main" else [],
            ),
            **live[dataset],
        }
        if dataset == "main":
            core["registered_public_pool_sha256"].append(
                sha(ART / "context/causal_test_panel_v1/public.jsonl")
            )
        registries[dataset] = core
    source_paths = [
        WORK / "selection" / name
        for name in ("collect.py", "render_transitions.py", "seal_public_pool.py")
    ]
    source_paths.append(WORK / "analysis/run_frozen_collection.py")
    source_paths += sorted((ART / "source/nd-takehome-dan").rglob("*.py"))
    parent = {
        "status": "decisions_frozen_pending_public_seal",
        "frozen_at_utc": args.utc,
        "builder_sha256": sha(__file__),
        "selection_split": "dev",
        "inputs": {str(path.resolve()): sha(path) for path in paths.values()},
        "collection_sources": {str(path.resolve()): sha(path) for path in source_paths},
        "registry_paths": loaded["selection_manifest"]["registry_paths"],
        "registries": registries,
        "collection": {name: loaded[name]["collection"] for name in ("main", "historical")},
        "test_outcomes_generated_before_parent": False,
        "test_outcomes_inspected_before_parent": False,
        "seal_rule": (
            "Only status, public pool hashes, parent/source seal provenance, seal time, "
            "and historical reference to finalized MAIN registry may be added or changed."
        ),
    }
    for collection in parent["collection"].values():
        output = Path(collection["public_raw"]).parent
        if output.exists() or Path(collection["private_stdout_stderr_redirect"]).exists():
            raise ValueError("Independent TEST sampling already has output")
    validate_parent(parent)
    write_new(Path(args.out), parent)
    Path(args.out).with_suffix(".builder.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"parent": args.out, "sha256": sha(args.out)}))


def seal(args):
    parent = read(args.parent)
    validate_parent(parent)
    final = copy.deepcopy(parent["registries"])
    for dataset in ("main", "historical"):
        collection = parent["collection"][dataset]
        seal_path = Path(collection["public_raw"]).parent / "public_seal.json"
        metadata = read(seal_path)
        launch_path = Path(collection["public_raw"]).parent.with_suffix(".collection_launch.json")
        launch = read(launch_path)
        if (
            launch["parent_sha256"] != sha(args.parent)
            or launch["collection_plan"] != collection
            or launch["status"] != "public_seal_complete_pending_registry"
            or launch["private_outcomes_read"] is not False
            or launch["public_seal_sha256"] != sha(seal_path)
            or launch["launcher_sha256"]
            != parent["collection_sources"][
                str((WORK / "analysis/run_frozen_collection.py").resolve())
            ]
            or [row["stage"] for row in launch["processes"]]
            != ["argv", "renderer_argv", "sealer_argv"]
            or any(row["returncode"] != 0 for row in launch["processes"])
            or any(row["command"] != collection[row["stage"]] for row in launch["processes"])
        ):
            raise ValueError("Collection launch does not match the frozen parent")
        if (
            metadata["private_outcomes_read"] is not False
            or metadata["theorems"] != final[dataset]["test_theorems"]
            or metadata["attempts_per_theorem"] != 8
            or metadata["expected_data_sha256"] != collection["input_sha256"]
            or metadata["sealer_sha256"] != collection["sealer_sha256"]
        ):
            raise ValueError("Public seal differs from frozen collection plan")
        for public_key, seal_key in (
            ("public_raw", "raw_public_sha256"),
            ("public_transitions", "transition_public_sha256"),
        ):
            if sha(collection[public_key]) != metadata[seal_key]:
                raise ValueError("Public pool changed after metadata sealing")
            final[dataset]["registered_public_pool_sha256"].append(metadata[seal_key])
        final[dataset].update(
            status="frozen",
            frozen_decisions_parent_sha256=sha(args.parent),
            public_seal_path=str(seal_path),
            public_seal_sha256=sha(seal_path),
            collection_launch_sha256=sha(launch_path),
            public_sealed_at_utc=args.utc,
            private_outcomes_inspected_before_seal=False,
        )
    main_path = Path(parent["registry_paths"]["main"])
    historical_path = Path(parent["registry_paths"]["historical"])
    if main_path.exists() or historical_path.exists():
        raise ValueError("Final registries already exist")
    write_new(main_path, final["main"])
    final["historical"]["main_frozen_protocol_sha256"] = sha(main_path)
    write_new(historical_path, final["historical"])
    print(json.dumps({name: sha(path) for name, path in parent["registry_paths"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("parent")
    create.add_argument("--live-entries", required=True)
    create.add_argument("--analysis-entries", required=True)
    create.add_argument("--selection-inputs", required=True)
    create.add_argument("--out", required=True)
    create.add_argument("--utc", required=True)
    finalize = subparsers.add_parser("seal")
    finalize.add_argument("--parent", required=True)
    finalize.add_argument("--utc", required=True)
    args = parser.parse_args()
    (create_parent if args.action == "parent" else seal)(args)
