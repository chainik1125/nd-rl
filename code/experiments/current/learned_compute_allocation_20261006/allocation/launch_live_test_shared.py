"""Registered live commands with explicitly amended shared-GPU admission."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from evaluation_protocol import (
    fingerprint,
    signature,
    validate_saved_implementation,
    validate_test_freeze,
)

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = WORK / "allocation"
PYTHON = "/data/users/dmitry/venvs/proof-state/bin/python"
POSTPROCESSORS = (
    "fit_counter_hazards.py",
    "reanalyze_counter.py",
    "fit_budget_baselines.py",
    "reanalyze_budget_baselines.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_operational_addendum(args):
    path = Path(args.operational_addendum)
    amendment = json.loads(path.read_text())
    original = ALLOC / "launch_live_test.py"
    identity = amendment["launchers"]["live"]
    if (
        amendment["status"] != "approved_operational_addendum"
        or amendment["allowed_gpus"] != [1, 2, 3, 4]
        or amendment["minimum_free_memory_mib"] != 65536
        or amendment["original_registry_sha256"]["main"]
        != sha(ART / "final_evaluation/main_protocol.json")
        or amendment["original_registry_sha256"]["historical"]
        != sha(ART / "final_evaluation/historical_protocol.json")
        or identity["original_path"] != str(original)
        or identity["original_sha256"] != sha(original)
        or identity["successor_path"] != str(Path(__file__).resolve())
        or identity["successor_sha256"] != sha(__file__)
    ):
        raise ValueError("Unbound or unapproved operational successor")
    if amendment["original_registry_sha256"][args.benchmark] != sha(args.registry):
        raise ValueError("Live request names a different generation registry")
    return amendment


def admit_shared_gpu(gpu, minimum_free_mib):
    if gpu not in (1, 2, 3, 4) or minimum_free_mib != 65536:
        raise ValueError("GPU admission outside the declared operational scope")
    snapshot = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    device, total, free, utilization = [value.strip() for value in snapshot.split(",")]
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    occupants = []
    for row in processes.splitlines():
        gpu_uuid, pid, memory = [value.strip() for value in row.split(",")]
        if gpu_uuid != device:
            continue
        try:
            uid = Path(f"/proc/{int(pid)}").stat().st_uid
        except FileNotFoundError:
            uid = None
        occupants.append({"pid": int(pid), "uid": uid, "used_memory_mib": memory})
        if uid == os.getuid():
            raise ValueError("Another same-user model occupies the assigned GPU")
    if int(free) < minimum_free_mib:
        raise ValueError("Insufficient free memory for the shared-GPU operational admission")
    return {
        "gpu": gpu,
        "uuid": device,
        "total_memory_mib": int(total),
        "free_memory_mib": int(free),
        "utilization_percent": int(utilization),
        "occupants": occupants,
        "minimum_free_memory_mib": minimum_free_mib,
        "wall_time_caveat": "Shared device; model timings are not isolated latency measurements.",
    }


def preflight(args):
    amendment = validate_operational_addendum(args)
    registry_bytes = Path(args.registry).read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    freeze = json.loads(registry_bytes)
    if freeze.get("status") != "frozen" or freeze.get("selection_split") != "dev":
        raise ValueError("Root must seal the registry before any TEST launch")
    if freeze.get("launch_helper_sha256") != sha(ALLOC / "launch_live_test.py"):
        raise ValueError("Launch helper source is not registered")
    entries = freeze.get("registered_models", [])
    if len(entries) != 9 or any(entry.get("pending") for entry in entries):
        raise ValueError("All nine final models must be resolved before TEST")
    matches = [
        entry
        for entry in entries
        if entry["arm"] == args.arm and entry["rl_seed"] == args.training_seed
    ]
    if len(matches) != 1:
        raise ValueError("Requested model/seed is not uniquely registered")
    entry = matches[0]
    config_file = Path(entry["dev_evaluation_path"]) / "config.json"
    if sha(config_file) != entry["dev_config_sha256"]:
        raise ValueError("Selected DEV configuration changed")
    config = json.loads(config_file.read_text())
    validate_saved_implementation(config)
    if config["evaluation_split"] != "dev":
        raise ValueError("Model initialization was not frozen from DEV")
    selected_signature = signature(config)
    if fingerprint(selected_signature) != entry["inference_sha256"]:
        raise ValueError("Selected inference signature differs from registry")
    if any(entry[key] != config[key] for key in ("base_sha256", "checkpoint_sha256")):
        raise ValueError("Registry model hashes disagree with selected inference")
    expected_settings = {
        "context": 16384,
        "attempt_bound": 1024,
        "state_reserve": 1024,
        "max_attempts": 8,
        "max_action": 256,
        "max_steps": 48,
        "temperature": 0.8,
        "sampling_precision": "fp32",
        "replay_precision": "fp32",
        "cost": 0.03 if args.arm == "cost" else 0.0,
    }
    if any(config.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Selected inference differs from the bounded MAIN method")
    base = ROOT / Path(entry["base_path"])
    checkpoint = ROOT / Path(entry["checkpoint_path"]) if entry["checkpoint_path"] else None
    if (
        sha(base) != entry["base_sha256"]
        or (sha(checkpoint) if checkpoint else None) != entry["checkpoint_sha256"]
    ):
        raise ValueError("Selected model bytes changed")
    for name, expected in config["source_hashes"].items():
        if sha(ROOT / Path(config["source"]) / name) != expected:
            raise ValueError(f"Upstream source changed: {name}")
    seed, theorems = (62006, 400) if args.benchmark == "main" else (92006, 72)
    validate_test_freeze(freeze, selected_signature, sha(args.data), seed, 2, theorems)
    if sum(bool(line.strip()) for line in Path(args.data).read_bytes().splitlines()) != theorems:
        raise ValueError("Registered theorem count disagrees with dataset")
    statistics = freeze.get("statistical_analysis", {})
    statistical_script = (
        "aggregate_live.py" if args.benchmark == "main" else "aggregate_historical.py"
    )
    expected_statistics = dict(
        script_sha256=sha(ALLOC / statistical_script),
        bootstrap_samples=10000,
        bootstrap_seed=72006 if args.benchmark == "main" else 102006,
        confidence=0.95,
        family_data_sha256=sha(args.family_data or args.data),
    )
    if args.benchmark == "historical":
        expected_statistics["aggregate_live_dependency_sha256"] = sha(ALLOC / "aggregate_live.py")
        if not args.main_registry:
            raise ValueError("Historical launch requires the original MAIN registry")
        from aggregate_historical import validate_registration

        validate_registration(
            freeze, args.main_registry, args.family_data or args.data, args.exposure_audit
        )
    if any(statistics.get(key) != value for key, value in expected_statistics.items()):
        raise ValueError("Statistical source/settings/metadata are not registered")
    if any(
        freeze.get("postprocessing_hashes", {}).get(name) != sha(ALLOC / name)
        for name in POSTPROCESSORS
    ):
        raise ValueError("Postprocessing source changed since registration")
    primary = args.arm in ("cost", "no_cost")
    fits = []
    if primary:
        for key, argument, script in [
            ("counter_hazards", "--hazards", "reanalyze_counter.py"),
            ("budget_baselines", "--fit", "reanalyze_budget_baselines.py"),
        ]:
            path = Path(entry[key + "_path"])
            if sha(path) != entry[key + "_sha256"]:
                raise ValueError("Registered DEV fitting file changed")
            fit = json.loads(path.read_text())
            if fit["fit_split"] != "dev" or fit["inference_sha256"] != entry["inference_sha256"]:
                raise ValueError("Fitting file does not belong to selected DEV model")
            fits.append((path, argument, script))
    output = Path(args.out).resolve()
    if not output.is_relative_to(ART.resolve()):
        raise ValueError("TEST output must remain inside sprint artifacts")
    destinations = [
        output,
        output.with_suffix(".launch.json"),
        output.with_suffix(".launch.py"),
        output.with_suffix(".launch.log"),
        output.with_suffix(".evaluator.py"),
    ]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("TEST output or launch artifacts already exist")
    runtime = subprocess.check_output(
        [args.python, "-c", "import torch; print(str(torch.__version__))"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", ND_OFFLINE="1"),
        text=True,
    ).strip()
    if runtime != config["torch_version"]:
        raise ValueError("Torch version differs from selected DEV inference")
    if sha(args.registry) != registry_hash:
        raise ValueError("Immutable registry changed during preflight")
    admit_shared_gpu(args.gpu, amendment["minimum_free_memory_mib"])
    command = [
        args.python,
        str(ALLOC / "evaluate_live_path.py"),
        "--source",
        str(ROOT / config["source"]),
        "--base",
        str(base),
        "--data",
        str(Path(args.data).resolve()),
        "--out",
        str(output),
        "--frozen-protocol",
        str(Path(args.registry).resolve()),
        "--seed",
        str(seed),
        "--repeats",
        "2",
    ]
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    for key in (
        "context",
        "attempt_bound",
        "state_reserve",
        "max_attempts",
        "max_action",
        "max_steps",
        "temperature",
        "cost",
    ):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    return output, command, fits, registry_hash


def main():
    whole_started = time.monotonic()
    started_utc = datetime.datetime.now(datetime.UTC).isoformat()
    parser = argparse.ArgumentParser()
    for key in ("registry", "data", "arm", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--operational-addendum", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--benchmark", choices=("main", "historical"), required=True)
    parser.add_argument("--family-data")
    parser.add_argument("--main-registry")
    parser.add_argument(
        "--exposure-audit",
        default=str(ART / "data/textbook72_metadata_audit/exposure.json"),
    )
    parser.add_argument("--python", default=PYTHON)
    args = parser.parse_args()
    amendment = validate_operational_addendum(args)
    amendment_hash = sha(args.operational_addendum)
    output, command, fits, registry_hash = preflight(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_suffix(".launch.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with output.with_suffix(".evaluator.py").open("xb") as stream:
        stream.write((ALLOC / "evaluate_live_path.py").read_bytes())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), ND_OFFLINE="1")
    record = dict(
        arm=args.arm,
        rl_seed=args.training_seed,
        gpu=args.gpu,
        benchmark=args.benchmark,
        registry_sha256=registry_hash,
        launch_helper_sha256=sha(__file__),
        original_launch_helper_sha256=sha(ALLOC / "launch_live_test.py"),
        operational_addendum_path=str(Path(args.operational_addendum).resolve()),
        operational_addendum_sha256=amendment_hash,
        gpu_admissions=[],
        command=command,
        started_utc=started_utc,
        pid=os.getpid(),
        phases=[],
        preflight_wall_seconds=time.monotonic() - whole_started,
        status="running",
    )
    ledger = output.with_suffix(".launch.json")
    with ledger.open("x") as stream:
        json.dump(record, stream, indent=2)
    try:
        jobs = [("generation_and_checking", command)]
        for path, argument, script in fits:
            name = (
                "counter_integrated.json" if argument == "--hazards" else "budget_integrated.json"
            )
            jobs.append(
                (
                    script,
                    [
                        args.python,
                        str(ALLOC / script),
                        "--paths",
                        str(output / "public_paths.jsonl"),
                        "--labels",
                        str(output / "private_labels.jsonl"),
                        argument,
                        str(path),
                        "--frozen-protocol",
                        str(Path(args.registry).resolve()),
                        "--out",
                        str(output / name),
                    ],
                )
            )
        with output.with_suffix(".launch.log").open("x") as log:
            for name, job in jobs:
                if sha(args.registry) != registry_hash:
                    raise ValueError("Immutable registry changed during launch")
                if sha(args.operational_addendum) != amendment_hash:
                    raise ValueError("Operational addendum changed during launch")
                validate_operational_addendum(args)
                if name == "generation_and_checking":
                    record["gpu_admissions"].append(
                        admit_shared_gpu(args.gpu, amendment["minimum_free_memory_mib"])
                    )
                    ledger.write_text(json.dumps(record, indent=2) + "\n")
                started = time.monotonic()
                result = subprocess.run(
                    job, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
                record["phases"].append(
                    dict(
                        name=name,
                        command=job,
                        returncode=result.returncode,
                        monotonic_process_wall_seconds=time.monotonic() - started,
                    )
                )
                ledger.write_text(json.dumps(record, indent=2) + "\n")
                if result.returncode:
                    raise RuntimeError(f"{name} failed with code{result.returncode}")
        record["status"] = "complete"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        raise
    finally:
        record["whole_process_wall_seconds"] = time.monotonic() - whole_started
        ledger.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
