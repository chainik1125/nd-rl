"""Original registered selector commands with amended shared-GPU admission."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
sys.path.insert(0, str(ROOT / "code/shared_code"))
from evaluation_freeze import file_sha256, validate_frozen_evaluation  # noqa: E402


def pairs(argv):
    result = {}
    for i in range(2, len(argv), 2):
        if i + 1 >= len(argv) or not argv[i].startswith("--"):
            raise ValueError("Expected explicit flag/value inference arguments")
        if argv[i] in result:
            raise ValueError("Duplicate inference option")
        result[argv[i]] = argv[i + 1]
    return result


COMMON_ADDENDUM = (
    ROOT / "artifacts/learned_compute_allocation_20261006/operational_amendment_v1/addendum.json"
)
COMMON_ADDENDUM_SHA256 = "c0176301d56965d4b355c071f27c5df032ba546e9a8bfd2891da7dbaf5e89431"
ORIGINAL = FOLDER / "run_frozen_test_queue.py"


def validate_operational_addendum(args):
    amendment = json.loads(Path(args.operational_addendum).read_text())
    common = amendment["original_operational_addendum"]
    identity = amendment["launchers"]["selection"]
    registries = {
        key: ROOT / "artifacts/learned_compute_allocation_20261006/final_evaluation" / filename
        for key, filename in (
            ("main", "main_protocol.json"), ("historical", "historical_protocol.json")
        )
    }
    if (
        amendment["status"] != "approved_operational_addendum"
        or amendment["allowed_gpus"] != [1, 2, 3, 4]
        or amendment["minimum_free_memory_mib"] != 65536
        or common["path"] != str(COMMON_ADDENDUM)
        or common["sha256"] != COMMON_ADDENDUM_SHA256
        or file_sha256(COMMON_ADDENDUM) != COMMON_ADDENDUM_SHA256
        or identity["original_path"] != str(ORIGINAL)
        or identity["original_sha256"] != file_sha256(ORIGINAL)
        or identity["successor_path"] != str(Path(__file__).resolve())
        or identity["successor_sha256"] != file_sha256(__file__)
    ):
        raise ValueError("Unbound or unapproved selection operational successor")
    for key, path in registries.items():
        if (
            amendment["original_registry_path"][key] != str(path)
            or amendment["original_registry_sha256"][key] != file_sha256(path)
        ):
            raise ValueError("Original sealed registry changed")
    if Path(args.registry).resolve() != registries[args.benchmark]:
        raise ValueError("Selection request names a different registry")
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--benchmark", required=True, choices=("main", "historical"))
    parser.add_argument("--operational-addendum", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    amendment = validate_operational_addendum(args)
    amendment_sha = file_sha256(args.operational_addendum)
    original_sha = file_sha256(ORIGINAL)
    entries_sha = file_sha256(args.entries)
    registry_sha = file_sha256(args.registry)
    entries = json.loads(Path(args.entries).read_text())
    protocol = json.loads(Path(args.registry).read_text())
    helper_sha = file_sha256(__file__)
    if protocol.get("selection_launch_helper_sha256") != original_sha:
        raise ValueError("Selection launch helper is not registered at its exact source hash")
    jobs = [j for j in entries["commands"] if "_perm" in j["label"] or j["label"] == "frozen_head"]
    if len(jobs) != 37 or len({j["label"] for j in jobs}) != 37:
        raise ValueError("Queue must include exactly36 original inferences plus frozen head")
    signatures = {j["label"]: j["signature"] for j in entries["registered_analyses"]}
    env = dict(os.environ, ND_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(args.gpu))
    for job in jobs:
        command = job["argv"]
        if Path(command[1]).resolve() not in {
            (FOLDER / "evaluate.py").resolve(),
            (FOLDER / "evaluate_frozen_head.py").resolve(),
        }:
            raise ValueError("Only locked selector evaluators may run")
        if command[0] != "/data/users/dmitry/venvs/proof-state/bin/python":
            raise ValueError("Only the pinned local Python environment may run")
        values = pairs(command)
        if Path(values["--frozen-test-protocol"]).resolve() != Path(args.registry).resolve():
            raise ValueError("Command references another registry")
        if values["--split"] != "test":
            raise ValueError("Queue is only for registered TEST inference")
        if Path(values["--out"]).exists() or Path(values["--out"]).is_symlink():
            raise ValueError("Refuse overwriting any existing result directory: " + values["--out"])
        validate_frozen_evaluation(protocol, values["--public"], signatures[job["label"]])
        check = subprocess.run(
            command + ["--signature-only"], env=env, capture_output=True, text=True, check=True
        )
        if json.loads(check.stdout) != signatures[job["label"]]:
            raise ValueError("Current source/model differs from frozen command signature")
    if args.dry_run:
        print(json.dumps({"registered_jobs": len(jobs), "out": args.out, "launched": False}))
        return
    occupancy = admit_shared_gpu(args.gpu, amendment["minimum_free_memory_mib"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "queue_source.py").write_bytes(Path(__file__).read_bytes())

    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    write(
        out / "config.json",
        {
            "args": vars(args),
            "registry_sha256": registry_sha,
            "entries_sha256": entries_sha,
            "initial_occupancy": occupancy,
            "operational_addendum_path": str(Path(args.operational_addendum).resolve()),
            "operational_addendum_sha256": amendment_sha,
            "original_launcher_sha256": original_sha,
            "successor_launcher_sha256": helper_sha,
            "original_operational_addendum_sha256": COMMON_ADDENDUM_SHA256,
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "jobs": len(jobs),
        },
    )
    source = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
    for job in jobs:
        if file_sha256(__file__) != helper_sha:
            raise ValueError("Launch helper source changed during queue")
        if file_sha256(args.registry) != registry_sha or file_sha256(args.entries) != entries_sha:
            raise ValueError("Immutable registry/inventory changed during queue")
        if file_sha256(args.operational_addendum) != amendment_sha:
            raise ValueError("Selection operational addendum changed during queue")
        validate_operational_addendum(args)
        destination = Path(pairs(job["argv"])["--out"])
        if destination.exists() or destination.is_symlink():
            raise ValueError("Result appeared after preflight; refusing overwrite")
        launch = out / job["label"]
        launch.mkdir()
        snapshot = launch / "source_snapshot"
        snapshot.mkdir()
        paths = list(FOLDER.glob("*.py")) + [
            source / name
            for name in [
                "model.py",
                "best_model.py",
                "lean_tok.py",
                "tokenizer.py",
                "nd2lean.py",
                "record.py",
                "sample.py",
                "state_sample.py",
                "state_env.py",
            ]
        ]
        paths += [
            source / "nd_verify/verify.py", source / "nd_verify/__init__.py",
            Path(args.operational_addendum).resolve(), COMMON_ADDENDUM,
        ]
        hashes = {}
        for path in paths:
            relative = path.relative_to(ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = file_sha256(path)
        if (
            file_sha256(args.registry) != registry_sha
            or file_sha256(args.entries) != entries_sha
            or file_sha256(args.operational_addendum) != amendment_sha
        ):
            raise ValueError("Immutable inputs changed before child launch")
        validate_operational_addendum(args)
        admission = admit_shared_gpu(args.gpu, amendment["minimum_free_memory_mib"])
        started = time.monotonic()
        with (launch / "stdout_stderr.log").open("w") as stream:
            process = subprocess.Popen(
                job["argv"], env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            manifest = {
                "argv": job["argv"],
                "pid": process.pid,
                "gpu": args.gpu,
                "started_utc": datetime.now(UTC).isoformat(),
                "source_hashes": hashes,
                "registry_sha256": registry_sha,
                "entries_sha256": entries_sha,
                "operational_addendum_path": str(Path(args.operational_addendum).resolve()),
                "operational_addendum_sha256": amendment_sha,
                "original_launcher_sha256": original_sha,
                "successor_launcher_sha256": helper_sha,
                "original_operational_addendum_sha256": COMMON_ADDENDUM_SHA256,
                "gpu_admission": admission,
            }
            write(launch / "run_manifest.json", manifest)
            code = process.wait()
        manifest.update(
            exit_code=code,
            process_wall_seconds_monotonic=time.monotonic() - started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        write(launch / "run_manifest.json", manifest)
        if code:
            raise RuntimeError(
                f"Registered inference failed: {job['label']}, exit{code}; see saved log"
            )
        print(
            json.dumps(
                {"completed": job["label"], "seconds": manifest["process_wall_seconds_monotonic"]}
            ),
            flush=True,
        )
    write(
        out / "completion.json",
        {"completed_jobs": len(jobs), "completed_utc": datetime.now(UTC).isoformat()},
    )


if __name__ == "__main__":
    main()
