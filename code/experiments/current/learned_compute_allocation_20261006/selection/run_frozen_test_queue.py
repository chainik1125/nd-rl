"""Run only the 37 registered selector inferences, with immutable job records."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--gpu", required=True, choices=("1", "2", "3", "4"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    entries = json.loads(Path(args.entries).read_text())
    protocol = json.loads(Path(args.registry).read_text())
    helper_sha = file_sha256(__file__)
    if protocol.get("selection_launch_helper_sha256") != helper_sha:
        raise ValueError("Selection launch helper is not registered at its exact source hash")
    jobs = [j for j in entries["commands"] if "_perm" in j["label"] or j["label"] == "frozen_head"]
    if len(jobs) != 37 or len({j["label"] for j in jobs}) != 37:
        raise ValueError("Queue must include exactly36 original inferences plus frozen head")
    signatures = {j["label"]: j["signature"] for j in entries["registered_analyses"]}
    env = dict(os.environ, ND_OFFLINE="1", CUDA_VISIBLE_DEVICES=args.gpu)
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
        if Path(values["--out"]).exists():
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
    occupancy = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + args.gpu,
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory, utilization = [int(v.strip()) for v in occupancy.strip().split(",")]
    if memory > 500 or utilization > 5:
        raise ValueError("Assigned GPU is occupied; queue not launched")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    registry_sha = file_sha256(args.registry)
    entries_sha = file_sha256(args.entries)
    (out / "queue_source.py").write_bytes(Path(__file__).read_bytes())

    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    write(
        out / "config.json",
        {
            "args": vars(args),
            "registry_sha256": registry_sha,
            "entries_sha256": entries_sha,
            "initial_occupancy": occupancy.strip(),
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
        if Path(pairs(job["argv"])["--out"]).exists():
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
        paths += [source / "nd_verify/verify.py", source / "nd_verify/__init__.py"]
        hashes = {}
        for path in paths:
            relative = path.relative_to(ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = file_sha256(path)
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
