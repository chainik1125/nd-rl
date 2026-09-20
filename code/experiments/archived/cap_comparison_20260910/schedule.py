"""Assign each frozen arm once to an idle GPU; copy remote results locally."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from dispatch import state


PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes"]


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_state(root):
    code = "import sys,json; sys.path.insert(0," + repr(str(root / "code")) + "); from dispatch import state; print(json.dumps(state()))"
    result = subprocess.run([*SSH, "simplex2", shlex.join([PYTHON, "-c", code])],
                            text=True, capture_output=True, timeout=15, check=True)
    return {int(k): v for k, v in json.loads(result.stdout).items()}


def sync_results(root):
    # Transfer committed artifacts and logs; partial files are renamed atomically
    # by rsync. Never copy back code/data or rewrite immutable source manifests.
    for directory in ("runs", "jobs"):
        subprocess.run(["rsync", "-a", "--exclude=*.tmp", "--exclude=*.lock",
                        f"simplex2:{root}/{directory}/", str(root / directory) + "/"],
                       check=True, timeout=180, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / "reports/scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "reports/scheduler_state.json"
    if status_path.exists():
        raise RuntimeError("Scheduler state already exists; inspect owned jobs before a manual recovery")
    protocol = json.loads((root / "plans/protocol.json").read_text())
    frozen = sha(root / "plans/protocol.json")
    pending = list(protocol["arms"])
    running, finished, assignments = {}, {}, {}
    def persist():
        write(status_path, {"updated_utc": datetime.now(timezone.utc).isoformat(),
                            "protocol_sha256": frozen, "pending": pending,
                            "assignments": assignments,
                            "running": {k: {"pid": v["process"].pid, "host": v["host"], "gpu": v["gpu"]} for k,v in running.items()},
                            "finished": finished})
    # A crash must leave a recovery record before any possible child launch.
    persist()
    for directory in ("jobs", "logs", "plans/jobs"):
        (root / directory).mkdir(exist_ok=True)
    last_sync = 0
    while pending or running:
        assert sha(root / "plans/protocol.json") == frozen, "Frozen protocol changed"
        if any(v["host"] == "simplex2" for v in running.values()) and time.monotonic() - last_sync > 60:
            try:
                sync_results(root)
                last_sync = time.monotonic()
            except (subprocess.SubprocessError, OSError) as error:
                print(json.dumps({"event": "sync_error", "error": str(error)}), flush=True)
        for name, item in list(running.items()):
            rc = item["process"].poll()
            if rc is None:
                continue
            item["log"].close()
            if item["host"] == "simplex2":
                sync_results(root)
            result_path = root / "jobs" / name / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            complete = root / "runs" / name / "pipeline_completed.json"
            success = rc == 0 and result.get("returncode") == 0 and complete.exists()
            finished[name] = {"host": item["host"], "gpu": item["gpu"], "success": success,
                              "dispatcher_returncode": rc, "result": result}
            print(json.dumps({"event": "arm_finished", "arm": name, **finished[name]}), flush=True)
            del running[name]
        states = {}
        if pending:
            states["simplex1"] = state()
            try:
                states["simplex2"] = remote_state(root)
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                print(json.dumps({"event": "remote_unavailable", "error": str(error)}), flush=True)
        for host, devices in states.items():
            candidates = [2] if host == "simplex1" else [1, 2, 3, 4]
            reserved = [0, 1] if host == "simplex1" else [5, 7]
            for gpu in candidates:
                if not pending:
                    break
                if any(v["host"] == host and v["gpu"] == gpu for v in running.values()):
                    continue
                device = devices[gpu]
                if device["pids"] or device["memory"] > 64 or device["util"]:
                    continue
                if host == "simplex2":
                    try:
                        subprocess.run([*SSH, host, shlex.join(["mkdir", "-p", str(root / "jobs"), str(root / "plans/jobs")])],
                                       check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except (subprocess.SubprocessError, OSError) as error:
                        print(json.dumps({"event": "remote_unavailable", "error": str(error)}), flush=True)
                        break
                name = pending.pop(0)
                assignments[name] = {"host": host, "gpu": gpu, "status": "launch_intent"}
                persist()
                config = root / "plans" / f"{name}.json"
                inputs = {str(root / p): h for p, h in protocol["input_sha256"].items()}
                inputs[str(config)] = sha(config)
                # Dispatch rechecks both occupancy and an exclusive per-device
                # lease for 15 seconds before starting the owned process group.
                job = {"name": name, "root": str(root), "cwd": str(root),
                       "candidate_gpus": [gpu], "reserved_gpus": reserved,
                       "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat(),
                       "timeout_seconds": 12 * 3600, "input_sha256": inputs,
                       "env": {"PYTHONPATH": str(root / "snapshots/main_v1/vendor"),
                               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                       "command": [PYTHON, str(root / "snapshots/main_v1/pipeline.py"), "--config", str(config)]}
                job_path = root / "plans/jobs" / f"{name}.json"
                write(job_path, job)
                command = [PYTHON, str(root / "code/dispatch.py"), "--job", str(job_path)]
                if host == "simplex2":
                    subprocess.run(["rsync", "-a", str(job_path), f"simplex2:{job_path}"], check=True)
                    command = [*SSH, host, shlex.join(command)]
                log = (root / "logs" / f"{name}_dispatch.log").open("w")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                running[name] = {"process": process, "log": log, "host": host, "gpu": gpu}
                assignments[name]["status"] = "dispatched"
                persist()
                print(json.dumps({"event": "arm_assigned", "arm": name, "host": host, "gpu": gpu}), flush=True)
        persist()
        if pending or running:
            time.sleep(20)
    if not all(v["success"] for v in finished.values()):
        raise SystemExit("One or more arms failed; no failed arm was silently retried")


if __name__ == "__main__":
    main()
