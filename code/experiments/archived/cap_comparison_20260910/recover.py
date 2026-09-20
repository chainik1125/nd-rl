"""Coordinate infrastructure retries without changing the frozen experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from dispatch import state
from schedule import PYTHON, SSH, remote_state, sha, sync_results, write


def idle(device):
    return not device["pids"] and device["memory"] <= 64 and device["util"] == 0


def owned_alive(pid, job_path):
    """Identify an adopted dispatcher by its entire recorded job argument."""
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return False
    command = path.read_bytes().split(b"\0")
    return str(job_path).encode() in command and any(p.endswith(b"/code/dispatch.py") for p in command)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="Adopt recorded local workers after coordinator loss")
    args = ap.parse_args()
    root = args.root.resolve()
    lock = (root / "reports/recovery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "reports/recovery_state.json"
    if status_path.exists() and not args.resume:
        raise RuntimeError("Recovery state exists; audit owned jobs before restarting coordinator")
    previous = json.loads(status_path.read_text()) if args.resume else None
    protocol = json.loads((root / "plans/protocol.json").read_text())
    receipt = json.loads((root / "reports/recovery_protocol.json").read_text())
    frozen = sha(root / "plans/protocol.json")
    assert frozen == receipt["training_protocol_sha256"]
    if previous:
        assert previous["protocol_sha256"] == frozen
        pending = previous["pending"]
        finished, assignments = previous["finished"], previous["assignments"]
        retained = previous["running"]
        assert set(pending) | set(finished) | set(retained) == set(protocol["arms"])
        assert not (set(pending) & set(retained) or set(pending) & set(finished) or set(retained) & set(finished))
        for arm, item in finished.items():
            if item["success"]:
                assert (root / "runs" / arm / "pipeline_completed.json").exists()
                assert json.loads((root / "jobs" / item["job"] / "result.json").read_text()) == item["result"]
    else:
        pending = [a for a in protocol["arms"] if a not in receipt["retained_live_arm"]]
        finished, assignments = {}, {}
        retained = receipt["retained_live_arm"]
    running = {}
    for arm, old in retained.items():
        job_name = old.get("job", arm)
        job_path = root / "plans/jobs" / f"{job_name}.json"
        assert old["host"] == "simplex1", "Remote worker adoption requires a remote ownership audit"
        assert owned_alive(old["pid"], job_path) or (root / "jobs" / job_name / "result.json").exists()
        running[arm] = {**old, "job": job_name, "process": None, "log": None, "adopted": True}
        assignments[arm] = {**old, "job": job_name, "status": "adopted_live"}

    def persist():
        write(status_path, {"updated_utc": datetime.now(timezone.utc).isoformat(),
                           "protocol_sha256": frozen, "pending": pending,
                           "assignments": assignments,
                           "running": {a: {k: v for k, v in item.items() if k not in ("process", "log")}
                                       for a, item in running.items()}, "finished": finished})

    persist()
    quiet_since = {}
    last_sync = 0
    while pending or running:
        assert sha(root / "plans/protocol.json") == frozen
        if any(v["host"] == "simplex2" for v in running.values()) and time.monotonic() - last_sync > 60:
            try:
                sync_results(root)
                last_sync = time.monotonic()
            except (subprocess.SubprocessError, OSError) as error:
                print(json.dumps({"event": "sync_error", "error": str(error)}), flush=True)
        for arm, item in list(running.items()):
            if item["adopted"]:
                if owned_alive(item["pid"], root / "plans/jobs" / f"{item['job']}.json"):
                    continue
                rc = None
            else:
                rc = item["process"].poll()
                if rc is None:
                    continue
                item["log"].close()
            if item["host"] == "simplex2":
                sync_results(root)
            result_path = root / "jobs" / item["job"] / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            success = result.get("returncode") == 0 and (root / "runs" / arm / "pipeline_completed.json").exists()
            finished[arm] = {"host": item["host"], "gpu": item["gpu"], "job": item["job"],
                             "success": success, "dispatcher_returncode": rc, "result": result}
            del running[arm]
            print(json.dumps({"event": "arm_finished", "arm": arm, **finished[arm]}), flush=True)
        states = {}
        if pending:
            states["simplex1"] = state()
            try:
                states["simplex2"] = remote_state(root)
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                quiet_since = {k: v for k,v in quiet_since.items() if k[0] != "simplex2"}
                print(json.dumps({"event": "remote_unavailable", "error": str(error)}), flush=True)
        for host, devices in states.items():
            if pending and host not in receipt.get("allowed_hosts_by_arm", {}).get(pending[0], ["simplex1", "simplex2"]):
                continue
            reserved = [1, 7] if host == "simplex1" else [5, 7]
            for gpu in sorted(set(devices) - set(reserved)):
                key = (host, gpu)
                if not idle(devices[gpu]) or any(v["host"] == host and v["gpu"] == gpu for v in running.values()):
                    quiet_since.pop(key, None)
                    continue
                quiet_since.setdefault(key, time.monotonic())
                required = 30 if host == "simplex1" else receipt["remote_quiet_seconds"]
                if not pending or time.monotonic() - quiet_since[key] < required:
                    continue
                arm = pending.pop(0)
                job_name = arm + ("_retry1" if arm in receipt["interrupted"] else "")
                assignments[arm] = {"host": host, "gpu": gpu, "job": job_name, "status": "launch_intent"}
                persist()
                config = root / "plans" / f"{arm}.json"
                assert not (root / "runs" / arm).exists(), "Retry must start in an empty output path"
                inputs = {str(root / p): h for p,h in protocol["input_sha256"].items()}
                for path, digest in inputs.items():
                    assert sha(Path(path)) == digest, f"Frozen input changed: {path}"
                job = {"name": job_name, "root": str(root), "cwd": str(root),
                       "candidate_gpus": [gpu], "reserved_gpus": reserved,
                       "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat(),
                       "timeout_seconds": 43200, "input_sha256": inputs,
                       "env": {"PYTHONPATH": str(root / "snapshots/main_v1/vendor"),
                               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                       "command": [PYTHON, str(root / "snapshots/main_v1/pipeline.py"), "--config", str(config)]}
                job_path = root / "plans/jobs" / f"{job_name}.json"
                write(job_path, job)
                command = [PYTHON, str(root / "code/dispatch.py"), "--job", str(job_path)]
                if host == "simplex2":
                    subprocess.run(["rsync", "-a", str(job_path), f"simplex2:{job_path}"], check=True)
                    command = [*SSH, host, shlex.join(command)]
                log = (root / "logs" / f"{job_name}_dispatch.log").open("w")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                running[arm] = {"process": process, "log": log, "pid": process.pid,
                                "host": host, "gpu": gpu, "job": job_name, "adopted": False}
                assignments[arm]["status"] = "dispatched"
                persist()
                print(json.dumps({"event": "arm_assigned", "arm": arm, "host": host, "gpu": gpu,
                                  "job": job_name, "quiet_seconds": required}), flush=True)
        persist()
        if pending or running:
            time.sleep(20)
    if not all(v["success"] for v in finished.values()):
        raise SystemExit("Incomplete arms require an explicit infrastructure audit; none silently retried")


if __name__ == "__main__":
    main()
