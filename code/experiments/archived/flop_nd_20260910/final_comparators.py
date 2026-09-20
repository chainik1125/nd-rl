"""Release and run every prospectively specified CPU comparator endpoint.

Waits for all LM endpoints and symbolic training completion, then binds exact
checkpoint, population, program, and command hashes before opening final inputs.
No outcome enters endpoint selection. Existing output directories are never
overwritten. A per-task failure does not cancel unrelated frozen comparisons.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import common


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_hashes(mapping):
    for path, digest in mapping.items():
        if common.sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")


def freeze_release(spec, spec_path, lm_release):
    verify_hashes(spec["input_sha256"])
    lm = json.loads(Path(lm_release).read_text())
    if lm.get("status") != "frozen":
        raise ValueError("LM endpoints have not been frozen")
    for name, path in lm["checkpoint_paths"].items():
        if common.sha(path) != lm["checkpoint_sha256"][name]:
            raise ValueError("LM endpoint changed")
    for path in spec["required_training_summaries"]:
        result = json.loads(Path(path).read_text())
        if result.get("status") not in ("complete", "completed"):
            raise ValueError(f"Incomplete symbolic training: {path}")
    endpoints = sorted({p for t in spec["tasks"] for p in t.get("endpoint_paths", [])})
    record = {
        "status": "released", "all_comparator_endpoints_frozen": True,
        "frozen_utc": utc(), "protocol_sha256": common.sha(spec_path),
        "lm_endpoints_sha256": common.sha(lm_release),
        "checkpoint_sha256": {p: common.sha(p) for p in endpoints},
        "input_sha256": spec["input_sha256"],
        "commands_sha256": {t["name"]: common.digest(t["command"]) for t in spec["tasks"]},
        "selection": spec["selection"],
    }
    return record


def verify_task(task, release, spec_path):
    if release.get("status") != "released" or not release.get("all_comparator_endpoints_frozen"):
        raise ValueError("Comparator endpoint release is absent")
    if release["protocol_sha256"] != common.sha(spec_path):
        raise ValueError("Comparator protocol changed after release")
    if release["commands_sha256"].get(task["name"]) != common.digest(task["command"]):
        raise ValueError("Final inference command/profile changed")
    verify_hashes(release["input_sha256"])
    verify_hashes({p: release["checkpoint_sha256"][p] for p in task.get("endpoint_paths", [])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    spec_hash = common.sha(args.config)
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00")).timestamp()
    status = {"status": "waiting_for_all_endpoints", "pid": os.getpid(),
              "started_utc": utc(), "protocol_sha256": spec_hash, "tasks": {}}
    common.write_json(out / "progress.json", status)
    stopped = threading.Event()
    children = {}
    lock = threading.Lock()

    def stop(signum, frame):
        stopped.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        ready = [spec["lm_endpoints_release"], *spec["required_training_summaries"]]
        while not all(Path(p).exists() for p in ready):
            if common.sha(args.config) != spec_hash:
                raise ValueError("Frozen comparator task list changed")
            if stopped.is_set() or time.time() >= deadline - 900:
                raise TimeoutError("All endpoints were not ready before final comparison reserve")
            stopped.wait(10)
        release = freeze_release(spec, args.config, spec["lm_endpoints_release"])
        release_path = out / "release.json"
        common.write_json(release_path, release)
        status.update(status="evaluating", released_utc=utc())
        common.write_json(out / "progress.json", status)

        def run(task):
            name = task["name"]
            record = {"status": "starting", "started_utc": utc()}
            started = time.monotonic()
            child = None
            try:
                verify_task(task, release, args.config)
                if stopped.is_set() or time.time() >= deadline:
                    record.update(status="deadline_or_interruption")
                    return name, record
                command = list(task["command"])
                if task["kind"] in ("classical", "legal_lm"):
                    local_release = {**release}
                    if task["kind"] == "classical":
                        local_release["protocol_sha256"] = common.sha(task["classical_protocol"])
                    else:
                        local_release["data_sha256"] = common.sha(task["data"])
                        local_release["inference_profiles"] = [task["profile"]]
                    local_path = out / (name + "_release.json")
                    common.write_json(local_path, local_release)
                    command.extend(["--release", str(local_path)])
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                with (out / (name + ".log")).open("w") as log:
                    child = subprocess.Popen(command, cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                    with lock:
                        children[name] = child
                        record.update(status="active", pid=child.pid, command=command)
                        status["tasks"][name] = record.copy()
                        common.write_json(out / "progress.json", status)
                    try:
                        code = child.wait(timeout=max(.1, min(task["timeout_seconds"], deadline - time.time())))
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                        record.update(status="timeout")
                    else:
                        record.update(status="completed" if code == 0 else "failed", exit_code=code)
                        verify_task(task, release, args.config)
                        if code == 0:
                            result = json.loads(Path(task["result_path"]).read_text())
                            if result.get("status") not in ("complete", "completed"):
                                raise ValueError("Final comparator returned only partial coverage")
                            record["result_sha256"] = common.sha(task["result_path"])
            except Exception as error:
                record.update(status="failed", error=type(error).__name__, message=str(error))
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                with lock:
                    children.pop(name, None)
                record.update(elapsed_seconds=time.monotonic() - started, finished_utc=utc())
            return name, record

        with ThreadPoolExecutor(max_workers=spec["parallel_tasks"]) as executor:
            futures = [executor.submit(run, task) for task in spec["tasks"]]
            for future in as_completed(futures):
                name, record = future.result()
                with lock:
                    status["tasks"][name] = record
                    common.write_json(out / "progress.json", status)
                print(json.dumps({"name": name, "status": record["status"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
        status["status"] = "completed" if all(r["status"] == "completed" for r in status["tasks"].values()) else "completed_with_failures"
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__, message=str(error))
        raise
    finally:
        status["finished_utc"] = utc()
        common.write_json(out / "result.json", status)
        common.write_json(out / "progress.json", status)


if __name__ == "__main__":
    main()
