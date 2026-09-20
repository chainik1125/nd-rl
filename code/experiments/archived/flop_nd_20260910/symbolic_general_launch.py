"""Freeze and sequentially launch the root-authorized CPU diagnostic matrix."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import common

ROOT = common.ROOT
PLAN = ROOT / "reports/symbolic_general_main_manifest.json"
LEDGER = ROOT / "results/symbolic_general_main_ledger.jsonl"


def job_specs():
    jobs = []
    for seed in (4301, 4302):
        for mode in ("full", "reset"):
            tag = f"{mode}_s{seed}"
            out = ROOT / f"results/symbolic_general_{tag}"
            command = [sys.executable, str(ROOT / "code/symbolic_general_train.py"),
                "--out", str(out), "--mode", mode, "--seed", str(seed), "--steps", "131072",
                "--width", "500", "--layers", "2", "--threads", "4", "--max-seconds", "3600"]
            jobs.append({"tag": tag, "out": str(out), "command": command, "timeout_seconds": 3660})
    for seed in (4301, 4302):
        tag = f"bc_s{seed}"
        out = ROOT / f"results/symbolic_general_{tag}"
        command = [sys.executable, str(ROOT / "code/symbolic_general_bc.py"), "--out", str(out),
            "--seed", str(seed), "--full-batch-updates", "1000", "--width", "500", "--layers", "2",
            "--threads", "4", "--max-seconds", "3600"]
        jobs.append({"tag": tag, "out": str(out), "command": command, "timeout_seconds": 3660})
    return jobs


def record(event):
    with LEDGER.open("a") as f:
        f.write(json.dumps(event) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    identities = {str(p.relative_to(ROOT)): common.sha(p) for p in sorted((ROOT / "code").glob("symbolic_general_*.py"))}
    for name in ["sources", "targets"]:
        path = ROOT / f"data/v1/{name}.jsonl"
        identities[str(path.relative_to(ROOT))] = common.sha(path)
    identities["code/nd_environment.py"] = common.sha(ROOT / "code/nd_environment.py")
    identities["reports/symbolic_general_main_protocol.md"] = common.sha(ROOT / "reports/symbolic_general_main_protocol.md")
    if not PLAN.exists():
        common.write_json(PLAN, {"frozen_unix": time.time(), "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "identities": identities, "jobs": job_specs(), "external_source_count": 104,
            "unique_bc_actions": 278, "bc_presentations_per_seed": 278000,
            "target_statement_count": 512, "max_concurrent_jobs": 1, "cpu_threads": 4,
            "deadline_utc": "2026-09-11T03:30:00+00:00", "development_released": True,
            "final_released": False, "online_discovery_curriculum": False})
        print("Frozen:", PLAN, flush=True)
    plan = json.loads(PLAN.read_text())
    if identities != plan["identities"]:
        raise ValueError("Code/input/protocol identity changed after freeze")
    if not args.launch:
        return
    deadline = datetime.datetime.fromisoformat(plan["deadline_utc"]).timestamp()
    process_env = os.environ.copy()
    process_env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    for job in plan["jobs"]:
        if time.time() >= deadline:
            raise RuntimeError("Owned launch deadline reached")
        out = Path(job["out"])
        if (out / "symbolic_general_train_summary.json").exists():
            summary = json.loads((out / "symbolic_general_train_summary.json").read_text())
            if summary["status"] != "complete":
                raise RuntimeError(f"Incomplete prior job requires explicit recovery: {job['tag']}")
            print("Already complete:", job["tag"], flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with (out / "symbolic_general_train.log").open("w") as log:
            proc = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, env=process_env)
            record({"event": "start", "tag": job["tag"], "pid": proc.pid, "driver_pid": os.getpid(),
                    "start_unix": started, "command": job["command"], "manifest_sha256": common.sha(PLAN),
                    "timeout_seconds": min(job["timeout_seconds"], deadline - started)})
            print("Started", job["tag"], "pid", proc.pid, flush=True)
            try:
                code = proc.wait(timeout=min(job["timeout_seconds"], deadline - started))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                record({"event": "timeout", "tag": job["tag"], "pid": proc.pid, "elapsed_seconds": time.time() - started})
                raise
        record({"event": "finish", "tag": job["tag"], "pid": proc.pid, "exit_code": code,
                "elapsed_seconds": time.time() - started})
        print("Finished", job["tag"], "exit", code, "seconds", round(time.time() - started, 1), flush=True)
        if code:
            raise RuntimeError(f"Training subprocess failed: {job['tag']}")
        summary = json.loads((out / "symbolic_general_train_summary.json").read_text())
        if summary["status"] != "complete":
            raise RuntimeError(f"Job stopped before its frozen budget: {job['tag']}")


if __name__ == "__main__":
    main()
