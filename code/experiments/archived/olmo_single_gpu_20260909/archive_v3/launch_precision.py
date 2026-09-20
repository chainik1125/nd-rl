"""Own exactly one idle local GPU for a bounded, recorded sprint job."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "r1-sprint-20260909-1625"


def gpu_snapshot():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = {row.split(",")[0].strip() for row in apps.splitlines() if row.strip()}
    result = []
    for row in rows.splitlines():
        idx, uuid, mem, util = [field.strip() for field in row.split(",")]
        result.append(dict(index=int(idx), uuid=uuid, memory_mib=int(mem), utilization=int(util), process_present=uuid in occupied))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    out = Path(spec["job_dir"]).resolve()
    if not out.is_relative_to(ROOT / "jobs"):
        raise ValueError("Job metadata must be inside this sprint's jobs directory")
    out.mkdir(parents=True, exist_ok=False)
    (out / "spec.json").write_bytes(spec_bytes)
    (out / "launcher_pid.json").write_text(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    events = (out / "events.jsonl").open("a", buffering=1)
    def event(kind, **fields):
        record = dict(event=kind, utc=datetime.now(timezone.utc).isoformat(), **fields)
        events.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    allowed = spec.get("candidate_gpus", [3, 4])
    if any(i in {0, 2} for i in allowed):
        authority = json.loads((ROOT / "gpu_authorization.json").read_text())
        if authority.get("source") != "explicit_user_reply" or not set(allowed) <= set(authority["allowed_gpus"]):
            raise ValueError("Reserved GPU requires recorded user authorization")
    entry = (ROOT / "code" / spec["entrypoint"]).resolve()
    if entry.parent != ROOT / "code" or entry.name not in {"capacity.py", "train.py", "evaluate.py", "pipeline.py", "diagnose_likelihood.py", "diagnose_precision.py"}:
        raise ValueError("Only sprint training/evaluation entrypoints are allowed")
    locks = []
    deadline = datetime.fromisoformat(spec["not_after_utc"]).timestamp()
    queue_deadline = min(deadline, time.time() + spec.get("queue_timeout_seconds", 3600))
    chosen = None
    child = None
    try:
        for dependency in spec.get("wait_for_success", []):
            path = Path(dependency["path"]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Dependencies must belong to this sprint")
            while not path.exists():
                if time.time() >= queue_deadline:
                    raise TimeoutError("Dependency did not complete before queue deadline")
                event("waiting_for_dependency", path=str(path))
                time.sleep(15)
            result = json.loads(path.read_text())
            if any(result.get(key) != value for key, value in dependency["equals"].items()):
                raise ValueError("A required predecessor did not succeed")
        while chosen is None and time.time() < queue_deadline:
            snapshots = gpu_snapshot()
            for gpu in snapshots:
                idx = gpu["index"]
                if idx not in allowed or gpu["process_present"] or gpu["memory_mib"] > 100 or gpu["utilization"] > 5:
                    continue
                paths = [Path("/tmp") / f"olmo_single_gpu_{idx}.lock"]
                # Cooperate with the prior sprint's scheduler and older per-GPU leases.
                if (PRIOR / "leases" / f"gpu_{idx}.lock").exists():
                    paths.append(PRIOR / "leases" / f"gpu_{idx}.lock")
                if (Path("/tmp") / f"crra_sse_gpu_{idx}.lock").exists():
                    paths.append(Path("/tmp") / f"crra_sse_gpu_{idx}.lock")
                trial = []
                try:
                    for path in paths:
                        handle = path.open("a+")
                        trial.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    current = next(row for row in gpu_snapshot() if row["index"] == idx)
                    if current["process_present"] or current["memory_mib"] > 100 or current["utilization"] > 5:
                        raise BlockingIOError("GPU became occupied during lease acquisition")
                    locks = trial
                    chosen = current
                    break
                except BlockingIOError:
                    for handle in trial:
                        handle.close()
            if chosen is None:
                event("waiting_for_idle_gpu", candidates=allowed, snapshot=snapshots)
                time.sleep(15)
        if chosen is None:
            raise TimeoutError("No idle authorized GPU before queue deadline")
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=chosen["uuid"], TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                   HF_HOME=str(ROOT / "cache" / "huggingface"),
                   HF_DATASETS_CACHE=str(ROOT / "cache" / "datasets"),
                   MPLCONFIGDIR=str(ROOT / "cache" / "matplotlib"),
                   TRITON_CACHE_DIR=str(ROOT / "cache" / "triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache" / "inductor"),
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        command = [str(ROOT / ".venv" / "bin" / "python"), "-u", str(entry), *spec.get("args", [])]
        event("acquired", gpu=chosen, command=command,
              spec_sha256=hashlib.sha256(spec_bytes).hexdigest(), entry_sha256=hashlib.file_digest(entry.open("rb"), "sha256").hexdigest())
        started = time.time()
        with (out / "worker.log").open("w", buffering=1) as logfile:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            (out / "pid.json").write_text(json.dumps({"pid": child.pid, "gpu": chosen["index"], "started_unix": started}))
            event("started", pid=child.pid)
            timeout = min(spec.get("timeout_seconds", 3600), max(1, deadline-started))
            try:
                status = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise TimeoutError("Owned worker exceeded its preregistered deadline")
        completion = dict(exit_code=status, elapsed_seconds=time.time()-started, gpu=chosen["index"])
        (out / "completed.json").write_text(json.dumps(completion, indent=2) + "\n")
        event("completed", **completion)
        if status:
            raise SystemExit(status)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        event("failed", error=type(exc).__name__, message=str(exc))
        if not (out / "completed.json").exists():
            (out / "completed.json").write_text(json.dumps({"exit_code": 1,
                "status": "failed", "error": type(exc).__name__, "message": str(exc),
                "gpu": chosen["index"] if chosen else None,
                "utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        raise
    finally:
        for handle in locks:
            handle.close()
        events.close()


if __name__ == "__main__":
    main()
