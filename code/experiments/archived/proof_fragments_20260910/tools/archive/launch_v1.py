"""Execute one frozen Python job on an idle, exclusively leased local GPU."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "r1-sprint-20260909-1625"
PYTHON = Path("/data/users/dmitry/simplex-research/.venv/bin/python")
HARD_END = datetime.fromisoformat("2026-09-11T04:00:00+00:00").timestamp()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate(spec, spec_path):
    if not spec_path.is_relative_to(ROOT / "configs"):
        raise ValueError("Job specs must be inside this sprint's configs directory")
    name = spec["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Invalid job name")
    command = spec["command"]
    if command[0] != str(PYTHON) or not Path(command[1]).resolve().is_relative_to(ROOT / "snapshots"):
        raise ValueError("Only Python entrypoints in frozen sprint snapshots can run")
    if str(Path(command[1]).resolve()) not in spec["input_sha256"]:
        raise ValueError("Entrypoint must have a frozen identity")
    gpus = spec.get("candidate_gpus", [4])
    if not gpus or not set(gpus) <= {1, 3, 4, 5, 6, 7}:
        raise ValueError("Reserved or invalid GPU requested")
    end = datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00")).timestamp()
    if not time.time() < end <= HARD_END:
        raise ValueError("Job must end within the sprint")
    if not 0 < spec["timeout_seconds"] <= 9 * 3600:
        raise ValueError("Invalid timeout")
    for path in spec["input_sha256"]:
        p = Path(path).resolve()
        if any("sealed" in x.lower() for x in p.parts) or "textbook_final" in p.name:
            raise ValueError("Protected payload cannot be launcher metadata")
    return end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    deadline = validate(spec, spec_path)
    spec_sha = digest(spec_path)
    helper_path = PRIOR / "code/dispatch.py"
    helper_spec = importlib.util.spec_from_file_location("prior_gpu_dispatch", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    out = ROOT / "jobs" / spec["name"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "launcher.json").write_text(json.dumps({"pid": os.getpid(), "utc": utc(),
        "launcher_sha256": digest(__file__), "helper_sha256": digest(helper_path)}))
    events = (out / "events.jsonl").open("a", buffering=1)

    def event(kind, **fields):
        row = dict(utc=utc(), kind=kind, **fields)
        events.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def verify():
        if digest(spec_path) != spec_sha:
            raise ValueError("Job spec changed")
        for path, sha in spec["input_sha256"].items():
            if digest(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")

    child = None
    leases = []
    chosen = None
    started = None
    result = {"status": "failed", "exit_code": None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: (_ for _ in ()).throw(SystemExit(128 + s)))
    try:
        verify()
        queue_end = min(deadline, time.time() + spec.get("queue_timeout_seconds", 300))
        while chosen is None and time.time() < queue_end:
            current = helper.state(timeout=min(5.0, max(.01, queue_end-time.time())))
            for idx in spec.get("candidate_gpus", [4]):
                gpu = current[idx]
                if gpu["pids"] or gpu["memory"] > 64 or gpu["util"]:
                    continue
                paths = [Path("/tmp") / f"olmo_single_gpu_{idx}.lock",
                         ROOT / "leases" / f"gpu_{idx}.lock"]
                paths += [p for p in (PRIOR / "leases" / f"gpu_{idx}.lock",
                         Path("/tmp") / f"crra_sse_gpu_{idx}.lock") if p.exists()]
                trial = []
                try:
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handle = path.open("a+")
                        trial.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fresh = helper.state(timeout=min(5.0, max(.01, queue_end-time.time())))[idx]
                    if fresh["pids"] or fresh["memory"] > 64 or fresh["util"]:
                        raise BlockingIOError("GPU became busy")
                    chosen, leases = fresh, trial
                    break
                except BlockingIOError:
                    for handle in trial:
                        handle.close()
            if chosen is None:
                event("waiting", candidates=spec.get("candidate_gpus", [4]))
                time.sleep(min(10, max(0, queue_end-time.time())))
        if chosen is None:
            raise TimeoutError("No idle leased GPU before queue deadline")
        verify()
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=chosen["uuid"], PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
            WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(ROOT / "cache/huggingface"), MPLCONFIGDIR=str(ROOT / "cache/matplotlib"),
            TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
            TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/inductor"))
        for key in ("PYTHONPATH",):
            if key in spec.get("env", {}):
                env[key] = spec["env"][key]
        started = time.monotonic()
        event("acquired", gpu=chosen, spec_sha256=spec_sha)
        with (out / "worker.log").open("w", buffering=1) as log:
            child = subprocess.Popen(spec["command"], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            (out / "pid.json").write_text(json.dumps({"pid": child.pid,
                "gpu": chosen["index"], "gpu_uuid": chosen["uuid"], "utc": utc()}))
            event("started", pid=child.pid, command=spec["command"])
            while child.poll() is None:
                remaining = min(deadline-time.time(), spec["timeout_seconds"]-(time.monotonic()-started))
                if remaining <= 0:
                    raise TimeoutError("Owned job reached its time limit")
                current = helper.state(timeout=min(5, remaining))[chosen["index"]]
                for pid in current["pids"]:
                    with contextlib.suppress(ProcessLookupError):
                        if os.getpgid(pid) != child.pid:
                            raise RuntimeError(f"Yielding occupied GPU to external process {pid}")
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=min(15, remaining))
            verify()
            result.update(status="completed" if child.returncode == 0 else "failed",
                exit_code=child.returncode)
    except BaseException as exc:
        result.update(error=type(exc).__name__, message=str(exc))
        event("failed", **result)
        raise
    finally:
        helper.stop(child, grace_seconds=min(10, max(0, deadline-time.time())))
        result.update(utc=utc(), elapsed_seconds=time.monotonic()-started if started else 0,
            gpu=chosen["index"] if chosen else None, cloud_spend_usd=0, spec_sha256=spec_sha)
        (out / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
        for handle in leases:
            handle.close()
        event("released", **result)
        events.close()
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
