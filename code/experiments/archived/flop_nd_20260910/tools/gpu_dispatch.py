"""Run one immutable sprint job on an unused GPU, with deadline and ownership checks."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RESERVED = {0, 2}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(timeout=5.0):
    # Bound both queries together so a hung driver cannot defeat the deadline.
    end = time.monotonic() + max(0.01, timeout)
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=max(0.01, end - time.monotonic()),
    )
    by_uuid = {}
    for line in raw.splitlines():
        i, uuid, memory, util = [s.strip() for s in line.split(",")]
        by_uuid[uuid] = dict(index=int(i), uuid=uuid, memory=int(memory), util=int(util), pids=[])
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=max(0.01, end - time.monotonic()),
    )
    for line in raw.splitlines():
        uuid, pid = [s.strip() for s in line.split(",")]
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].append(int(pid))
    return {g["index"]: g for g in by_uuid.values()}


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop(child, grace_seconds=15.0):
    """Stop only this start_new_session child's process group, including orphans.

    A pipeline leader can exit before its training descendants. Its returncode
    therefore cannot establish that the owned process group is empty.
    """
    if child is None:
        return
    pgid = child.pid
    if pgid == os.getpgrp():
        raise RuntimeError("Refusing to signal supervisor's own process group")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    end = time.monotonic() + max(0.0, grace_seconds)
    while group_alive(pgid) and time.monotonic() < end:
        child.poll()  # Reap the leader, while still checking for descendants.
        time.sleep(min(0.1, max(0.0, end - time.monotonic())))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    # Reap the direct child with a bounded wait; descendants are not our children.
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    job_path = Path(a.job).resolve()
    spec = json.loads(job_path.read_text())
    root = Path(spec["root"]).resolve()
    assert job_path.is_relative_to(root)
    job_sha = sha(job_path)
    deadline = datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00")).timestamp()
    out = root / "jobs" / spec["name"]
    out.mkdir(parents=True, exist_ok=False)
    leases = root / "leases"
    leases.mkdir(exist_ok=True)
    events = out / "events.jsonl"

    def event(kind, **kw):
        with events.open("a") as f:
            f.write(json.dumps(dict(utc=now(), kind=kind, **kw), sort_keys=True) + "\n")

    def verify_inputs():
        assert sha(job_path) == job_sha, "Job declaration changed"
        for path, digest in spec.get("input_sha256", {}).items():
            assert sha(path) == digest, f"Input changed: {path}"

    verify_inputs()
    candidates = spec.get("candidate_gpus", [3, 4, 1, 5, 6, 7])
    assert not (set(candidates) & RESERVED)
    child = None
    lease = None
    selected = None
    idle_since = {}
    last_event = 0
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: (_ for _ in ()).throw(SystemExit(128 + s)))
    try:
        while time.time() < deadline:
            current = state(timeout=min(5.0, max(0.01, deadline - time.time())))
            for i in candidates:
                g = current[i]
                if g["pids"] or g["memory"] > 64 or g["util"]:
                    idle_since.pop(i, None)
                    continue
                idle_since.setdefault(i, time.monotonic())
                if time.monotonic() - idle_since[i] < 15:
                    continue
                lock = (leases / f"gpu_{i}.lock").open("a+")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                fresh = state(timeout=min(5.0, max(0.01, deadline - time.time())))[i]
                if fresh["pids"] or fresh["memory"] > 64 or fresh["util"]:
                    lock.close()
                    continue
                lease, selected = lock, i
                event("acquired", gpu=i, snapshot=current, reserved=sorted(RESERVED))
                break
            if selected is not None:
                break
            if time.monotonic() - last_event > 60:
                event("waiting", snapshot=current)
                last_event = time.monotonic()
            time.sleep(min(10.0, max(0.0, deadline - time.time())))
        if selected is None:
            raise TimeoutError("No idle GPU before deadline")
        verify_inputs()
        env = {
            **os.environ,
            **spec.get("env", {}),
            "CUDA_VISIBLE_DEVICES": str(selected),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
        limit = min(spec["timeout_seconds"], deadline - time.time())
        assert limit > 5
        with (out / "stdout.log").open("w") as log:
            child = subprocess.Popen(
                spec["command"],
                cwd=spec["cwd"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            event(
                "started", pid=child.pid, gpu=selected, command=spec["command"], job_sha256=job_sha
            )
            (out / "pid.json").write_text(json.dumps(dict(pid=child.pid, gpu=selected)))
            started = time.monotonic()

            def remaining():
                return min(limit - (time.monotonic() - started), deadline - time.time())

            while child.poll() is None:
                if remaining() <= 0:
                    event("deadline", limit=limit)
                    stop(child, grace_seconds=0)
                    break
                current = state(timeout=min(5.0, max(0.01, remaining())))
                other = []
                for pid in current[selected]["pids"]:
                    try:
                        if os.getpgid(pid) != child.pid:
                            other.append(pid)
                    except ProcessLookupError:
                        pass
                if other:
                    event("yield_external", external_pids=other)
                    stop(child, grace_seconds=min(15.0, max(0.0, remaining())))
                    break
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=min(15.0, max(0.0, remaining())))
            result = dict(
                returncode=child.returncode,
                seconds=time.monotonic() - started,
                completed_utc=now(),
                gpu=selected,
                job_sha256=job_sha,
            )
            (out / "result.json").write_text(json.dumps(result, indent=2))
            event("finished", **result)
    finally:
        stop(child, grace_seconds=min(15.0, max(0.0, deadline - time.time())))
        if lease is not None:
            lease.close()
        event("released", gpu=selected)


if __name__ == "__main__":
    main()
