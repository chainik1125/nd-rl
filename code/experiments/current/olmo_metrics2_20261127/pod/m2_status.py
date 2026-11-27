#!/usr/bin/env python3
"""Status of m2 jobs (stdlib only), POD side:  python3 m2_status.py [--root /workspace/m2] [job ...]
One line per job (key=value, parsed by m2_watch.sh):
  <job> state=<PENDING|RUNNING|STALE|DONE|FAILED> rc=<n|-> hb=<age>s started=<epoch|na> progress=<rows>[/<expected>]
        gpu=<MiB|na> disk=<GB free> newest=<path|-> newest_size=<bytes> newest_age=<s>
STALE = state/RUNNING exists but the recorded runner process is gone.  progress = complete lines of the newest
samples.jsonl.gz under the run directory (state/expected, if present, gives the expected count).
"""

import argparse
import contextlib
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path


def count_gz_rows(p):
    n = 0
    try:
        with gzip.open(p, "rb") as f:
            for _ in f:
                n += 1
    except (EOFError, OSError, zlib.error):
        pass
    return n


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def gpu_used():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.split()
        return "+".join(out) or "na"
    except (OSError, subprocess.TimeoutExpired):
        return "na"


def newest_file(run):
    best = None
    st = run / "state"
    for root, _, files in os.walk(run):
        if Path(root) == st:
            continue
        for f in files:
            p = Path(root) / f
            with contextlib.suppress(OSError):
                s = p.stat()
                if best is None or s.st_mtime > best[0]:
                    best = (s.st_mtime, s.st_size, p)
    return best


def newest_samples(run):
    best = None
    for p in run.rglob("samples.jsonl.gz"):
        with contextlib.suppress(OSError):
            m = p.stat().st_mtime
            if best is None or m > best[0]:
                best = (m, p)
    return best[1] if best else None


def job_line(run, now, disk, gpu):
    st = run / "state"
    pid, started = 0, "na"
    if (st / "pid").exists():
        f = (st / "pid").read_text().split()
        with contextlib.suppress(ValueError, IndexError):
            pid = int(f[0])
    rc = "-"
    if (st / "DONE").exists():
        state = "DONE"
    elif (st / "FAILED").exists():
        state = "FAILED"
        m = re.search(r"rc=(\d+)", (st / "FAILED").read_text())
        rc = m[1] if m else "?"
    elif (st / "RUNNING").exists():
        state = "RUNNING" if pid and pid_alive(pid) else "STALE"
    else:
        state = "PENDING"
    if (st / "DONE").exists():
        rc = "0"
    hb = "na"
    if (st / "heartbeat").exists():
        m = re.search(r"ts=(\d+)", (st / "heartbeat").read_text())
        if m:
            hb = str(int(now - int(m[1])))
    if (st / "RUNNING").exists():
        with contextlib.suppress(OSError, ValueError):
            started = str(int((st / "RUNNING").stat().st_mtime))
    elif (st / "pid").exists():
        started = str(int((st / "pid").stat().st_mtime))
    sp = newest_samples(run)
    prog = "-"
    if sp:
        prog = str(count_gz_rows(sp))
        if (st / "expected").exists():
            prog += "/" + (st / "expected").read_text().strip()
    nf = newest_file(run)
    nf_path = str(nf[2].relative_to(run)) if nf else "-"
    nf_size = nf[1] if nf else 0
    nf_age = int(now - nf[0]) if nf else -1
    return (
        f"{run.name} state={state} rc={rc} hb={hb}s started={started} progress={prog} gpu={gpu} disk={disk} "
        f"newest={nf_path} newest_size={nf_size} newest_age={nf_age}"
    )


def chain_line(cdir, now):
    st = cdir / "state"
    pid = 0
    if (st / "pid").exists():
        with contextlib.suppress(ValueError, IndexError):
            pid = int((st / "pid").read_text().split()[0])
    if (st / "DONE").exists():
        state = "DONE"
    elif (st / "FAILED").exists():
        state = "FAILED"
    elif (st / "RUNNING").exists():
        state = "RUNNING" if pid and pid_alive(pid) else "STALE"
    else:
        state = "PENDING"
    total = 0
    with contextlib.suppress(OSError):
        total = sum(
            1
            for ln in (cdir / "chain.txt").read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
    rcs = []
    with contextlib.suppress(OSError):
        rcs = [ln.split() for ln in (st / "rcs").read_text().splitlines() if ln.strip()]
    cur = (st / "current").read_text().strip() if (st / "current").exists() else "-"
    stopped = (st / "FAILED").read_text().strip().replace(" ", "_") if state == "FAILED" else "-"
    return (
        f"chain:{cdir.name} state={state} rc={'0' if state == 'DONE' else '-'} hb=0s started=na "
        f"progress={len(rcs)}/{total} current={cur} rcs={','.join(f'{a}:{b}' for a, b, *_ in rcs) or '-'} "
        f"stopped={stopped} gpu=na disk=na newest=- newest_size=0 newest_age=0"
    )


def status_lines(root, jobs=None, now=None):
    now = time.time() if now is None else now
    runs = Path(root) / "runs"
    try:
        disk = f"{shutil.disk_usage(runs if runs.exists() else root).free / 1e9:.0f}"
    except OSError:
        disk = "na"
    gpu = gpu_used()
    names = jobs or (
        sorted(p.name for p in runs.iterdir() if (p / "state").is_dir()) if runs.exists() else []
    )
    out = []
    for j in names:
        if j.startswith("chain:"):
            c = runs / "_chains" / j[6:]
            out.append(chain_line(c, now) if (c / "state").is_dir() else f"{j} state=ABSENT")
        else:
            out.append(
                job_line(runs / j, now, disk, gpu) if (runs / j).exists() else f"{j} state=ABSENT"
            )
    if not jobs and (runs / "_chains").is_dir():
        out += [
            chain_line(c, now)
            for c in sorted((runs / "_chains").iterdir())
            if (c / "state").is_dir()
        ]
    return out


def parse_line(ln):
    """{'job':..., key: value} from one status line."""
    parts = ln.split()
    d = {"job": parts[0]}
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--root", default="/workspace/m2")
    a = ap.parse_args()
    print("\n".join(status_lines(a.root, a.jobs or None)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
