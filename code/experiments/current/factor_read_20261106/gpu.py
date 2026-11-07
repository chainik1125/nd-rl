"""GPU ownership checks (every launch and, in the supervisor, continuously).

    python3 gpu.py [HOST]      # print each GPU: index, bus id, users of its compute processes

A GPU is free for us only when no other user has a compute process on it (we never share a GPU).
`HOST` is reached with `ssh -i ~/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes
-o ConnectTimeout=10` (serrano is local). Returns None for an unreachable host.
"""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import sys

ME = getpass.getuser()
HOST = socket.gethostname().split(".")[0]
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/id_ynez"), "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]  # fmt: skip
Q = ("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader; echo ===; "
     "nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | "
     "while IFS=, read p b; do echo \"$p $b $(ps -o user:32= -p $p)\"; done")  # fmt: skip


def run(host: str, cmd: str, timeout: int = 40) -> subprocess.CompletedProcess | None:
    local = host in (HOST, "localhost")
    argv = ["bash", "-c", cmd] if local else [*SSH, host, cmd]
    env = {**os.environ, "SSH_AUTH_SOCK": ""}
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None


def gpus(host: str) -> dict[int, set[str]] | None:
    """GPU index -> users with a compute process on it; None if the host does not answer."""
    p = run(host, Q)
    if p is None or p.returncode != 0 or "===" not in p.stdout:
        return None
    head, procs = p.stdout.split("===", 1)
    bus = {}
    for line in head.strip().splitlines():
        i, b = (x.strip() for x in line.split(","))
        bus[b.lower()] = int(i)
    out: dict[int, set[str]] = {i: set() for i in bus.values()}
    for line in procs.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 3 and parts[1].lower() in bus:
            out[bus[parts[1].lower()]].add(parts[2])
    return out


def ours(host: str) -> set[str]:
    """Process owners that are us on `host` (a Runpod pod runs everything as root)."""
    return {ME, "root"} if host.startswith("rp-") else {ME}


def free_for_us(host: str, gpu: int) -> bool:
    g = gpus(host)
    return g is not None and gpu in g and not (g[gpu] - ours(host) - {ME})


if __name__ == "__main__":
    h = sys.argv[1] if len(sys.argv) > 1 else "serrano"
    print(h, gpus(h))
