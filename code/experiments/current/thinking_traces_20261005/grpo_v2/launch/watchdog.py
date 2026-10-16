#!/usr/bin/env python3
"""Watchdog for long jobs: STALL / EXIT / OVERRUN alerts and periodic STATUS lines (stdout).

    python3 watchdog.py NAME:PID:GPU:LOG:EXPECTED_H [...]

Every 5 min, per job: CPU seconds of the process and its descendants, the GPU's utilisation,
and the log's size.
- STALL: all three flat for 25 min (no CPU, GPU under 5%, no log growth).
- EXIT: the PID is gone; prints the log's last line.
- OVERRUN: the job has run more than 1.5x EXPECTED_H.
- STATUS: one line per job every hour, every 2 h once a job has run 3 h (Leon's check-in rule).
"""

import os
import subprocess
import sys
import time

POLL, STALL_S, OVERRUN = 300, 1500, 1.5


def tree_cpu(pid: int) -> float:
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,times="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    cpu: dict[int, int] = {}
    for line in out.splitlines():
        p, pp, t = line.split()
        kids.setdefault(int(pp), []).append(int(p))
        cpu[int(p)] = int(t)
    tot, stack = 0, [pid]
    while stack:
        q = stack.pop()
        tot += cpu.get(q, 0)
        stack += kids.get(q, [])
    return float(tot)


def gpu_util(g: int) -> int:
    q = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", str(g)]
    try:
        return int(subprocess.run(q, capture_output=True, text=True).stdout.strip() or 0)
    except ValueError:
        return -1


def last_line(path: str) -> str:
    try:
        with open(path, "rb") as f:
            lines = [x for x in f.read()[-4000:].decode(errors="replace").splitlines() if x.strip()]
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M', time.gmtime())}Z {msg}", flush=True)


def main() -> None:
    jobs = []
    for spec in sys.argv[1:]:
        name, pid, gpu, log, exp = spec.split(":")
        jobs.append(
            dict(name=name, pid=int(pid), gpu=int(gpu), log=log, exp=float(exp) * 3600,
                 t0=time.time(), hist=[], alive=True, stalled=False, over=False, next_status=0.0)
        )
    say("watching " + ", ".join(j["name"] for j in jobs))
    while any(j["alive"] for j in jobs):
        now = time.time()
        for j in jobs:
            if not j["alive"]:
                continue
            if not os.path.exists(f"/proc/{j['pid']}"):
                j["alive"] = False
                say(f"EXIT {j['name']} after {(now - j['t0']) / 3600:.1f}h: {last_line(j['log'])}")
                continue
            size = os.path.getsize(j["log"]) if os.path.exists(j["log"]) else 0
            j["hist"].append((now, tree_cpu(j["pid"]), gpu_util(j["gpu"]), size))
            j["hist"] = [h for h in j["hist"] if now - h[0] <= STALL_S + POLL]
            h0, h1 = j["hist"][0], j["hist"][-1]
            flat = (
                h1[0] - h0[0] >= STALL_S
                and h1[1] - h0[1] < 30
                and all(h[2] < 5 for h in j["hist"])
                and h1[3] == h0[3]
            )
            if flat and not j["stalled"]:
                say(f"STALL {j['name']}: no CPU, GPU or log progress for {STALL_S // 60} min")
            j["stalled"] = flat
            ran = now - j["t0"]
            if ran > OVERRUN * j["exp"] and not j["over"]:
                j["over"] = True
                say(f"OVERRUN {j['name']}: {ran / 3600:.1f}h > {OVERRUN}x {j['exp'] / 3600:.1f}h")
            if now >= j["next_status"]:
                if j["next_status"]:
                    say(f"STATUS {j['name']} {ran / 3600:.1f}h gpu {h1[2]}%: {last_line(j['log'])}")
                j["next_status"] = now + (3600 if ran < 3 * 3600 else 7200)
        time.sleep(POLL)
    say("all jobs ended")


if __name__ == "__main__":
    main()
