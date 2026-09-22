#!/usr/bin/env python3
# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""Concurrency benchmark: one pod per (n, mps) cell, N harness processes at once.

python3 bench.py
"""

import os
import shlex
import subprocess
import sys
import threading
import time

import pod

CELLS = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)]


def one(n, mps):
    name = f"bench_n{n}_mps{mps}"
    with open(os.path.join(pod.LOCAL_ART, f"{name}.driver.log"), "w") as log:

        def run(*a):
            return subprocess.run(
                [sys.executable, pod.__file__, *a, "--pod", name],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        t0 = time.time()
        try:
            if run("up").returncode != 0:
                print(f"{name}: pod up FAILED", flush=True)
                return
            with open(os.path.join(pod.LOCAL_ART, f"pod_{name}.json")) as f:
                st = pod.json.load(f)
            subprocess.run(
                f"scp -q -P {st['port']} -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "
                f"{shlex.quote(os.path.join(pod.HERE, 'bench.sh'))} "
                f"root@{st['ip']}:{pod.REMOTE}/bench.sh",
                shell=True,
                check=True,
            )
            out = f"{pod.REMOTE}/artifacts/{name}"
            pod.ssh_run(
                st,
                f"mkdir -p {pod.REMOTE}/artifacts && bash {pod.REMOTE}/bench.sh {out} {n} {mps}",
                check=False,
            )
            ssh = "ssh " + " ".join(pod.ssh_args(st))
            subprocess.run(
                f'{ssh} "cd {pod.REMOTE}/artifacts && tar czf - --exclude=*.pt {name}" '
                f"| tar xzf - -C {shlex.quote(pod.LOCAL_ART)}",
                shell=True,
            )
            print(f"{name}: done after {(time.time() - t0) / 60:.0f} min", flush=True)
        finally:
            run("down")


if __name__ == "__main__":
    ts = [threading.Thread(target=one, args=c) for c in CELLS]
    for t in ts:
        t.start()
        time.sleep(3)
    for t in ts:
        t.join()
    print("BENCH_DONE", flush=True)
