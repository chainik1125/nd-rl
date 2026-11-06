#!/usr/bin/env python3
"""One A40: upload the long_rl files + checkpoints, run remote.sh (passk.py per checkpoint), pull results, terminate.

  python passk/launch.py <ckpt-dir>        # every *.pt in it is scored; results land in passk/results/
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import launch  # noqa: E402  (up(), FILES, and ../autoresearch/pod.py as launch.pod)

pod = launch.pod
REMOTE = "/workspace/long_rl"


def main():
    ckpts = os.path.abspath(sys.argv[1])
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    state = Path(HERE, "pod.json")
    st = launch.up(key, "longrl-passk", state)
    try:
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f'{ssh} -n "mkdir -p {REMOTE}/passk/textbook72 {REMOTE}/ckpt"')
        sh(f"tar czf - -C {os.path.join(HERE, '..')} {launch.FILES} passk/passk.py passk/remote.sh | {ssh} 'tar xzf - -C {REMOTE}'")
        sh(f"tar czf - -C {os.path.join(HERE, '..', '..', 'autoresearch', 'textbook72')} . | {ssh} 'tar xzf - -C {REMOTE}/passk/textbook72'")
        sh(f"tar cf - -C {ckpts} . | {ssh} 'tar xf - -C {REMOTE}/ckpt'")
        rc = pod.run_detached(st, f"bash {REMOTE}/passk/remote.sh", f"{REMOTE}/job")
        sh(f"{ssh} -n 'cd {REMOTE} && tar czf - results job.stream' | tar xzf - -C {HERE}")
        print(f"remote rc {rc}", flush=True)
    finally:
        pod.terminate(key, st["id"])
        state.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})", flush=True)


if __name__ == "__main__":
    main()
