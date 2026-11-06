#!/usr/bin/env python3
"""textbook72 at k = 10,240 and T = 1 (Robbie, 2026-10-14): one A40 for one checkpoint; run one process per tag.

  python passk/long_k.py <ckpt-dir> <tag> > passk/long_k_<tag>.log     # scores <ckpt-dir>/<tag>.pt

Each pod runs remote.sh's setup with an empty jobs file, then passk.py in chunks of 1,024. Results land in
passk/results_10k/<tag>.json, kept apart from the k = 256, T = 0.8 results/.
"""

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
K, TEMP, CHUNK = 10240, 1.0, 1024


def run_one(key, ckpts, tag):
    state = Path(HERE, f"long_k_pod_{tag}.json")
    st = launch.up(key, f"longrl-longk-{tag}", state)
    try:
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f'{ssh} -n "mkdir -p {REMOTE}/passk/textbook72 {REMOTE}/ckpt"')
        sh(f"tar czf - -C {os.path.join(HERE, '..')} {launch.FILES} passk/passk.py passk/remote.sh | {ssh} 'tar xzf - -C {REMOTE}'")
        sh(f"tar czf - -C {os.path.join(HERE, '..', '..', 'autoresearch', 'textbook72')} . | {ssh} 'tar xzf - -C {REMOTE}/passk/textbook72'")
        sh(f"cat {ckpts}/{tag}.pt | {ssh} 'cat > {REMOTE}/ckpt/{tag}.pt'")
        job = (f"bash {REMOTE}/passk/remote.sh /dev/null && cd {REMOTE} && export ND_FORK=/workspace/nd-takehome "
               f"ND_ARCH=untied ND_RL=ei LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG={REMOTE}/lean_gate.jsonl "
               f"ND_DATA={REMOTE}/data && PY=$(for p in python python3; do $p -c 'import torch' 2>/dev/null && echo $p && break; done) "
               f"&& $PY passk/passk.py "
               f"ckpt/{tag}.pt --tag {tag} --out results_10k/{tag}.json --sets textbook72 --k {K} --temp {TEMP} "
               f"--chunk {CHUNK}")
        rc = pod.run_detached(st, job, f"{REMOTE}/job")
        sh(f"{ssh} -n 'cd {REMOTE} && tar czf - results_10k' | tar xzf - -C {HERE}")
        print(f"remote rc {rc}", flush=True)
    finally:
        pod.terminate(key, st["id"])
        state.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})", flush=True)


def main():
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    run_one(key, os.path.abspath(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
