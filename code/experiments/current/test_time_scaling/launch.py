#!/usr/bin/env python3
"""Run a job list of ttscale.py invocations on one rented A40: rent, upload, run, pull, terminate.

  source .envrc.local && code/.venv/bin/python \
      code/experiments/current/test_time_scaling/launch.py <name> <jobs.txt>

Each line of jobs.txt is `<out-name> <ttscale.py args...>` or `<out-name> <script.py> <args...>`
(e.g. value.py); the jobs run one after another and
write artifacts/test_time_scaling/<name>/<out-name>.json (+ .log). Reuses the autoresearch pod.py
(rental, watchdog, detached run) with its upload replaced, as the combined pipeline's launch.py
does. ND_POD_MAX_SECS (pod.py's self-destruct) defaults to 4 h here. If the pod's state file
already exists (a launcher killed mid-run), it re-uploads and reruns the jobs on that pod.
"""

import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "autoresearch")
COMBINED = os.path.join(HERE, "..", "combined_pipeline_20260928")
sys.path.insert(0, AR)
import pod  # noqa: E402

LOCAL = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "test_time_scaling")

RUN_JOBS = """set -uo pipefail
export ND_FORK=/workspace/nd-takehome LEAN_GATE_WORKERS=$(nproc)
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
cd /workspace/autoresearch
mkdir -p "$2"
while read -r name args; do
  [ -z "$name" ] && continue
  echo "== $name $args"
  case "$args" in *.py\\ *) ;; *) args="ttscale.py $args" ;; esac  # a job may name its script
  LEAN_GATE_LOG="$2/$name.lean_gate.jsonl" $PY $args --out "$2/$name.json" \\
    > "$2/$name.log" 2>&1 < /dev/null || echo "FAILED $name"
  grep -v '^\\[lean_gate\\]' "$2/$name.log" | tail -2
done < "$1"
"""


def stage(name, jobs):
    d = os.path.join(LOCAL, "stage", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for f in ("remote.sh", "harness.py"):
        shutil.copy2(os.path.join(AR, f), d)
    for f in ("fast_sample.py", "guided.py", "line_checker.py"):
        shutil.copy2(os.path.join(COMBINED, f), d)
    for f in ("engine.py", "ttscale.py", "value.py"):
        shutil.copy2(os.path.join(HERE, f), d)
    shutil.copy2(jobs, os.path.join(d, "jobs.txt"))
    with open(os.path.join(d, "run_jobs.sh"), "w") as f:
        f.write(RUN_JOBS)
    return d


def main():
    name, jobs = sys.argv[1], sys.argv[2]
    os.environ.setdefault("ND_POD_MAX_SECS", str(4 * 3600))
    d = stage(name, jobs)

    def upload(st, _pretrain_file):
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        subprocess.run(
            f"tar czf - -C {shlex.quote(d)} . "
            f'| {ssh} "mkdir -p {pod.REMOTE} && tar xzf - -C {pod.REMOTE}"',
            shell=True,
            check=True,
        )

    pod.upload = upload
    # ~/.ssh/config sends Host * through the 1Password agent, which blocks every ssh while it is
    # locked (2026-09-29: all six result pulls hung 40 min); pods take pod.py's key file directly.
    pod_ssh_args = pod.ssh_args
    key = os.path.expanduser("~/.ssh/id_ed25519")
    pod.ssh_args = lambda st: ["-o", "IdentityAgent=none", "-i", key, *pod_ssh_args(st)]
    pod_name = f"tts-{name}"
    state = os.path.join(pod.LOCAL_ART, f"pod_{pod_name}.json")
    out = f"{pod.REMOTE}/artifacts/{name}"
    try:
        if os.path.exists(state):  # resume on the pod a crashed launcher left up
            pod.STATE = state
            st = pod.load_state()
            upload(st, None)
        else:
            sys.argv = ["pod.py", "up", "--pod", pod_name]
            pod.main()
            st = pod.load_state()
        rc = pod.run_detached(st, f"bash {pod.REMOTE}/run_jobs.sh {pod.REMOTE}/jobs.txt {out}", out)
        print(f"jobs exit {rc}", flush=True)
        os.makedirs(LOCAL, exist_ok=True)
        ssh = "ssh -n " + " ".join(pod.ssh_args(st))
        for _ in range(10):  # the pull can hit this Mac's network drops
            pulled = subprocess.run(
                f'{ssh} "cd {pod.REMOTE}/artifacts && tar czf - {name}" '
                f"| tar xzf - -C {shlex.quote(LOCAL)}",
                shell=True,
                check=False,
            )
            if pulled.returncode == 0:
                break
    finally:
        if os.path.exists(state):
            sys.argv = ["pod.py", "down", "--pod", pod_name]
            pod.main()


if __name__ == "__main__":
    main()
