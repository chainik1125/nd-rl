#!/usr/bin/env python3
"""One RunPod A40 per (arm, seed): rent, run the harness, pull the output, terminate.

  python launch.py go ARM SEED [harness args]   # ARM: iid | retry | blank
  python launch.py down ARM SEED                # terminate a pod `go` left behind

Output lands in artifacts/retry_ei_20261008/<arm>_s<seed>/. Pod plumbing is imported from
../autoresearch/pod.py. Needs RUNPOD_API_KEY (repo .envrc.local).
"""

import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "autoresearch"))
import pod  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ART = os.path.join(REPO, "artifacts", "retry_ei_20261008")
REMOTE = "/workspace/retry_ei"
MAX_SECS = 3 * 3600  # watchdog: the harness's own limit is 7200 s
FILES = "remote.sh harness.py pretrain.py episodes.py errors.py lean_prefilter.py"


def up(key, name, state):
    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while True:
        p = pod.create(key, name, pubkey, "SECURE") or pod.create(key, name, pubkey, "COMMUNITY")
        if p is None:
            print("no A40 capacity, retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        st = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
        state.write_text(json.dumps(st))  # before waiting, so `down` works if this is interrupted
        print(f"pod {p['id']} at ${p['costPerHr']}/hr", flush=True)
        if addr := pod.wait_ssh(key, p["id"]):
            break
        pod.terminate(key, p["id"])
        state.unlink()
    st["ip"], st["port"] = addr
    state.write_text(json.dumps(st))
    probe = ["ssh", *pod.ssh_args(st), "true"]
    for _ in range(60):
        if subprocess.run(probe, capture_output=True, stdin=subprocess.DEVNULL).returncode == 0:
            break
        time.sleep(5)
    else:
        raise SystemExit("ssh never accepted the key")
    watchdog = Path(HERE, "..", "autoresearch", "watchdog.sh").read_bytes()
    pod.ssh_run(
        st,
        f"echo {base64.b64encode(watchdog).decode()} | base64 -d > /tmp/watchdog.sh && "
        f"{{ nohup setsid bash /tmp/watchdog.sh {MAX_SECS} > /tmp/wd.log 2>&1 < /dev/null & }}",
    )
    return st


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check).returncode


def run(st, arm, name, args):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    sh(f'{ssh} -n "mkdir -p {REMOTE}"')
    sh(f'tar czf - -C {shlex.quote(HERE)} {FILES} | {ssh} "tar xzf - -C {REMOTE}"')
    job = f"ND_ARM={arm} bash {REMOTE}/remote.sh {REMOTE}/{name} " + " ".join(
        map(shlex.quote, args)
    )
    rc = pod.run_detached(st, job, f"{REMOTE}/job")
    for _ in range(10):  # the pull can hit the same network drops as the stream
        pull = f'{ssh} -n "cd {REMOTE} && tar czf - --exclude=*.pt {name} {name}.log"'
        if sh(f"{pull} | tar xzf - -C {shlex.quote(ART)}", check=False) == 0:
            break
        time.sleep(30)
    return rc


def main():
    cmd, arm, seed, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    name = f"{arm}_s{seed}"
    state = Path(ART, f"pod_{name}.json")
    if cmd == "go":
        if state.exists():
            raise SystemExit(f"{state} exists; `launch.py down {arm} {seed}` first")
        st = up(key, f"retry-{name}", state)
        try:
            rc = run(st, arm, name, ["--seed", seed, *rest])
        finally:
            pod.terminate(key, st["id"])
            state.unlink()
            mins = (time.time() - st["t0"]) / 60
            print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})")
        print(f"output in {ART}/{name}; remote rc {rc}", flush=True)
    elif cmd == "down":
        st = json.loads(state.read_text())
        pod.terminate(key, st["id"])
        state.unlink()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
