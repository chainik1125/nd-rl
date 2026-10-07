#!/usr/bin/env python3
"""Rent one RunPod A40 for the temperature sweep, run it, pull the pools back.

  python launch.py up                 # rent, wait for ssh, arm the self-destruct watchdog
  python launch.py run [sweep args]   # upload code + checkpoint, run remote.sh, pull the output
  python launch.py down               # terminate

Pod plumbing is imported from ../autoresearch/pod.py. Needs RUNPOD_API_KEY (repo .envrc.local).
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
ART = os.path.join(REPO, "artifacts", "temp_sweep_20261006")
STATE = Path(ART, "pod.json")
# from hf://buckets/dan-pandori/nd-rl/rl-continue/ckpts/rc/ladder/
CKPT = os.path.join(ART, "la_T1_best12_s0_r16.pt")
# nd-rl dmitry/main: the selection sprint's collector + scorer
DMITRY = "af00a5842ead361da81c9b610ff26eaac0a42bfb"
DMITRY_CODE = "code/experiments/current/proof_selection_20261006/code"
REMOTE = "/workspace/tempsweep"
MAX_SECS = 6 * 3600


def up(key):
    if STATE.exists():
        raise SystemExit(f"{STATE} exists; `launch.py down` first")
    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while True:
        p = pod.create(key, "tempsweep", pubkey, "SECURE") or pod.create(
            key, "tempsweep", pubkey, "COMMUNITY"
        )
        if p is None:
            print("no A40 capacity, retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        st = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
        # saved before waiting, so `down` works if the wait is interrupted
        STATE.write_text(json.dumps(st))
        print(f"pod {p['id']} at ${p['costPerHr']}/hr", flush=True)
        if addr := pod.wait_ssh(key, p["id"]):
            break
        pod.terminate(key, p["id"])
        STATE.unlink()
    st["ip"], st["port"] = addr
    STATE.write_text(json.dumps(st))
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
    print(f"up after {time.time() - st['t0']:.0f}s: ssh -p {st['port']} root@{st['ip']}")


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check).returncode


def run(st, args):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    scp_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    repo, here, art = shlex.quote(REPO), shlex.quote(HERE), shlex.quote(ART)
    sh(f'{ssh} -n "mkdir -p {REMOTE}/dm"')
    sh(
        f"git -C {repo} archive {DMITRY} {DMITRY_CODE} "
        f'| {ssh} "tar xf - --strip-components=5 -C {REMOTE}/dm"'
    )
    sh(f'tar czf - -C {here} remote.sh sweep_remote.py | {ssh} "tar xzf - -C {REMOTE}"')
    sh(
        f'{ssh} -n "test -f {REMOTE}/{os.path.basename(CKPT)}" '
        f"|| scp -q -P {st['port']} {scp_opts} {shlex.quote(CKPT)} root@{st['ip']}:{REMOTE}/"
    )
    job = f"bash {REMOTE}/remote.sh {REMOTE}/out " + " ".join(map(shlex.quote, args))
    rc = pod.run_detached(st, job, f"{REMOTE}/job")
    for _ in range(10):  # the pull can hit the same network drops as the stream
        if sh(f'{ssh} -n "tar czf - -C {REMOTE} out" | tar xzf - -C {art}', check=False) == 0:
            break
        time.sleep(30)
    print(f"output in {ART}/out; remote rc {rc}", flush=True)


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    if cmd == "up":
        up(key)
    elif cmd == "run":
        run(json.loads(STATE.read_text()), rest)
    elif cmd == "down":
        st = json.loads(STATE.read_text())
        pod.terminate(key, st["id"])
        STATE.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
