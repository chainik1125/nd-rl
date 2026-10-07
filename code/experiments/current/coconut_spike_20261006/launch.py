#!/usr/bin/env python3
"""Rent one RunPod A40 per Coconut arm, run it, and pull the results back.

  python launch.py go ARM [TAG]  # rent, run coconut_remote.sh for ARM, pull the output, terminate
  python launch.py rerun ARM   # run again on ARM's pod (still up after a failure), then terminate
  python launch.py finish TAG RUN  # on TAG's pod: dev eval + sweep of the local run RUN, which
                                   # timed out after RL (see finish_eval.py); then terminate
  python launch.py down ARM    # terminate ARM's pod by hand (after a failed or interrupted `go`)

ARM is one of pretrain_coconut.py's ND_ARM values. TAG (default ARM) names the pod and the output
directory, artifacts/coconut_spike_20261006/<tag>/ (checkpoints included), so one arm can run
twice. `rerun` and `down` take the TAG. Pod plumbing is imported from
../autoresearch/pod.py. Needs RUNPOD_API_KEY (repo .envrc.local). A failed job leaves its pod up
for inspection; the on-pod watchdog still terminates it after MAX_SECS.
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
AUTO = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, AUTO)
import pod  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ART = os.path.join(REPO, "artifacts", "coconut_spike_20261006")
REMOTE = pod.REMOTE  # /workspace/autoresearch, where the frozen remote.sh expects to run
MAX_SECS = 4 * 3600


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check).returncode


def up(key, state, arm):
    if state.exists():
        raise SystemExit(f"{state} exists; `launch.py down {arm}` first")
    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while True:
        name = f"coconut-{arm}"
        p = pod.create(key, name, pubkey, "SECURE") or pod.create(key, name, pubkey, "COMMUNITY")
        if p is None:
            print("no A40 capacity, retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        st = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
        # saved before waiting, so `down` works if the wait is interrupted
        state.write_text(json.dumps(st))
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
    watchdog = Path(AUTO, "watchdog.sh").read_bytes()
    pod.ssh_run(
        st,
        f"echo {base64.b64encode(watchdog).decode()} | base64 -d > /tmp/watchdog.sh && "
        f"{{ nohup setsid bash /tmp/watchdog.sh {MAX_SECS} > /tmp/wd.log 2>&1 < /dev/null & }}",
    )
    print(f"up after {time.time() - st['t0']:.0f}s: ssh -p {st['port']} root@{st['ip']}")
    return st


def run(st, arm, tag, finish=False):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    sh(f'{ssh} -n "mkdir -p {REMOTE}/out"')
    sources = (
        (AUTO, "harness.py remote.sh textbook72"),
        (HERE, "coconut_remote.sh thought_sweep.py finish_eval.py"),
    )
    if finish:  # the timed-out run's last RL checkpoint, source and solved-target list
        last = f"{tag}/rl_round_04.pt {tag}/source {tag}/found_targets.jsonl"
        sh(f'tar czf - -C {shlex.quote(ART)} {last} | {ssh} "tar xzf - -C {REMOTE}/out"')
    for src, names in sources:
        sh(f'tar czf - -C {shlex.quote(src)} {names} | {ssh} "tar xzf - -C {REMOTE}"')
    sh(f'{ssh} "cat > {REMOTE}/pretrain.py" < {shlex.quote(HERE)}/pretrain_coconut.py')
    env = "ND_COMPILE=0 " if os.environ.get("ND_COMPILE") == "0" else ""  # see pretrain_coconut.py
    env += "FINISH=1 " if finish else ""
    job = f"{env}ND_ARM={arm} bash {REMOTE}/coconut_remote.sh {REMOTE}/out/{tag} --seed 0"
    rc = pod.run_detached(st, job, f"{REMOTE}/job_{tag}")
    for _ in range(10):  # the pull can hit the same network drops as the stream
        pull = f"cd {REMOTE}/out && tar czf - {tag} $(ls {tag}.log 2>/dev/null)"
        if sh(f"{ssh} -n {shlex.quote(pull)} | tar xzf - -C {shlex.quote(ART)}", check=False) == 0:
            break
        time.sleep(30)
    else:
        rc = rc or 1  # nothing pulled: keep the pod
    print(f"output in {ART}/{tag}; remote rc {rc}", flush=True)
    return rc


def down(key, state):
    st = json.loads(state.read_text())
    pod.terminate(key, st["id"])
    state.unlink()
    mins = (time.time() - st["t0"]) / 60
    print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})")


def main():
    cmd, arm = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else arm
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    state = Path(ART, f"pod_{tag}.json")
    if cmd == "go":
        if run(up(key, state, tag), arm, tag) == 0:
            down(key, state)
    elif cmd == "rerun":  # the arm is the part of the tag before any "-"
        if run(json.loads(state.read_text()), arm.split("-")[0], arm) == 0:
            down(key, state)
    elif cmd == "finish":
        state, local = Path(ART, f"pod_{arm}.json"), sys.argv[3]
        if run(json.loads(state.read_text()), local.split("-")[0], local, finish=True) == 0:
            down(key, state)
    elif cmd == "down":
        down(key, state)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
