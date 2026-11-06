#!/usr/bin/env python3
"""Rent RunPod A40s for the looped-transformer spike, one pod per arm, and pull the results back.

  python launch.py up NAME                     # rent, wait for ssh, arm the self-destruct watchdog
  python launch.py run NAME ARM                # upload code, run spike_remote.sh, pull the output
  python launch.py more NAME ARM                # 8 more RL rounds on ARM's finished run, then sweep
  python launch.py sweep NAME ARM               # only the post-RL loop sweep of ARM's finished run
  python launch.py sync NAME                    # re-upload loop_sweep.py / more_rl.py mid-run
  python launch.py down NAME                   # terminate

ARM is `baseline` (run 198 seed 0 from Hugging Face, sweep only) or a comma-separated list of
training loop counts for pretrain_looped.py, e.g. `1,2,3,4` (random per batch) or `2` (fixed).
Output lands in artifacts/looped_spike_20261006/<arm>/. Pod plumbing is imported from
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
AUTO = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, AUTO)
import pod  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ART = os.path.join(REPO, "artifacts", "looped_spike_20261006")
REMOTE = pod.REMOTE  # /workspace/autoresearch, where the frozen remote.sh expects to run
MAX_SECS = 4 * 3600


def up(key, state):
    if state.exists():
        raise SystemExit(f"{state} exists; `launch.py down` first")
    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while True:
        p = pod.create(key, "looped-spike", pubkey, "SECURE") or pod.create(
            key, "looped-spike", pubkey, "COMMUNITY"
        )
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


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check).returncode


def upload(st, files="spike_remote.sh loop_sweep.py more_rl.py"):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    sh(f'{ssh} -n "mkdir -p {REMOTE}/out"')
    for src, names in ((AUTO, "harness.py remote.sh textbook72"), (HERE, files)):
        sh(f'tar czf - -C {shlex.quote(src)} {names} | {ssh} "tar xzf - -C {REMOTE}"')


def run(st, arm, mode):
    """`mode` run: the arm's first job. more / sweep: follow-ups on that arm's finished run."""
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    tag = "baseline" if arm == "baseline" else "loops" + arm.replace(",", "")
    out = f"{REMOTE}/out/{tag}"
    upload(st)
    # an arm trained beyond 4 loops is swept further out; beyond 8, RL fine-tuning at those
    # depths also needs more than the harness's 2400 s
    top = 0 if arm == "baseline" else max(map(int, arm.split(",")))
    env = "SWEEP_LOOPS='1 2 4 6 8 12 16' " if top > 4 else ""
    if top > 8:
        env = "ND_TOTAL_SECS=3600 SWEEP_LOOPS='1 2 4 8 16 24' "
    if mode == "more":
        job = f"bash {REMOTE}/spike_remote.sh {out} more"
    elif mode == "sweep":
        job = f"SWEEP_STAGES=final {env}bash {REMOTE}/spike_remote.sh {out} sweep"
    elif mode == "bar":  # last night's checkpoint on the autoresearch_loops metric: whole dev pool
        sub = os.environ.get("BAR_SUBDIR", "")  # "more_rl" scores the 12-round checkpoint
        src = f"{ART}/{tag}/{sub}" if sub else f"{ART}/{tag}"
        tag = f"bar_{tag}_rl12" if sub else f"bar_{tag}"  # own directory: the pull must not
        out = f"{REMOTE}/out/{tag}"  # overwrite the run's files
        sh(f'{ssh} -n "mkdir -p {out}"')
        sh(f'tar czf - -C {shlex.quote(src)} source final.pt | {ssh} "tar xzf - -C {out}"')
        job = (
            "SWEEP_STAGES=final SWEEP_LOOPS='2 8' SWEEP_ARGS='--pools dev' "
            f"bash {REMOTE}/spike_remote.sh {out} bar"
        )
    else:
        sh(f'{ssh} "cat > {REMOTE}/pretrain.py" < {shlex.quote(HERE)}/pretrain_looped.py')
        job = f"{env}ND_TRAIN_LOOPS={arm} bash {REMOTE}/spike_remote.sh {out} train --seed 0"
    rc = pod.run_detached(st, job, f"{REMOTE}/job_{tag}_{mode}")
    for _ in range(10):  # the pull can hit the same network drops as the stream
        pull = f"cd {REMOTE}/out && tar czf - {tag} $(ls {tag}.log 2>/dev/null)"
        if sh(f"{ssh} -n {shlex.quote(pull)} | tar xzf - -C {shlex.quote(ART)}", check=False) == 0:
            break
        time.sleep(30)
    print(f"output in {ART}/{tag}; remote rc {rc}", flush=True)


def main():
    cmd, name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    state = Path(ART, f"pod_{name}.json")
    if cmd == "up":
        up(key, state)
    elif cmd in ("run", "more", "sweep", "bar"):
        run(json.loads(state.read_text()), rest[0], cmd)
    elif cmd == "sync":  # refresh the eval scripts on a pod whose job is still training
        upload(json.loads(state.read_text()), "loop_sweep.py more_rl.py")
    elif cmd == "down":
        st = json.loads(state.read_text())
        pod.terminate(key, st["id"])
        state.unlink()
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
