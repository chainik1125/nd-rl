#!/usr/bin/env python3
"""One RunPod A40 per (arm, seed): rent, run the harness, pull the output, push checkpoints to HF, terminate.

  python launch.py go ARCH RL SEED [harness args]    # ARCH untied|looped, RL ei|grpo
  python launch.py down ARCH RL SEED                 # terminate a pod `go` left behind

Output lands in artifacts/long_rl_20261014/<RUN>/<arch>-<rl>_s<seed>/; checkpoints go to the HF repo at
runs/long-rl-20261014-<RUN>_<arch>-<rl>_s<seed>/. Pod plumbing is ../autoresearch/pod.py. Needs RUNPOD_API_KEY.
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
# v1 (09:05): no bridge, nothing solved. v2 (10:17): all six arms. v3 (13:00): GRPO only, with the PPO ratio, Muon
# and loss at the sampling temperature, after v2's GRPO collapsed. v4 (14:05): untied EI and v3's GRPO x 2 seeds with
# the per-round KL probes and checkpoints every 5 rounds. v5 (15:40): v2's untied EI seeds resumed (--resume) for
# 4 more hours of RL on the same pools.
RUN = "v5"
ART = os.path.join(REPO, "artifacts", "long_rl_20261014", RUN)
REMOTE = "/workspace/long_rl"
# The harness's own limit is 6.5 h unless ND_TOTAL_SECS (passed through to it) says more; the watchdog adds 0.75 h
# for setup and the pull.
TOTAL_SECS = int(os.environ.get("ND_TOTAL_SECS", 6.5 * 3600))
MAX_SECS = TOTAL_SECS + int(0.75 * 3600)
FILES = "remote.sh harness.py pretrain.py overlay/lean_tok.py overlay/lean_gate.py data/pretrain.jsonl.gz data/rl.jsonl data/dev.jsonl data/holdout.jsonl"


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


def run(st, arch, rl, name, args):
    ssh = "ssh " + " ".join(pod.ssh_args(st))
    sh(f'{ssh} -n "mkdir -p {REMOTE}"')
    resume = args[args.index("--resume") + 1] if "--resume" in args else ""  # a finished run's files, under HERE
    sh(f'tar czf - -C {shlex.quote(HERE)} {FILES} {resume} | {ssh} "tar xzf - -C {REMOTE}"')
    job = (f"ND_ARCH={arch} ND_RL={rl} ND_TOTAL_SECS={TOTAL_SECS} bash {REMOTE}/remote.sh {REMOTE}/{name} "
           + " ".join(map(shlex.quote, args)))
    rc = pod.run_detached(st, job, f"{REMOTE}/job")
    for _ in range(10):  # the pull can hit the same network drops as the stream
        pull = f'{ssh} -n "cd {REMOTE} && tar czf - {name} {name}.log"'
        if sh(f"{pull} | tar xzf - -C {shlex.quote(ART)}", check=False) == 0:
            break
        time.sleep(30)
    return rc


def main():
    cmd, arch, rl, seed, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    os.makedirs(ART, exist_ok=True)
    name = f"{arch}-{rl}_s{seed}"
    state = Path(ART, f"pod_{name}.json")
    if cmd == "go":
        if state.exists():
            raise SystemExit(f"{state} exists; `launch.py down {arch} {rl} {seed}` first")
        st = up(key, f"longrl-{name}", state)
        try:
            rc = run(st, arch, rl, name, ["--seed", seed, *rest])
        finally:
            pod.terminate(key, st["id"])
            state.unlink()
            mins = (time.time() - st["t0"]) / 60
            print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})", flush=True)
        print(f"output in {ART}/{name}; remote rc {rc}", flush=True)
        # checkpoints to HF runs/long-rl-20261014-<RUN>_<name>, deleted locally once HF has them
        hf_name = f"long-rl-20261014-{RUN}_{name}"
        os.replace(os.path.join(ART, name), os.path.join(ART, hf_name))
        pod.LOCAL_ART = ART
        pod.push_checkpoints(hf_name)
    elif cmd == "down":
        st = json.loads(state.read_text())
        pod.terminate(key, st["id"])
        state.unlink()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
