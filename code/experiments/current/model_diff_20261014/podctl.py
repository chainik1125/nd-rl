#!/usr/bin/env python3
# ruff: noqa: E501,SIM115
"""One RunPod A40 for the model-diffing jobs (pod plumbing from ../autoresearch/pod.py).

  python podctl.py up [name]          rent, wait for ssh, arm an 11 h self-destruct watchdog -> pods/<name>.json
  python podctl.py push [name]        rsync code, corpus and checkpoints to /workspace/md
  python podctl.py job [name] <tag> <cmd...>   run `cmd` detached in /workspace/md, output -> logs/<tag>.log on the pod
  python podctl.py pull [name]        rsync /workspace/md/runs and logs back to runs/ and logs/
  python podctl.py ssh [name] <cmd>   run a command
  python podctl.py down [name]        terminate
Needs RUNPOD_API_KEY (repo .envrc.local).
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "autoresearch"))
import pod  # noqa: E402

REMOTE = "/workspace/md"
MAX_SECS = 11 * 3600


def key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        for line in open(HERE.parents[3] / ".envrc.local"):
            if line.startswith("export RUNPOD_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
    return k


def state(name):
    return json.loads((HERE / "pods" / f"{name}.json").read_text())


def ssh(st):
    return "ssh " + " ".join(pod.ssh_args(st)[:-1]) + f" {pod.ssh_args(st)[-1]}"


def up(name):
    f = HERE / "pods" / f"{name}.json"
    f.parent.mkdir(exist_ok=True)
    if f.exists():
        raise SystemExit(f"{f} exists")
    pub = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while True:
        p = pod.create(key(), f"md-{name}", pub, "SECURE") or pod.create(
            key(), f"md-{name}", pub, "COMMUNITY"
        )
        if p is None:
            print("no capacity, retry in 30 s", flush=True)
            time.sleep(30)
            continue
        st = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
        f.write_text(json.dumps(st))
        print(f"pod {p['id']} ${p['costPerHr']}/hr", flush=True)
        if addr := pod.wait_ssh(key(), p["id"]):
            break
        pod.terminate(key(), p["id"])
        f.unlink()
    st["ip"], st["port"] = addr
    f.write_text(json.dumps(st))
    for _ in range(60):
        if (
            subprocess.run(
                ["ssh", *pod.ssh_args(st), "true"], capture_output=True, stdin=subprocess.DEVNULL
            ).returncode
            == 0
        ):
            break
        time.sleep(5)
    wd = (HERE.parent / "autoresearch" / "watchdog.sh").read_bytes()
    pod.ssh_run(
        st,
        f"echo {base64.b64encode(wd).decode()} | base64 -d > /tmp/watchdog.sh && "
        f"{{ nohup setsid bash /tmp/watchdog.sh {MAX_SECS} > /tmp/wd.log 2>&1 < /dev/null & }}; "
        f"mkdir -p {REMOTE}/logs {REMOTE}/runs; nvidia-smi --query-gpu=name,memory.total --format=csv",
    )
    print(f"up after {time.time() - st['t0']:.0f}s: ssh -p {st['port']} root@{st['ip']}")


def rsync(st, src, dst):
    e = f"ssh -p {st['port']} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    subprocess.run(["rsync", "-az", "-e", e, *src, dst], check=True)


def push(name):
    st = state(name)
    os.chdir(HERE)
    rsync(
        st,
        ["tiny.py", "cc.py", "analyze_remote.py"]
        if Path("analyze_remote.py").exists()
        else ["tiny.py", "cc.py"],
        f"root@{st['ip']}:{REMOTE}/",
    )
    rsync(st, ["--relative", "out/corpus.pt", "ckpt/runs"], f"root@{st['ip']}:{REMOTE}/")
    for extra in sys.argv[3:]:
        rsync(st, ["--relative", extra], f"root@{st['ip']}:{REMOTE}/")


def job(name, tag, cmd):
    st = state(name)
    pod.ssh_run(
        st,
        f"cd {REMOTE} && {{ nohup setsid bash -c {json.dumps(cmd)} > logs/{tag}.log 2>&1 < /dev/null & }}",
    )


def pull(name):
    st = state(name)
    os.chdir(HERE)
    rsync(st, [f"root@{st['ip']}:{REMOTE}/runs", f"root@{st['ip']}:{REMOTE}/logs"], "./")


if __name__ == "__main__":
    cmd, name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "a")
    if cmd == "up":
        up(name)
    elif cmd == "push":
        push(name)
    elif cmd == "job":
        job(name, sys.argv[3], " ".join(sys.argv[4:]))
    elif cmd == "pull":
        pull(name)
    elif cmd == "ssh":
        sys.exit(pod.ssh_run(state(name), " ".join(sys.argv[3:]), check=False))
    elif cmd == "down":
        st = state(name)
        pod.terminate(key(), st["id"])
        print(f"terminated {st['id']}, ~${st['cost'] * (time.time() - st['t0']) / 3600:.2f}")
        (HERE / "pods" / f"{name}.json").unlink()
