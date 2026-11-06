#!/usr/bin/env python3
"""Two RunPod A40s for the overnight queues, plumbing from ../autoresearch/pod.py.

  python launch.py up svd sae      # rent, wait for ssh, arm the self-destruct watchdog
  python launch.py start svd sae   # upload the code, set up, start each queue detached
  python launch.py monitor         # every 10 min: pull results, enforce the budget, terminate pods when queues finish
  python launch.py status | pull | down
Needs RUNPOD_API_KEY (repo .envrc.local). Spend stops at BUDGET: the monitor terminates everything there, and the
on-pod watchdog bounds each pod's lifetime if this Mac is asleep.
"""
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "autoresearch"))
import pod  # noqa: E402

STATE = HERE / "pods.json"
RESULTS = HERE / "results"
LOGS = HERE / "logs"
BUDGET = 38.0
MAX_SECS = 16 * 3600  # 2 pods x 16 h x $0.59 = $19, inside the budget even if nobody comes back
CODE = "common.py run_svd.py run_sae.py test_identities.py queue_svd.sh queue_sae.sh setup_pod.sh data"


def key():
    return os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def spent(st):
    return sum(p["cost"] * ((p.get("t1") or time.time()) - p["t0"]) / 3600 for p in st.values())


def up(names):
    st, pub = load(), Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    for name in names:
        while name not in st or "ip" not in st[name]:
            p = pod.create(key(), f"rldiff-{name}", pub, "SECURE") or pod.create(key(), f"rldiff-{name}", pub, "COMMUNITY")
            if p is None:
                print("no A40 capacity, retrying in 30 s", flush=True)
                time.sleep(30)
                continue
            st[name] = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
            save(st)
            print(f"{name}: pod {p['id']} at ${p['costPerHr']}/hr", flush=True)
            addr = pod.wait_ssh(key(), p["id"])
            if not addr:
                pod.terminate(key(), p["id"])
                del st[name]
                save(st)
                continue
            st[name]["ip"], st[name]["port"] = addr
            save(st)
        for _ in range(60):
            if subprocess.run(["ssh", *pod.ssh_args(st[name]), "true"], capture_output=True, stdin=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(5)
        else:
            sys.exit(f"{name}: ssh never accepted the key")
        wd = (HERE.parent / "autoresearch" / "watchdog.sh").read_bytes()
        pod.ssh_run(st[name], f"echo {base64.b64encode(wd).decode()} | base64 -d > /tmp/watchdog.sh && "
                              f"{{ nohup setsid bash /tmp/watchdog.sh {MAX_SECS} > /tmp/wd.log 2>&1 < /dev/null & }}")
        print(f"{name}: up, watchdog {MAX_SECS / 3600:.0f} h", flush=True)


def start(names):
    st = load()
    for name in names:
        p = st[name]
        ssh = "ssh " + " ".join(pod.ssh_args(p))
        subprocess.run(f'{ssh} -n "mkdir -p /workspace/job"', shell=True, check=True)
        subprocess.run(f'tar czf - -C {shlex.quote(str(HERE))} {CODE} | {ssh} "tar xzf - -C /workspace/job"', shell=True, check=True)
        job = f"bash /workspace/job/setup_pod.sh {name} && bash /workspace/job/queue_{name}.sh"
        pod.ssh_run(p, f"nohup setsid bash -lc {shlex.quote(job)} > /workspace/{name}.log 2>&1 < /dev/null &")
        print(f"{name}: started", flush=True)


def pull(st):
    """Copy results and logs here. Returns the names whose queue has finished."""
    finished = []
    for name, p in st.items():
        if p.get("t1"):
            continue
        ssh = ["ssh", *pod.ssh_args(p)]
        RESULTS.mkdir(exist_ok=True)
        LOGS.mkdir(exist_ok=True)
        got = subprocess.run([*ssh, "cd /workspace/results && find . -name '*.json' -o -name DONE -o -name sae.pt | tar czf - -T -"], capture_output=True, stdin=subprocess.DEVNULL)
        if got.returncode == 0:
            subprocess.run(["tar", "xzf", "-", "-C", str(RESULTS)], input=got.stdout, check=False)
        log = subprocess.run([*ssh, f"cat /workspace/{name}.log"], capture_output=True, stdin=subprocess.DEVNULL)
        if log.returncode == 0:
            (LOGS / f"{name}.log").write_bytes(log.stdout)
            if b"QUEUE_DONE" in log.stdout:
                finished.append(name)
    return finished


def terminate(st, names):
    for name in names:
        if not st[name].get("t1"):
            pod.terminate(key(), st[name]["id"])
            st[name]["t1"] = time.time()
            print(f"{name}: terminated", flush=True)
    save(st)


def monitor():
    while True:
        st = load()
        finished = pull(st)
        terminate(st, finished)
        live = [n for n, p in st.items() if not p.get("t1")]
        n = sum(1 for _ in RESULTS.rglob("*.json"))
        print(f"{time.strftime('%H:%M')} ${spent(st):.2f} spent, {n} result files, live: {live}", flush=True)
        if spent(st) > BUDGET:
            print("budget reached; final pull", flush=True)
            pull(st)
            terminate(st, live)
            return
        if not live:
            print("MONITOR_DONE", flush=True)
            return
        time.sleep(600)


def main():
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "up":
        up(names)
    elif cmd == "start":
        start(names)
    elif cmd == "monitor":
        monitor()
    elif cmd == "pull":
        pull(load())
    elif cmd == "status":
        st = load()
        print(f"${spent(st):.2f}", json.dumps(st))
    elif cmd == "down":
        st = load()
        terminate(st, names or list(st))


if __name__ == "__main__":
    main()
