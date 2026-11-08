#!/usr/bin/env python3
"""One RunPod A40 for the ladder, plumbing from ../autoresearch/pod.py (the rl_diff_figs launcher,
one pod).

  python launch.py up        # rent, wait for ssh, arm the self-destruct watchdog
  python launch.py start     # upload the code and data, set up, start queue.sh detached
  python launch.py monitor   # every 5 min: pull results and the log, enforce the budget, terminate
  when the queue ends
  python launch.py status | pull | down
Needs RUNPOD_API_KEY (repo .envrc.local). Spend stops at BUDGET; the on-pod watchdog bounds the
pod's lifetime if this
Mac is asleep. Results land in artifacts/ (crosscoders/*/{meta,eval}.json + latents.pt; seqs stay on
the pod).
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
ART = HERE / "artifacts"
LOGS = HERE / "logs"
BUDGET = 15.0
MAX_SECS = 9 * 3600
NAME = "cc"
# tarred from code/experiments/current so the layout (this dir beside long_rl_20261014) survives on
# the pod
CODE = (
    "crosscoder_ladder_20261106/common.py crosscoder_ladder_20261106/collect.py "
    "crosscoder_ladder_20261106/crosscoder.py "
    "crosscoder_ladder_20261106/setup_pod.sh crosscoder_ladder_20261106/queue.sh "
    "crosscoder_ladder_20261106/cpt.py crosscoder_ladder_20261106/queue_cpt.sh "
    "long_rl_20261014/pretrain.py long_rl_20261014/overlay/lean_tok.py "
    "long_rl_20261014/overlay/lean_gate.py "
    "long_rl_20261014/data/rl.jsonl long_rl_20261014/data/pretrain.jsonl.gz"
)


def key():
    return os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def spent(st):
    """An upper bound: counts to t1 or now; the on-pod watchdog may have killed the pod earlier."""
    return sum(p["cost"] * ((p.get("t1") or time.time()) - p["t0"]) / 3600 for p in st.values())


def up():
    st, pub = load(), Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
    while NAME not in st or "ip" not in st[NAME]:
        p = pod.create(key(), f"ccladder-{NAME}", pub, "SECURE") or pod.create(
            key(), f"ccladder-{NAME}", pub, "COMMUNITY"
        )
        if p is None:
            print("no A40 capacity, retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        st[NAME] = {"id": p["id"], "cost": p["costPerHr"], "t0": time.time()}
        save(st)
        print(f"pod {p['id']} at ${p['costPerHr']}/hr", flush=True)
        addr = pod.wait_ssh(key(), p["id"])
        if not addr:
            pod.terminate(key(), p["id"])
            del st[NAME]
            save(st)
            continue
        st[NAME]["ip"], st[NAME]["port"] = addr
        save(st)
    for _ in range(60):
        if (
            subprocess.run(
                ["ssh", *pod.ssh_args(st[NAME]), "true"],
                capture_output=True,
                stdin=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            break
        time.sleep(5)
    else:
        sys.exit("ssh never accepted the key")
    wd = (HERE.parent / "autoresearch" / "watchdog.sh").read_bytes()
    pod.ssh_run(
        st[NAME],
        f"echo {base64.b64encode(wd).decode()} | base64 -d > /tmp/watchdog.sh && "
        f"{{ nohup setsid bash /tmp/watchdog.sh {MAX_SECS} > /tmp/wd.log 2>&1 < /dev/null & }}",
    )
    print(f"up, watchdog {MAX_SECS / 3600:.0f} h", flush=True)


def start():
    p = load()[NAME]
    ssh = "ssh " + " ".join(pod.ssh_args(p))
    subprocess.run(f'{ssh} -n "mkdir -p /workspace/job"', shell=True, check=True)
    subprocess.run(
        f"tar czf - -C {shlex.quote(str(HERE.parent))} {CODE} "
        f"| {ssh} 'tar xzf - -C /workspace/job'",
        shell=True,
        check=True,
    )
    d = "/workspace/job/crosscoder_ladder_20261106"
    job = f"bash {d}/setup_pod.sh && bash {d}/queue.sh"
    pod.ssh_run(
        p, f"nohup setsid bash -lc {shlex.quote(job)} > /workspace/{NAME}.log 2>&1 < /dev/null &"
    )
    print("started", flush=True)


def pull(st):
    """Copy crosscoder results and the log here. Returns True when the queue has finished."""
    p = st[NAME]
    if p.get("t1"):
        return False
    ssh = ["ssh", *pod.ssh_args(p)]
    ART.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    got = subprocess.run(
        [
            *ssh,
            "cd /workspace/art && find crosscoders -name '*.json' -o -name latents.pt "
            "-o -name train_log.jsonl | tar czf - -T -",
        ],
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if got.returncode == 0 and got.stdout:
        subprocess.run(["tar", "xzf", "-", "-C", str(ART)], input=got.stdout, check=False)
    summ = subprocess.run(
        [*ssh, "cd /workspace/ckpts && tar czhf - */summary.json"],
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if summ.returncode == 0 and summ.stdout:
        (ART / "summaries").mkdir(exist_ok=True)
        subprocess.run(
            ["tar", "xzf", "-", "-C", str(ART / "summaries")], input=summ.stdout, check=False
        )
    done = False
    for name in (
        NAME,
        "cpt",
    ):  # queue.sh writes cc.log, queue_cpt.sh cpt.log; ALL_DONE ends the latter
        log = subprocess.run(
            [*ssh, f"cat /workspace/{name}.log"], capture_output=True, stdin=subprocess.DEVNULL
        )
        if log.returncode == 0:
            (LOGS / f"{name}.log").write_bytes(log.stdout)
            done = done or b"ALL_DONE" in log.stdout
    return done


def terminate(st):
    if not st[NAME].get("t1"):
        pod.terminate(key(), st[NAME]["id"])
        st[NAME]["t1"] = time.time()
        print("terminated", flush=True)
    save(st)


def monitor():
    while True:
        st = load()
        done = pull(st)
        n = sum(1 for _ in ART.glob("crosscoders/*/eval.json"))
        print(
            f"{time.strftime('%H:%M')} ${spent(st):.2f} spent, {n} crosscoders done, "
            f"queue finished: {done}",
            flush=True,
        )
        if done or spent(st) > BUDGET:
            if not done:
                print("budget reached; final pull", flush=True)
                pull(st)
            terminate(st)
            print("MONITOR_DONE", flush=True)
            return
        time.sleep(300)


def main():
    cmd = sys.argv[1]
    if cmd == "up":
        up()
    elif cmd == "start":
        start()
    elif cmd == "monitor":
        monitor()
    elif cmd == "pull":
        print(pull(load()))
    elif cmd == "status":
        st = load()
        print(f"${spent(st):.2f}", json.dumps(st))
    elif cmd == "down":
        terminate(load())


if __name__ == "__main__":
    main()
