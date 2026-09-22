#!/usr/bin/env python3
"""One long-lived RunPod GPU for the autoresearch loop (boot + clone once, run many experiments).

  python pod.py up [--pod P]          # rent an A40, wait for ssh, clone the fork; saves pod_P.json
  python pod.py run <name> [--pod P] [--pretrain F] [args...]
                                      # upload harness + pretrain (F as pretrain.py), run, pull
  python pod.py down [--pod P]        # terminate the pod in pod_P.json
  python pod.py list | kill <id|all>
--pod names a pod for parallel use (default "main"); sweep.py drives many at once.

Adapted from pretrain_poc/pod.py, which rented one pod per run. Needs RUNPOD_API_KEY
(~/repos/nd-takehome/.envrc.local has it) and ~/.ssh/id_ed25519.
"""

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA A40"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ART = os.path.join(
    HERE, "..", "..", "..", "..", "artifacts", "autoresearch"
)  # repo artifacts/ is gitignored
REMOTE = "/workspace/autoresearch"


def gql(query, key):
    req = urllib.request.Request(
        f"{API}?api_key={key}",
        method="POST",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
    )
    out = json.load(urllib.request.urlopen(req))
    if "errors" in out:
        raise SystemExit(json.dumps(out["errors"], indent=2))
    return out["data"]


def create(key, name, pubkey, cloud):
    q = (
        "mutation { podFindAndDeployOnDemand(input: {"
        f"cloudType: {cloud}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 40, "
        "minVcpuCount: 4, minMemoryInGb: 16, "
        f'gpuTypeId: "{GPU}", name: "{name}", imageName: "{IMAGE}", ports: "22/tcp", '
        'volumeMountPath: "/workspace", '
        f'supportPublicIp: true, env: [{{key: "PUBLIC_KEY", value: "{pubkey}"}}]'
        "}) { id costPerHr } }"
    )
    try:
        return gql(q, key)["podFindAndDeployOnDemand"]
    except SystemExit as e:
        if "SUPPLY_CONSTRAINT" in str(e):
            return None
        raise


def wait_ssh(key, pod_id, timeout=600):
    q = (
        f'query {{ pod(input: {{podId: "{pod_id}"}}) '
        "{ runtime { ports { ip publicPort privatePort isIpPublic } } } }"
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        rt = gql(q, key)["pod"]["runtime"]
        for p in (rt or {}).get("ports") or []:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], p["publicPort"]
        time.sleep(5)
    raise SystemExit("pod never exposed ssh")


def ssh_args(st):
    return [
        "-p",
        str(st["port"]),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ServerAliveInterval=30",
        f"root@{st['ip']}",
    ]


def ssh_run(st, cmd, check=True):
    return subprocess.run(
        ["ssh", *ssh_args(st), "bash", "-lc", shlex.quote(cmd)], check=check
    ).returncode


def upload(st, pretrain_file):
    """remote.sh + harness.py verbatim; `pretrain_file` lands on the pod as pretrain.py
    (sweep variants)."""
    ssh = "ssh " + " ".join(ssh_args(st))
    subprocess.run(
        f"tar czf - -C {shlex.quote(HERE)} remote.sh harness.py "
        f'| {ssh} "mkdir -p {REMOTE} && tar xzf - -C {REMOTE}"',
        shell=True,
        check=True,
    )
    subprocess.run(
        f"scp -q -P {st['port']} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "
        f"{shlex.quote(os.path.join(HERE, pretrain_file))} root@{st['ip']}:{REMOTE}/pretrain.py",
        shell=True,
        check=True,
    )


def terminate(key, pod_id):
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', key)


def load_state():
    if not os.path.exists(STATE):
        raise SystemExit(f"no {STATE}; run `pod.py up` first")
    with open(STATE) as f:
        return json.load(f)


def pop_opt(rest, flag, default):
    if flag in rest:
        i = rest.index(flag)
        val = rest[i + 1]
        del rest[i : i + 2]
        return val
    return default


def main():
    global STATE
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else "list"), sys.argv[2:]
    pod_name = pop_opt(rest, "--pod", "main")
    pretrain_file = pop_opt(rest, "--pretrain", "pretrain.py")
    STATE = os.path.join(LOCAL_ART, f"pod_{pod_name}.json")
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY unset")
    pods = gql("query { myself { clientBalance pods { id name desiredStatus costPerHr } } }", key)[
        "myself"
    ]
    os.makedirs(LOCAL_ART, exist_ok=True)

    if cmd == "list":
        print(json.dumps(pods, indent=1))
    elif cmd == "kill":
        for p in pods["pods"]:
            if rest[0] in (p["id"], "all"):
                terminate(key, p["id"])
                print("terminated", p["id"], p["name"])
        if os.path.exists(STATE):
            os.remove(STATE)
    elif cmd == "down":
        st = load_state()
        terminate(key, st["id"])
        os.remove(STATE)
        mins = (time.time() - st["t0"]) / 60
        print(f"terminated {st['id']} after {mins:.1f} min (~${st['cost'] * mins / 60:.2f})")
    elif cmd == "up":
        live = [p for p in pods["pods"] if p["desiredStatus"] == "RUNNING"]
        print(f"balance ${pods['clientBalance']:.2f}; running pods: {live}", flush=True)
        if os.path.exists(STATE):
            raise SystemExit(f"{STATE} exists; `pod.py down` first")
        with open(os.path.expanduser("~/.ssh/id_ed25519.pub")) as f:
            pubkey = f.read().strip()
        pod = None
        for attempt in range(60):  # up to ~30 min of waiting for capacity
            pod = create(key, f"autoresearch-{pod_name}", pubkey, "SECURE") or create(
                key, f"autoresearch-{pod_name}", pubkey, "COMMUNITY"
            )
            if pod:
                break
            print(f"no {GPU} capacity (attempt {attempt + 1}), retrying in 30 s", flush=True)
            time.sleep(30)
        if pod is None:
            raise SystemExit(f"no {GPU} capacity after 30 min")
        st = {"id": pod["id"], "cost": pod["costPerHr"], "t0": time.time()}
        with open(STATE, "w") as f:
            json.dump(st, f)  # saved before waiting, so `down` works if the wait is interrupted
        print(f"pod {pod['id']} at ${pod['costPerHr']}/hr", flush=True)
        st["ip"], st["port"] = wait_ssh(key, pod["id"])
        with open(STATE, "w") as f:
            json.dump(st, f)
        for _ in range(60):
            if subprocess.run(["ssh", *ssh_args(st), "true"], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        else:
            raise SystemExit("ssh never accepted the key")
        upload(st, pretrain_file)
        ssh_run(
            st,
            f"cd {REMOTE} && bash remote.sh /tmp/warm --help > /dev/null 2>&1; "
            "nvidia-smi --query-gpu=name --format=csv,noheader",
        )
        print(f"up after {time.time() - st['t0']:.0f}s: ssh -p {st['port']} root@{st['ip']}")
    elif cmd == "run":
        st = load_state()
        name, args = rest[0], rest[1:]
        upload(st, pretrain_file)
        out = f"{REMOTE}/artifacts/{name}"
        rc = ssh_run(
            st,
            f"mkdir -p {REMOTE}/artifacts && bash {REMOTE}/remote.sh {out} "
            + " ".join(map(shlex.quote, args)),
            check=False,
        )
        ssh = "ssh " + " ".join(ssh_args(st))
        subprocess.run(
            f'{ssh} "cd {REMOTE}/artifacts && tar czf - {name} {name}.log" '
            f"| tar xzf - -C {shlex.quote(LOCAL_ART)}",
            shell=True,
            check=False,
        )
        print(
            f"artifacts in {os.path.abspath(os.path.join(LOCAL_ART, name))}; remote rc {rc}",
            flush=True,
        )
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
