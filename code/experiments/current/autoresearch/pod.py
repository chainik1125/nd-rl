#!/usr/bin/env python3
"""One long-lived RunPod GPU for the autoresearch loop (boot + clone once, run many experiments).

  python pod.py up [--pod P]          # rent an A40, wait for ssh, clone the fork; saves pod_P.json
  python pod.py run <name> [--pod P] [--pretrain F] [args...]
                                      # upload harness + pretrain (F as pretrain.py), run, pull,
                                      # then move the checkpoints to Hugging Face (push below)
  python pod.py down [--pod P]        # terminate the pod in pod_P.json
  python pod.py list | kill <id|all>
  python pod.py push [name...]        # retry: move checkpoints still on this disk to HF
--pod names a pod for parallel use (default "main"); sweep.py drives many at once.

Adapted from pretrain_poc/pod.py, which rented one pod per run. Needs RUNPOD_API_KEY
(~/repos/nd-takehome/.envrc.local has it) and ~/.ssh/id_ed25519. Pushing needs the `hf` CLI,
logged in with write access to HF_REPO.
"""

import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.runpod.io/graphql"
IMAGE = os.environ.get("ND_POD_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
GPU = "NVIDIA A40"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ART = os.path.join(
    HERE, "..", "..", "..", "..", "artifacts", "autoresearch"
)  # repo artifacts/ is gitignored
REMOTE = "/workspace/autoresearch"
HF_REPO = "robbiethompson2018/nd-rl-checkpoints"  # the one HF repo for nd-rl; runs/<name>/


def post_json(req):
    """This Mac's network resets connections now and then: retry with backoff, last try raises."""
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except (OSError, json.JSONDecodeError):
            time.sleep(10 * (attempt + 1))
    return json.load(urllib.request.urlopen(req, timeout=60))


def gql(query, key):
    req = urllib.request.Request(
        f"{API}?api_key={key}",
        method="POST",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
    )
    out = post_json(req)
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
        transient = ("SUPPLY_CONSTRAINT", "does not have the resources", "INTERNAL_SERVER_ERROR")
        if any(s in str(e) for s in transient):
            return None  # no capacity / flaky RunPod API right now: retry in `up`
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
    return None


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
        "-o",
        "ConnectTimeout=20",
        f"root@{st['ip']}",
    ]


def ssh_run(st, cmd, check=True):
    """Exit 255 is ssh itself failing (this Mac's network drops); retry, callers' commands are
    idempotent."""
    rc = 255
    for _ in range(5):
        # stdin from /dev/null: newer images' login profile reads stdin and would block forever
        argv = ["ssh", *ssh_args(st), "bash", "-lc", shlex.quote(cmd)]
        rc = subprocess.run(argv, stdin=subprocess.DEVNULL).returncode
        if rc != 255:
            break
        time.sleep(15)
    if check and rc:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def run_detached(st, cmd, out, retry_secs=10, give_up_secs=1800):
    """Run `cmd` on the pod so an ssh drop cannot kill it, and stream its output to stdout.

    The job runs under setsid/nohup, writing `out`.stream and then its exit code to `out`.rc.
    We `tail` the stream over ssh, counting bytes received; after a drop we reconnect and resume
    at that byte offset, so the local log stays identical. Returns the job's exit code, or 255
    after `give_up_secs` without a working connection.
    """
    stream, rc_file = f"{out}.stream", f"{out}.rc"
    job = f"({cmd}) > {stream} 2>&1; echo $? > {rc_file}"
    ssh_run(
        st,
        f"mkdir -p {os.path.dirname(out)}; rm -f {rc_file} {stream}; "
        f"nohup setsid bash -c {shlex.quote(job)} >/dev/null 2>&1 </dev/null &",
    )
    offset, last_ok = 0, time.time()
    while True:
        follow = (
            f"tail -c +{offset + 1} -F {stream} 2>/dev/null & t=$!; "
            f"while [ ! -f {rc_file} ]; do sleep 2; done; sleep 2; kill $t; wait $t 2>/dev/null; "
            f'echo "exit-code $(cat {rc_file})" >&2'
        )
        proc = subprocess.Popen(
            ["ssh", *ssh_args(st), "bash", "-lc", shlex.quote(follow)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None and proc.stderr is not None
        while chunk := os.read(proc.stdout.fileno(), 65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            offset += len(chunk)
            last_ok = time.time()
        codes = re.findall(r"^exit-code (\d+)$", proc.stderr.read().decode(), re.M)
        if proc.wait() == 0 and codes:
            return int(codes[-1])
        if time.time() - last_ok > give_up_secs:
            print(f"lost the pod for {give_up_secs} s; giving up", flush=True)
            return 255
        print(f"ssh dropped at byte {offset}; reconnecting in {retry_secs} s", flush=True)
        time.sleep(retry_secs)


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def hf_sha256(paths):
    """{repo path: sha256} of these files on the public HF repo (an unauthenticated read)."""
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{HF_REPO}/paths-info/main",
        data=urllib.parse.urlencode({"paths": paths, "expand": "true"}, doseq=True).encode(),
    )
    return {
        f["path"]: (f.get("lfs") or {}).get("oid") for f in json.load(urllib.request.urlopen(req))
    }


def push_checkpoints(name, attempts=3, retry_secs=30):
    """Upload LOCAL_ART/<name>/ to HF runs/<name>, then delete each local .pt whose sha256 matches
    HF's copy: checkpoints live on HF, not on this disk. Any failure keeps the files for a later
    `pod.py push`, so no checkpoint is deleted before HF has it."""
    local = os.path.join(LOCAL_ART, name)
    pts = sorted(f for f in os.listdir(local) if f.endswith(".pt")) if os.path.isdir(local) else []
    if not pts:
        return
    remote = {}
    for attempt in range(attempts):
        if attempt:
            time.sleep(retry_secs)
        try:
            up = subprocess.run(
                ["hf", "upload", HF_REPO, local, f"runs/{name}", "--commit-message", f"Add {name}"],
                # From this Mac each Xet connection moves ~0.5 MB/s and adaptive concurrency
                # settles near 1, so a seed's ~200 MB took ~3 min while its pod idled.
                env={
                    **os.environ,
                    "HF_XET_FIXED_UPLOAD_CONCURRENCY": "8",
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # keep driver logs short
                },
            )
            if up.returncode == 0:
                remote = hf_sha256([f"runs/{name}/{f}" for f in pts])
                break
        except (OSError, ValueError) as e:  # no `hf` binary, network error, bad JSON
            print(f"HF push of {name}: {e}", flush=True)
    kept = [f for f in pts if remote.get(f"runs/{name}/{f}") != sha256(os.path.join(local, f))]
    for f in pts:
        if f not in kept:
            os.remove(os.path.join(local, f))
    if kept:
        print(f"HF push of {name}: {kept} not on HF, kept locally; retry: pod.py push", flush=True)
    else:
        print(
            f"checkpoints moved to https://huggingface.co/{HF_REPO}/tree/main/runs/{name}",
            flush=True,
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
    if cmd == "push":  # without names: every run dir, except ones a `run` may still be pulling
        for name in rest or sorted(os.listdir(LOCAL_ART)):
            if rest or time.time() - os.path.getmtime(os.path.join(LOCAL_ART, name)) > 600:
                push_checkpoints(name)
        return
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
        # Up to ~30 min of waiting for capacity; a pod that never exposes ssh is replaced.
        deadline, attempt, addr, st = time.time() + 1800, 0, None, {}
        while addr is None:
            if time.time() > deadline:
                raise SystemExit(f"no usable {GPU} pod after 30 min")
            attempt += 1
            pod = create(key, f"autoresearch-{pod_name}", pubkey, "SECURE") or create(
                key, f"autoresearch-{pod_name}", pubkey, "COMMUNITY"
            )
            if pod is None:
                print(f"no {GPU} capacity (attempt {attempt}), retrying in 30 s", flush=True)
                time.sleep(30)
                continue
            st = {"id": pod["id"], "cost": pod["costPerHr"], "t0": time.time()}
            with open(STATE, "w") as f:
                json.dump(st, f)  # saved before waiting, so `down` works if the wait is interrupted
            print(f"pod {pod['id']} at ${pod['costPerHr']}/hr", flush=True)
            if (addr := wait_ssh(key, pod["id"])) is None:
                terminate(key, pod["id"])
                os.remove(STATE)
                print(f"pod {pod['id']} never exposed ssh; terminated, retrying", flush=True)
        st["ip"], st["port"] = addr
        with open(STATE, "w") as f:
            json.dump(st, f)
        for _ in range(60):
            probe = ["ssh", *ssh_args(st), "true"]
            if subprocess.run(probe, capture_output=True, stdin=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(5)
        else:
            raise SystemExit("ssh never accepted the key")
        # Self-destruct, started before anything else can fail: pods are torn down only by this
        # machine, so if it sleeps or loses the network (2026-09-26: a bench pod idled ~8.5 h)
        # the pod deletes itself after this long.
        max_secs = int(os.environ.get("ND_POD_MAX_SECS", 5400))
        with open(os.path.join(HERE, "watchdog.sh"), "rb") as script:
            b64 = base64.b64encode(
                script.read()
            ).decode()  # ssh stdin gets eaten by the login shell
        ssh_run(
            st,
            f"echo {b64} | base64 -d > /tmp/watchdog.sh && "
            f"nohup setsid bash /tmp/watchdog.sh {max_secs} > /tmp/wd.log 2>&1 < /dev/null &",
        )
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
        rc = run_detached(
            st, f"bash {REMOTE}/remote.sh {out} " + " ".join(map(shlex.quote, args)), out
        )
        ssh = "ssh -n " + " ".join(ssh_args(st))  # -n: stdin from /dev/null
        pulled = False
        for _ in range(10):  # the pull can hit the same network drops as the stream
            pulled = (
                subprocess.run(
                    f'{ssh} "cd {REMOTE}/artifacts && tar czf - {name} {name}.log" '
                    f"| tar xzf - -C {shlex.quote(LOCAL_ART)}",
                    shell=True,
                    check=False,
                ).returncode
                == 0
            )
            if pulled:
                break
            time.sleep(30)
        print(
            f"artifacts in {os.path.abspath(os.path.join(LOCAL_ART, name))}; remote rc {rc}",
            flush=True,
        )
        if pulled:  # a failed pull may have left truncated checkpoints
            push_checkpoints(name)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
