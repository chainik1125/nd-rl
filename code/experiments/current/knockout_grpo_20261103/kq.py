"""A small per-machine GPU job queue for knockout_grpo (stdlib only; runs from KO/tools/).

    python3 kq.py worker GPU [--max_ours N] [--max_total N]   # one per GPU, under setsid nohup
    python3 kq.py status                                       # rewrite KO/STATUS.md
    python3 kq.py add '{"id": ..., "cmd": "...", "expected_h": 1.0}'

Jobs: KO/queue/jobs.jsonl, one JSON per line, in priority order: {id, cmd (bash), expected_h,
timeout_h (default 2 x expected + 0.5), deps [ids], gpus [allowed GPU indices] (optional)}.
State: KO/queue/state/<id>.json ({status: running / done / failed, host, gpu, pid, start, end,
rc}); delete a failed state file to requeue. Stops: KO/queue/STOP or STOP_gpuN (no new launch).

Gate, right before every launch: the GPU has no compute process of another user and < 500 MiB in
use; our GPU claim is a mkdir lock (KO/queue/locks/gpuN). `--max_ours` / `--max_total` cap the
machine's GPUs in use by us / by anyone (serrano's PSU rule: 2 / 3). Every job runs under
`timeout` with the watchdog (KO/tools/watchdog.py: STALL / EXIT / OVERRUN lines in
KO/logs/watchdog.log), with ND_OFFLINE=1, PYTHONDONTWRITEBYTECODE=1 and LEAN_GATE_WORKERS=8.
"""

from __future__ import annotations

import datetime as dt
import getpass
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

KO = Path(os.environ.get("KO", "/scratch/cowifornia/nd-rl/knockout_grpo"))
Q = KO / "queue"
ME = getpass.getuser()
HOST = socket.gethostname().split(".")[0]


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def jobs() -> list[dict]:
    p = Q / "jobs.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def state(jid: str) -> dict | None:
    p = Q / "state" / f"{jid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def put_state(jid: str, st: dict) -> None:
    (Q / "state").mkdir(parents=True, exist_ok=True)
    tmp = Q / "state" / f".{jid}.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(Q / "state" / f"{jid}.json")


def smi(q: str, gpu: int | None = None) -> list[str]:
    cmd = ["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"]
    if gpu is not None:
        cmd += ["-i", str(gpu)]
    return [
        x.strip()
        for x in subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
        if x.strip()
    ]


def owner(pid: str) -> str:
    r = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True)
    return r.stdout.strip() or "?"


def gpu_users() -> dict[int, set[str]]:
    """GPU index -> owners of its compute processes."""
    uuid2i = {}
    for line in smi("gpu=index,uuid"):
        i, u = [x.strip() for x in line.split(",")]
        uuid2i[u] = int(i)
    out: dict[int, set[str]] = {i: set() for i in uuid2i.values()}
    for line in smi("compute-apps=gpu_uuid,pid"):
        u, pid = [x.strip() for x in line.split(",")]
        if u in uuid2i:
            out[uuid2i[u]].add(owner(pid))
    return out


def gate(gpu: int, max_ours: int, max_total: int) -> tuple[bool, str]:
    users = gpu_users()
    if users.get(gpu, set()) - {ME}:
        return False, f"GPU {gpu} has another user's process {sorted(users[gpu])}"
    if users.get(gpu):
        return False, f"GPU {gpu} has a process of ours"
    used = int(smi("gpu=memory.used", gpu)[0])
    if used >= 500:
        return False, f"GPU {gpu} {used} MiB in use"
    ours = sum(1 for p in (Q / "locks").glob("gpu*") if p.is_dir())
    busy = {i for i, u in users.items() if u}
    busy |= {int(p.name[3:]) for p in (Q / "locks").glob("gpu*") if p.is_dir()}
    if ours + 1 > max_ours:
        return False, f"{ours} GPUs of ours in use (max {max_ours})"
    if len(busy | {gpu}) > max_total:
        return False, f"GPUs in use by anyone {sorted(busy)} + {gpu} > {max_total}"
    return True, "ok"


def runnable(j: dict, gpu: int) -> bool:
    if state(j["id"]) is not None:
        return False
    if j.get("gpus") is not None and gpu not in j["gpus"]:
        return False
    for d in j.get("deps", []):
        s = state(d)
        if s is None or s["status"] != "done":
            return False
    return True


def log(msg: str) -> None:
    with open(KO / "logs" / f"kq_{HOST}.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def run_job(j: dict, gpu: int) -> int:
    jid = j["id"]
    lg = KO / "logs" / f"{jid}.log"
    exp_h = float(j.get("expected_h", 1.0))
    to_s = int(3600 * float(j.get("timeout_h", 2 * exp_h + 0.5)))
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        ND_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        ND_REGISTRY_SYNC="0",
        LEAN_GATE_WORKERS="8",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        ND_LEAN=os.path.expanduser("~/.elan/bin/lean"),
        TMPDIR=str(KO / "tmp"),
        KO=str(KO),
        KO_GPU=str(gpu),
    )
    (KO / "tmp").mkdir(exist_ok=True)
    with open(lg, "a") as out:
        out.write(f"=== {now()} {HOST} GPU {gpu}: {j['cmd']}\n")
        out.flush()
        p = subprocess.Popen(
            ["timeout", "--signal=TERM", "--kill-after=300", str(to_s), "bash", "-c", j["cmd"]],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(KO),
            start_new_session=True,
        )
    put_state(jid, {"status": "running", "host": HOST, "gpu": gpu, "pid": p.pid, "start": now()})
    log(f"LAUNCH {jid} GPU {gpu} pid {p.pid} expected {exp_h} h timeout {to_s} s")
    wd = subprocess.Popen(
        ["python3", str(KO / "tools" / "watchdog.py"), f"{jid}:{p.pid}:{gpu}:{lg}:{exp_h}"],
        stdout=open(KO / "logs" / "watchdog.log", "a"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rc = p.wait()
    wd.terminate()
    st = state(jid) or {}
    st.update(status="done" if rc == 0 else "failed", end=now(), rc=rc)
    put_state(jid, st)
    log(f"EXIT {jid} GPU {gpu} rc {rc}")
    return rc


def worker(gpu: int, max_ours: int, max_total: int) -> None:
    (Q / "locks").mkdir(parents=True, exist_ok=True)
    (KO / "logs").mkdir(exist_ok=True)
    log(f"worker GPU {gpu} start (max_ours {max_ours}, max_total {max_total})")
    last_msg = ""
    while True:
        if (Q / "STOP").exists() or (Q / f"STOP_gpu{gpu}").exists():
            log(f"worker GPU {gpu}: STOP file, exiting")
            return
        todo = [j for j in jobs() if runnable(j, gpu)]
        if not todo:
            if all(state(j["id"]) is not None for j in jobs()):
                pass
            time.sleep(60)
            continue
        ok, why = gate(gpu, max_ours, max_total)
        if not ok:
            if why != last_msg:
                log(f"worker GPU {gpu}: wait ({why})")
                last_msg = why
            time.sleep(60)
            continue
        lock = Q / "locks" / f"gpu{gpu}"
        try:
            lock.mkdir()
        except FileExistsError:
            time.sleep(30)
            continue
        try:
            j = next((x for x in jobs() if runnable(x, gpu)), None)
            if j is not None:
                # claim atomically against other workers on this machine
                claim = Q / "state" / f"{j['id']}.claim"
                (Q / "state").mkdir(exist_ok=True)
                try:
                    claim.mkdir()
                except FileExistsError:
                    continue
                run_job(j, gpu)
                status()
        finally:
            lock.rmdir()
        last_msg = ""


def status() -> None:
    js = jobs()
    rows, n = [], {"done": 0, "running": 0, "failed": 0, "pending": 0}
    for j in js:
        s = state(j["id"])
        k = s["status"] if s else "pending"
        n[k] = n.get(k, 0) + 1
        extra = ""
        if s:
            extra = f"GPU {s.get('gpu')} start {s.get('start')} end {s.get('end', '')} rc {s.get('rc', '')}"  # noqa: E501
        rows.append(f"- `{j['id']}` **{k}** (expected {j.get('expected_h')} h) {extra}")
    txt = (
        f"# knockout_grpo queue on {HOST}\n\nUpdated {now()}. Jobs: {n['done']} done, "
        f"{n['running']} running, {n['pending']} pending, {n['failed']} failed.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    extra = KO / "STATUS_notes.md"
    if extra.exists():
        txt += "\n" + extra.read_text()
    (KO / f"STATUS_{HOST}.md").write_text(txt)


def main() -> None:
    a = sys.argv[1:]
    if a[0] == "worker":
        mo = int(a[a.index("--max_ours") + 1]) if "--max_ours" in a else 99
        mt = int(a[a.index("--max_total") + 1]) if "--max_total" in a else 99
        worker(int(a[1]), mo, mt)
    elif a[0] == "status":
        status()
        print((KO / f"STATUS_{HOST}.md").read_text())
    elif a[0] == "add":
        j = json.loads(a[1])
        assert "id" in j and "cmd" in j
        with open(Q / "jobs.jsonl", "a") as f:
            f.write(json.dumps(j) + "\n")


if __name__ == "__main__":
    main()
