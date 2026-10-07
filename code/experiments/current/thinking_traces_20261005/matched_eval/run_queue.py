"""Matched-attempt eval queue: EI rounds vs GRPO iterations at equal sampled attempts, one worker
per GPU.

    nohup timeout 64800 code/.venv/bin/python \
        code/experiments/current/thinking_traces_20261005/matched_eval/run_queue.py \
        [--takeover OLD_RUNNER_PID] \
        >> /scratch/cowifornia/nd-rl/think/runs/logs/matched_eval_runner.log 2>&1 &

One EI round (5,000 targets x 32 attempts) = 39 GRPO iterations (256 x 16): EI r1 <-> GRPO it39,
r2 <-> it78 (both GRPO runs stop at it78).  Arms: no think (eiT0, grpoT0; B 0) and the effort
model (eiT1v1e, grpoT1v1e; B 0,64,256, `sample.py` prefills `<eB>`).

Jobs are (checkpoint, set), one `code.thinking.eval` call with all the budgets, output in
OUT/<set>_<run>_<r|it>/.  Sets: h12 (hard12_test_1k, k 16), tb72 (textbook72 dev+train, k 64),
and t1320 (test_13_20 at k 8, `timeout 5400`) once that file exists: `../test_13_20/subsample.py`
at n 1,000, seed 0 (equal count per reference length), snapshotted once to
OUT/sets/t1320_n1000.jsonl so every checkpoint sees the same items.

Order: h12/tb72 jobs first (EI, then GRPO by iteration), t1320 jobs after them (they backfill
when no h12/tb72 job is runnable).  A checkpoint is ready 30 s after its `.upload.json` appears.
A worker claims a job by `mkdir OUT/.claims/<job>` (atomic; delete the directory to rerun a job)
and runs it under `timeout 3600` (t1320: 5400) once its GPU shows memory.used < 500 MiB.  GPUs 1
and 4 only: 0 belongs to other agents, and 2 and 3 are reserved for training once the GRPO runs
exit.

A job is done when its summary.json holds every expected cell (eval.py rewrites summary.json
after each budget, so a timed-out job leaves a partial one); done jobs are skipped.  Failed jobs
are logged and not retried.  One line per finished job goes to LOG; a checkpoint still missing
3 h after its expected time is logged as MISSING and dropped; `ALL DONE` is written when every
job is finished (t1320 included only if the set appeared before then), and the runner exits; a
follow-on pipeline waits on that line before using GPUs 1 and 4.  A runner stopped early (signal,
or every worker dead) writes `RUNNER STOPPED` instead.

`--takeover PID` replaces a running runner without killing an eval: every 10 s it SIGSTOPs the
old runner, and if that runner has no child process (no eval in flight, none awaiting its log
line) it SIGKILLs it, else SIGCONTs it.  Then every claim without a status.json (stale or
reserved; nothing is in flight) is removed and the queue starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

R = Path("/scratch/cowifornia/nd-rl/think/runs")
REPO = Path("/scratch/cowifornia/nd-rl/think/nd-rl")
PY = REPO / "code/.venv/bin/python"
OUT = R / "eval/matched"
CLAIMS = OUT / ".claims"
LOG = R / "logs/matched_eval.log"
TB = Path("/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72")
T1320_SRC = R / "pools/test_13_20/test_13_20.jsonl"
T1320_KEYS = R / "pools/test_13_20/test_13_20.keys.jsonl"
T1320 = OUT / "sets/t1320_n1000.jsonl"
SUBSAMPLE = Path(__file__).resolve().parent.parent / "test_13_20/subsample.py"
T1320_N, T1320_SETTLE = 1000, 600

SETS = {  # name -> (comma-joined files, k, timeout s)
    "h12": (str(R / "restart2_hard12/hard12_test_1k.jsonl"), 16, 3600),
    "tb72": (f"{TB}/textbook_dev.jsonl,{TB}/textbook_train.jsonl", 64, 3600),
    "t1320": (str(T1320), 8, 5400),
}
MAIN_SETS = ("h12", "tb72")
ENV = {
    "ND_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ND_REGISTRY_SYNC": "0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ND_LEAN": os.path.expanduser("~/.elan/bin/lean"),
    "TMPDIR": str(R / "tmp"),
}
EVAL_ARGS = ["--arms", "plain", "--batch", "1024", "--seed", "0", "--temperature", "0.8"]
EVAL_ARGS += ["--max_action", "512", "--max_steps", "96", "--max_rej", "10"]
READY_DELAY, GRACE, POLL, FREE_MIB = 30, 3 * 3600, 60, 500
WORKERS: dict[int, int | None] = {1: None, 4: None}  # GPU -> PID that must exit first


def utc(s: str) -> float:
    return datetime.fromisoformat(s + "+00:00").timestamp()


@dataclass(frozen=True)
class Ckpt:
    run: str
    tag: str
    path: Path
    think: bool
    expected: float  # unix time it should exist by

    def ready(self, now: float) -> bool:
        up = Path(str(self.path) + ".upload.json")
        return self.path.exists() and up.exists() and up.stat().st_mtime + READY_DELAY <= now


def _ei(run: str, d: str, r: int, think: bool) -> Ckpt:
    p = R / f"restart2_hard12/{d}/ckpts/{d}_r{r}.pt"
    return Ckpt(run, f"r{r}", p, think, 0.0)


def _grpo(run: str, d: str, it: int, think: bool, when: str) -> Ckpt:
    p = R / f"restart3_grpofix/{d}/ckpts/{d}_it{it}.pt"
    return Ckpt(run, f"it{it}", p, think, utc(when))


CKPTS = [
    _ei("eiT0", "ei_dan_T0", 1, False),
    _ei("eiT1v1e", "ei_T1v1e", 1, True),
    _ei("eiT0", "ei_dan_T0", 2, False),
    _ei("eiT1v1e", "ei_T1v1e", 2, True),
    _grpo("grpoT0", "grpo_T0", 39, False, "2026-10-07T20:10"),
    _grpo("grpoT1v1e", "grpo_T1v1e", 39, True, "2026-10-07T21:00"),
    _grpo("grpoT0", "grpo_T0", 78, False, "2026-10-07T20:55"),
    _grpo("grpoT1v1e", "grpo_T1v1e", 78, True, "2026-10-07T22:45"),
]


@dataclass(frozen=True)
class Job:
    set: str
    ck: Ckpt

    @property
    def name(self) -> str:
        return f"{self.set}_{self.ck.run}_{self.ck.tag}"

    @property
    def out(self) -> Path:
        return OUT / self.name

    @property
    def budgets(self) -> str:
        return "0,64,256" if self.ck.think else "0"

    def done(self) -> bool:
        try:
            cells = json.loads((self.out / "summary.json").read_text())["cells"]
        except (OSError, ValueError, KeyError):
            return False
        return all(f"plain_B{b}" in cells for b in self.budgets.split(","))

    def cmd(self) -> list[str]:
        files, k, tmo = SETS[self.set]
        return [
            "timeout", str(tmo), str(PY), "-m", "code.thinking.eval",
            "--ckpt", str(self.ck.path), "--sets", files, "--out", str(self.out),
            "--k", str(k), "--budgets", self.budgets, *EVAL_ARGS,
        ]  # fmt: skip


LOCK = threading.Lock()
FINISHED: set[str] = set()
MINE: set[str] = set()  # claimed by this runner
MISSING: set[str] = set()
RUNNING: dict[int, subprocess.Popen[bytes]] = {}
ALL_DONE = threading.Event()


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def log(line: str) -> None:
    with open(LOG, "a") as f:
        f.write(line + "\n")
    say(f"LOG {line}")


def gpu_mib(g: int) -> int:
    q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(g)]
    try:
        return int(subprocess.run(q, capture_output=True, text=True, timeout=60).stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 1 << 30


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def t1320_ready() -> bool:
    """Snapshot the 13-20 set once it exists and has settled (call under LOCK)."""
    if T1320.exists():
        return True
    if not T1320_SRC.exists():
        return False
    if not T1320_KEYS.exists() and time.time() - T1320_SRC.stat().st_mtime < T1320_SETTLE:
        return False
    T1320.parent.mkdir(parents=True, exist_ok=True)
    n_src = sum(1 for line in T1320_SRC.read_text().splitlines() if line.strip())
    tmp = T1320.with_suffix(".tmp")
    if n_src <= T1320_N:
        shutil.copyfile(T1320_SRC, tmp)
        how = "whole set"
    else:
        sub = [str(PY), str(SUBSAMPLE), "--set", str(T1320_SRC), "--out", str(tmp)]
        sub += ["--n", str(T1320_N), "--seed", "0"]
        subprocess.run(sub, check=True, capture_output=True)
        how = f"subsample.py --n {T1320_N} --seed 0 (equal per reference length)"
    n = sum(1 for line in tmp.read_text().splitlines() if line.strip())
    meta = dict(source=str(T1320_SRC), source_mtime=T1320_SRC.stat().st_mtime, n_source=n_src)
    meta |= dict(n=n, how=how, made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    T1320.with_suffix(".source.json").write_text(json.dumps(meta, indent=1))
    os.replace(tmp, T1320)
    log(f"t1320 set: {n} of {n_src} items ({how}) -> {T1320}")
    return True


def jobs() -> list[Job]:
    out = [Job(s, c) for c in CKPTS for s in MAIN_SETS]
    if t1320_ready():
        out += [Job("t1320", c) for c in CKPTS]
    return out


def claim(now: float) -> Job | None:
    """The next runnable job, claimed; None if nothing is runnable now (call under LOCK)."""
    for j in jobs():
        if j.name in FINISHED or j.name in MINE:
            continue
        if j.done():
            FINISHED.add(j.name)
            say(f"{j.name}: summary.json complete, skipped")
            continue
        if (CLAIMS / j.name).exists():
            FINISHED.add(j.name)
            say(f"{j.name}: claimed elsewhere, skipped")
            continue
        if not j.ck.ready(now):
            if now > j.ck.expected + GRACE:
                FINISHED.add(j.name)
                if j.ck.path.name not in MISSING:
                    MISSING.add(j.ck.path.name)
                    log(f"MISSING {j.ck.path} over 3 h past its expected time; its jobs dropped")
            continue
        CLAIMS.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(CLAIMS / j.name)
        except FileExistsError:
            continue
        MINE.add(j.name)
        return j
    return None


def all_finished() -> bool:
    return all(j.name in FINISHED for j in jobs())


def run(g: int, j: Job) -> None:
    t0 = time.time()
    jl = R / f"logs/eval_matched_{j.name}.log"
    (CLAIMS / j.name / "claim.json").write_text(json.dumps(dict(gpu=g, start=t0, log=str(jl))))
    say(f"GPU {g}: start {j.name} (budgets {j.budgets}) -> {jl}")
    env = os.environ | ENV | {"CUDA_VISIBLE_DEVICES": str(g)}
    with open(jl, "w") as f:
        p = subprocess.Popen(j.cmd(), cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        with LOCK:
            RUNNING[g] = p
        rc = p.wait()
    with LOCK:
        RUNNING.pop(g, None)
    secs = round(time.time() - t0)
    lines = [x for x in jl.read_text(errors="replace").splitlines() if x.startswith("[plain")]
    (CLAIMS / j.name / "status.json").write_text(json.dumps(dict(gpu=g, exit=rc, secs=secs)))
    with LOCK:
        FINISHED.add(j.name)
        log(f"{j.name} exit {rc} {secs}s: {' '.join(lines)}")


def worker(g: int, pid: int | None) -> None:
    if pid is not None:
        say(f"GPU {g}: waiting for PID {pid} to exit")
        while alive(pid) and not ALL_DONE.is_set():
            time.sleep(POLL)
        say(f"GPU {g}: PID {pid} gone")
    waited = False
    while not ALL_DONE.is_set():
        with LOCK:
            if all_finished():
                if not ALL_DONE.is_set():
                    log("ALL DONE")
                    ALL_DONE.set()
                return
        mib = gpu_mib(g)
        if mib >= FREE_MIB:
            if not waited:
                say(f"GPU {g}: busy ({mib} MiB), waiting")
            waited = True
            time.sleep(POLL)
            continue
        waited = False
        with LOCK:
            j = claim(time.time())
        if j is None:
            time.sleep(POLL)
            continue
        run(g, j)


def stop(signum: int, _frame: object) -> None:
    procs = list(RUNNING.values())
    log(f"RUNNER STOPPED (signal {signum}; {len(procs)} running eval(s) terminated)")
    for p in procs:
        p.terminate()  # `timeout` forwards it to the eval
    sys.exit(128 + signum)


def children(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def evals_in_flight() -> list[int]:
    out = subprocess.run(["pgrep", "-f", f"code.thinking.eval .*{OUT}/"], capture_output=True)
    return [int(x) for x in out.stdout.split()]


def takeover(old: int) -> None:
    """Kill runner `old` at a moment it has no eval in flight, then clear claims left open."""
    say(f"takeover: waiting for runner {old} to go idle")
    while alive(old):
        os.kill(old, signal.SIGSTOP)
        try:
            if not children(old) and not evals_in_flight():
                os.kill(old, signal.SIGKILL)
                say(f"takeover: runner {old} idle, killed")
                break
        finally:
            os.kill(old, signal.SIGCONT)
        time.sleep(10)
    for _ in range(30):
        if not alive(old):
            break
        time.sleep(1)
    assert not evals_in_flight(), "an eval is in flight after the takeover"
    for d in sorted(CLAIMS.iterdir()) if CLAIMS.exists() else []:
        if d.is_dir() and not (d / "status.json").exists():
            shutil.rmtree(d)
            say(f"takeover: removed open claim {d.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takeover", type=int, default=0, help="PID of the runner to replace")
    a = ap.parse_args()
    if a.takeover:
        takeover(a.takeover)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOCK:
        n = len(jobs())
    say(f"runner PID {os.getpid()}: {n} jobs now, GPUs {sorted(WORKERS)}, log {LOG}")
    threads = [
        threading.Thread(target=worker, args=(g, p), daemon=True) for g, p in WORKERS.items()
    ]
    for t in threads:
        t.start()
        time.sleep(5)  # stagger the first claims and model loads
    while not ALL_DONE.wait(POLL):
        if not any(t.is_alive() for t in threads):
            log("RUNNER STOPPED (every worker exited before ALL DONE)")
            break
    say("runner exiting")


if __name__ == "__main__":
    main()
