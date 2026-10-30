#!/usr/bin/env python3
"""The high-RL pass@k queue on ynez (2026-10-30, agent:claude for Leon): one worker per GPU on a
shared job file, GPU-gated, every job under a hard timeout and watched by ../watchdog.py.

    python3 hirl_queue.py worker GPU [--slots N] [--free_for MIN]   one worker for GPU (setsid)
    python3 hirl_queue.py status                                     print STATUS.md (rewritten)
    python3 hirl_queue.py merge GROUP                                merge a finished chunk group

Layout (H = /scratch/cowifornia/nd-rl/think_hirl):
  queue/jobs.jsonl   one job per line, in priority order (hirl_plan.py writes it). Edit it only
                     under `flock queue/jobs.lock`. A job is taken when its state file is created.
  queue/state/ID.json  running | done | failed (gpu, pids, start, end, exit code, wall seconds)
  queue/slots_gpuN   optional: the number of concurrent jobs on GPU N (read every loop; default
                     --slots). A train job always runs alone on its GPU.
  queue/STOP, queue/STOP_gpuN   no new launches (all workers / GPU N); a worker exits when idle
  locks/gpuN         mkdir lock: one worker per GPU (holds the worker's pid)
  logs/ID.log        each job's output; logs/worker_gpuN.log; logs/watchdog.log (STALL/EXIT/OVERRUN)
  eval/chunks/GROUP/seedS   one eval.py call; eval/final/GROUP: the merged read (or a single read)

GPU gate, checked right before every launch (and for a GPU's first launch, continuously for
--free_for minutes): no compute process that is not one of this worker's jobs, and the memory in
use, less what this worker's own processes hold, under 500 MiB. A GPU with another user's process
is never launched on; the worker waits (polling every 20 s) and never touches other processes.
Outputs are never overwritten: a read whose out dir has its rows file is recorded done; a partial
one (a crash) is moved aside to OUT.crashed_<UTC> first. Reads are eval.py with line.sh's read1
flags (plain, B64, T 0.8, --batch 1024 --max_action 512 --max_steps 96 --max_rej 10).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

H = Path("/scratch/cowifornia/nd-rl/think_hirl")
WT = H / "nd-rl-hirl"  # the code worktree (branch leon/high-rl-reads)
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"  # read-only use of the venv
WATCHDOG = WT / "code/experiments/current/thinking_traces_20261005/grpo_v2/launch/watchdog.py"
Q = H / "queue"
JOBS, LOCK, ST = Q / "jobs.jsonl", Q / "jobs.lock", Q / "state"
LOGS = H / "logs"
FINAL, CHUNKS = H / "eval/final", H / "eval/chunks"
ENV = {
    "ND_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ND_REGISTRY_SYNC": "0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ND_LEAN": str(Path.home() / ".elan/bin/lean"),
    "TMPDIR": str(H / "tmp"),
    "LEAN_GATE_WORKERS": "16",
}
READ_FLAGS = ["--budgets", "64", "--arms", "plain", "--temperature", "0.8"]
READ_FLAGS += ["--max_action", "512", "--max_steps", "96", "--max_rej", "10"]
MEM_MAX = 500  # MiB
POLL = 20


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(msg: str) -> None:
    print(f"{utc()} {msg}", flush=True)


class Locked:
    """flock on queue/jobs.lock (every read-modify-write of the queue's files)."""

    def __enter__(self) -> Locked:
        self.fh = open(LOCK, "a")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def jobs() -> list[dict[str, Any]]:
    out = []
    for line in JOBS.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            out.append(json.loads(line))
    ids = [j["id"] for j in out]
    assert len(ids) == len(set(ids)), "duplicate job ids in jobs.jsonl"
    return out


def state(jid: str) -> dict[str, Any] | None:
    p = ST / f"{jid}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_state(jid: str, st: dict[str, Any], create: bool = False) -> bool:
    p = ST / f"{jid}.json"
    if create:
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(st) + "\n")
        return True
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st) + "\n")
    os.replace(tmp, p)
    return True


def out_dir(j: dict[str, Any]) -> Path:
    return Path(j["out"])


def rows_file(j: dict[str, Any]) -> Path:
    return out_dir(j) / "rows_plain_B64.jsonl"


def missing_needs(j: dict[str, Any]) -> list[str]:
    """The job's input files that are not there yet, and the jobs it runs after that are not
    done (a train job's checkpoints are read once that job is done: save_ckpt is not atomic)."""
    miss = [p for p in j.get("needs", []) if not Path(p).exists()]
    return miss + [x for x in j.get("after", []) if (state(x) or {}).get("status") != "done"]


# ---------------------------------------------------------------- the GPU gate
def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def gpu_procs(g: int) -> tuple[int, list[tuple[int, int]]] | None:
    """(memory used MiB, [(pid, used MiB)] of its compute processes), None if nvidia-smi fails."""
    try:
        m = sh(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(g)]
        ).strip()
        a = sh(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(g),
            ]
        )
        procs = []
        for line in a.splitlines():
            if line.strip():
                p, u = (x.strip() for x in line.split(","))
                procs.append((int(p), int(u) if u.isdigit() else 0))
        return int(m), procs
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def descendants(roots: list[int]) -> set[int]:
    kids: dict[int, list[int]] = {}
    for line in sh(["ps", "-eo", "pid=,ppid="]).splitlines():
        p, pp = line.split()
        kids.setdefault(int(pp), []).append(int(p))
    out, stack = set(), list(roots)
    while stack:
        q = stack.pop()
        if q not in out:
            out.add(q)
            stack += kids.get(q, [])
    return out


def gate(g: int, ours: list[int]) -> tuple[bool, str]:
    """True when GPU g has no compute process but ours and < 500 MiB besides ours."""
    r = gpu_procs(g)
    if r is None:
        return False, f"GPU {g}: nvidia-smi failed"
    mem, procs = r
    mine = descendants(ours) if ours else set()
    foreign = [p for p, _ in procs if p not in mine]
    own_mem = sum(u for p, u in procs if p in mine)
    if foreign:
        return False, f"GPU {g}: compute processes not ours ({' '.join(map(str, foreign))})"
    if mem - own_mem >= MEM_MAX:
        return False, f"GPU {g}: {mem} MiB used, {own_mem} ours (>= {MEM_MAX} MiB besides ours)"
    return True, f"GPU {g}: free ({mem} MiB used, {own_mem} ours)"


# ---------------------------------------------------------------- jobs
def read_cmd(j: dict[str, Any], g: int, batch: int = 1024) -> list[str]:
    return [
        "timeout",
        "--kill-after=120",
        str(int(j["timeout_s"])),
        PY,
        "-m",
        "code.thinking.eval",
        "--ckpt",
        j["ckpt"],
        "--sets",
        j["sets"],
        "--out",
        j["out"],
        "--k",
        str(j["k"]),
        "--batch",
        str(batch),
        *READ_FLAGS,
        "--seed",
        str(j["seed"]),
    ]


def train_cmd(j: dict[str, Any], g: int) -> list[str]:
    return [
        "timeout",
        "--kill-after=120",
        str(int(j["timeout_s"])),
        PY,
        "-m",
        "code.thinking.grpo",
        *j["argv"],
    ]


def prepare(j: dict[str, Any]) -> str | None:
    """Before a launch: None to run; else why the job is already done (its output exists)."""
    o = out_dir(j)
    if j["kind"] == "read" and rows_file(j).exists():
        return f"{rows_file(j)} exists"
    if j["kind"] == "train" and Path(j["last_ckpt"]).exists():
        return f"{j['last_ckpt']} exists"
    if o.exists():  # a partial earlier run of ours: aside, never removed
        t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        o.rename(o.with_name(o.name + f".crashed_{t}"))
        say(f"{j['id']}: partial {o} moved aside to {o.name}.crashed_{t}")
    return None


def exclusive(j: dict[str, Any]) -> bool:
    """A job that runs alone on its GPU, taking all its slots: a train job, and a read at
    --batch 1024 (one that is not flex_batch: J3, the reproduction chunks). Two batch-1024 reads
    need about 2 x 12 GB and run out of memory on a 24 GB 3090 (2026-10-30, 20:55Z and 22:28Z).
    A reuse job (an earlier read's output, recorded done without a GPU) is not exclusive."""
    if j.get("reuse"):
        return False
    return j["kind"] == "train" or (j["kind"] == "read" and not j.get("flex_batch"))


def finished_ok(j: dict[str, Any]) -> bool:
    if j["kind"] == "read":
        return rows_file(j).exists() and (out_dir(j) / "summary.json").exists()
    return Path(j["last_ckpt"]).exists()


# ---------------------------------------------------------------- merging chunk groups
def pass_at_k(n: int, c: int, k: int) -> float:  # eval.pass_at_k
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def ks_upto(n: int) -> list[int]:  # eval.ks_upto
    out, k = [], 1
    while k <= n:
        out.append(k)
        k *= 2
    if out[-1] != n:
        out.append(n)
    return out


COUNTERS = ("tokens", "think", "close", "action", "rejected", "prefill", "draws", "rej", "forced")
LISTS = ("ok", "status", "steps", "fail_step", "budget", *COUNTERS)


def merge_group(group: str, members: list[dict[str, Any]]) -> Path:
    """The chunks' rows, per theorem in chunk order (seed order), as one read with k = the sum.
    A chunk with `keep` (a names file) contributes only those theorems (its extra rows dropped).
    Written to eval/final/GROUP (refused if it exists): rows_plain_B64.jsonl, summary.json
    (eval.summarize's 'all' cell for plain_B64, recomputed), chunks.json (provenance)."""
    dst = FINAL / group
    assert not dst.exists(), f"{dst} exists"
    members = sorted(members, key=lambda j: j["chunk"])
    names: list[str] | None = None
    merged: dict[str, dict[str, Any]] = {}
    prov = []
    for j in members:
        keep = set(Path(j["keep"]).read_text().split()) if j.get("keep") else None
        rows = [json.loads(x) for x in rows_file(j).read_text().splitlines()]
        if keep is not None:
            rows = [r for r in rows if r["name"] in keep]
        these = sorted(r["name"] for r in rows)
        if names is None:
            names = these
        assert these == names, f"{group}: chunk {j['chunk']} has other theorems"
        args = json.loads((out_dir(j) / "args.json").read_text())
        prov.append(
            {
                "chunk": j["chunk"],
                "seed": j["seed"],
                "k": j["k"],
                "dir": j["out"],
                "sets": j["sets"],
                "keep": j.get("keep"),
                "git_sha": args.get("_meta", {}).get("git_sha"),
                "host": args.get("_meta", {}).get("host"),
                "utc": args.get("_meta", {}).get("utc"),
            }
        )
        for r in rows:
            assert r["k"] == j["k"] == len(r["ok"]), (group, j["id"], r["name"])
            m = merged.get(r["name"])
            if m is None:
                m = merged[r["name"]] = {x: r[x] for x in ("file", "name", "prompt", "min_lines")}
                m.update(
                    k=0,
                    n_ok=0,
                    intervene=r.get("intervene", "none"),
                    causes=[],
                    accepted={},
                    **{x: [] for x in LISTS},
                )
                m["total"] = {c: 0 for c in COUNTERS}
            m["k"] += r["k"]
            m["n_ok"] += r["n_ok"]
            for x in LISTS:
                m[x] += r[x]
            for c in COUNTERS:
                m["total"][c] += r["total"][c]
            m["causes"] += r["causes"]
            for t, n in r["accepted"].items():
                m["accepted"][t] = m["accepted"].get(t, 0) + n
    assert names is not None
    last = {
        r["name"]: r
        for r in (json.loads(x) for x in rows_file(members[-1]).read_text().splitlines())
    }
    for nm, m in merged.items():  # file / prompt fields as the last chunk's set has them
        if nm in last:
            m["file"], m["prompt"] = last[nm]["file"], last[nm]["prompt"]
    k = sum(j["k"] for j in members)
    order = [
        r["name"] for r in (json.loads(x) for x in rows_file(members[-1]).read_text().splitlines())
    ]
    order = [n for n in order if n in merged] + sorted(set(merged) - set(order))
    out_rows = [merged[n] for n in order]
    assert all(r["k"] == k for r in out_rows)
    n = len(out_rows)
    tot = {c: sum(r["total"][c] for r in out_rows) for c in COUNTERS}
    cell = {
        "n": n,
        "solved": sum(r["n_ok"] > 0 for r in out_rows),
        "solved_pct": 100.0 * sum(r["n_ok"] > 0 for r in out_rows) / n,
        "pass_at_k_pct": {
            str(i): 100.0 * sum(pass_at_k(k, r["n_ok"], i) for r in out_rows) / n
            for i in ks_upto(k)
        },
        "tokens_total": tot,
        "tokens_per_theorem": {c: v / n for c, v in tot.items()},
        "tokens_per_attempt": {c: v / (n * k) for c, v in tot.items()},
    }
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / "rows_plain_B64.jsonl", "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "merged_from_chunks": True,
        "ckpt": members[0]["ckpt"],
        "k": k,
        "n_theorems": n,
        "config": {
            "read_flags": READ_FLAGS,
            "chunk_k": [j["k"] for j in members],
            "seeds": [j["seed"] for j in members],
        },
        "cells": {"plain_B64": {"all": cell}},
    }
    (tmp / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (tmp / "chunks.json").write_text(json.dumps(prov, indent=1) + "\n")
    os.replace(tmp, dst)
    return dst


def mark_ready(name: str) -> None:
    """eval/ready.txt: the finished reads in eval/final (merged or single), one name a line, for
    hirl_sync.sh (serrano pulls them into runs/eval/v2, never over an existing dir)."""
    with open(H / "eval/ready.txt", "a") as fh:
        fh.write(name + "\n")


def maybe_merge(j: dict[str, Any], all_jobs: list[dict[str, Any]]) -> None:
    g = j.get("group")
    if not g or j["kind"] != "read" or (FINAL / g).exists():
        return
    mem = [x for x in all_jobs if x.get("group") == g]
    if len(mem) != mem[0]["nchunks"]:
        return
    if all((state(x["id"]) or {}).get("status") == "done" for x in mem):
        p = merge_group(g, mem)
        mark_ready(g)
        c = json.loads((p / "summary.json").read_text())["cells"]["plain_B64"]["all"]
        pk = c["pass_at_k_pct"]
        say(
            f"merged {g}: n {c['n']}, pass@1 {pk['1']:.1f}%, pass@{max(map(int, pk))} "
            f"{pk[str(max(map(int, pk)))]:.1f}%"
        )


# ---------------------------------------------------------------- status
def fmt_pt(t: float) -> str:
    return datetime.fromtimestamp(t, ZoneInfo("America/Los_Angeles")).strftime("%a %-I:%M %p PT")


def est_minutes(j: dict[str, Any], done: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    """The job's estimate, scaled by the observed / estimated wall time of finished jobs of its
    class (same set and model class) on any GPU."""
    same = [(x, s) for x, s in done if x.get("cls") == j.get("cls") and s.get("wall_s")]
    if not same:  # no finished job of its class yet: all finished reads (or trains)
        same = [(x, s) for x, s in done if x["kind"] == j["kind"] and s.get("wall_s")]
    if same:
        ratio = sum(s["wall_s"] / 60 for _, s in same) / sum(x["est_min"] for x, _ in same)
        return j["est_min"] * ratio
    return float(j["est_min"])


_PK_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def read_passk(rows: Path) -> dict[str, Any]:
    """pass@1, @5, @64, @256 and @k (as % of the set) of a rows file, cached by mtime."""
    m = rows.stat().st_mtime
    hit = _PK_CACHE.get(str(rows))
    if hit and hit[0] == m:
        return dict(hit[1])
    rs = [json.loads(x) for x in rows.read_text().splitlines() if x.strip()]
    k = len(rs[0]["ok"])
    out: dict[str, Any] = {"n": len(rs)}
    for x in dict.fromkeys(i for i in (1, 5, 64, 256, k) if i <= k):
        out[str(x)] = 100.0 * sum(pass_at_k(len(r["ok"]), sum(r["ok"]), x) for r in rs) / len(rs)
    _PK_CACHE[str(rows)] = (m, out)
    return dict(out)


def write_status() -> str:
    js = jobs()
    sts = {j["id"]: state(j["id"]) for j in js}
    done = [(j, sts[j["id"]]) for j in js if (sts[j["id"]] or {}).get("status") == "done"]
    running = [(j, sts[j["id"]]) for j in js if (sts[j["id"]] or {}).get("status") == "running"]
    failed = [(j, sts[j["id"]]) for j in js if (sts[j["id"]] or {}).get("status") == "failed"]
    pending = [j for j in js if sts[j["id"]] is None]
    now = time.time()
    workers = []
    for p in sorted((H / "locks").glob("gpu*")):
        try:
            pid = int((p / "pid").read_text().split()[0])
            alive = Path(f"/proc/{pid}").exists()
        except (OSError, ValueError, IndexError):
            pid, alive = -1, False
        slots = Q / f"slots_{p.name}"
        workers.append((p.name, pid, alive, slots.read_text().strip() if slots.exists() else "-"))
    n_slots = max(
        1, sum(int(s) if s.isdigit() else 1 for _, _, a, s in workers if a)
    )  # concurrent jobs: per-job wall times are measured at this concurrency
    per_gpu = n_slots / max(1, sum(1 for _, _, a, _ in workers if a))

    def w(j: dict[str, Any]) -> float:  # an exclusive job holds all of its GPU's slots
        return per_gpu if exclusive(j) else 1.0

    rem_run = sum(
        w(j) * max(0.0, est_minutes(j, done) - (now - s["start"]) / 60) for j, s in running
    )
    rem_pend = sum(w(j) * est_minutes(j, done) for j in pending)
    eta = now + 60 * (rem_run + rem_pend) / n_slots
    # each block's rough finish: the queue's remaining work in order, over the slots
    cum, block_eta = rem_run, {}
    for j in pending:
        cum += w(j) * est_minutes(j, done)
        block_eta[j.get("block", "?")] = now + 60 * cum / n_slots
    lines = [
        "# think_hirl queue status",
        "",
        f"Updated {utc()} ({fmt_pt(now)}) by the queue. Jobs: {len(done)} done, {len(running)} "
        f"running, {len(pending)} pending, {len(failed)} failed.",
        "",
        "Workers (lock, pid, alive, slots file): "
        + "; ".join(f"{n} pid {p} {'alive' if a else 'GONE'} slots {s}" for n, p, a, s in workers),
        "",
        f"Rough ETA for the whole queue at the current worker count: {fmt_pt(eta)} "
        f"({(rem_run + rem_pend) / 60:.1f} GPU-slot-hours left; estimates scaled by finished jobs "
        "of the same class).",
        "",
        "Blocks (pending work in queue order over the current slots; rough): "
        + "; ".join(f"{b} done by ~{fmt_pt(t)}" for b, t in block_eta.items()),
        "",
        "## Running",
    ]
    for j, s in running:
        lines.append(
            f"- {j['id']} on GPU {s['gpu']} since {fmt_pt(s['start'])} "
            f"(est {est_minutes(j, done):.0f} min, timeout {j['timeout_s'] // 60} min)"
        )
    lines += ["", "## By group (J = priority block)"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for j in js:
        groups.setdefault(j.get("group") or j["id"], []).append(j)
    for g, mem in groups.items():
        st = [(sts[x["id"]] or {}).get("status", "pending") for x in mem]
        fin = st.count("done") == len(mem) and (
            mem[0]["kind"] != "read" or (FINAL / g / "rows_plain_B64.jsonl").exists()
        )
        tag = "DONE" if fin else f"{st.count('done')}/{len(mem)} done"
        if st.count("failed"):
            tag += f", {st.count('failed')} FAILED"
        extra = ""
        if mem[0]["kind"] == "read" and (FINAL / g / "rows_plain_B64.jsonl").exists() and fin:
            pk = read_passk(FINAL / g / "rows_plain_B64.jsonl")
            extra = (
                " — n "
                + str(pk.pop("n"))
                + ": "
                + ", ".join(f"pass@{x} {v:.1f}%" for x, v in pk.items())
            )
        lines.append(f"- [{mem[0].get('block', '?')}] {g}: {tag}{extra}")
    if failed:
        lines += ["", "## Failed (state files in queue/state; delete one to requeue)"]
        for j, s in failed:
            lines.append(
                f"- {j['id']} exit {s.get('rc')} on GPU {s.get('gpu')}: {s.get('why', '')}"
            )
    wl = LOGS / "watchdog.log"
    if wl.exists():
        alerts = [
            x for x in wl.read_text().splitlines() if any(w in x for w in ("STALL", "OVERRUN"))
        ]
        exits = [x for x in wl.read_text().splitlines() if "EXIT" in x]
        lines += [
            "",
            "## Watchdog (logs/watchdog.log)",
            f"STALL / OVERRUN lines: {len(alerts)}; EXIT lines: {len(exits)}.",
        ]
        lines += [f"- {x}" for x in alerts[-10:]]
    notes = H / "STATUS_notes.md"  # findings written by hand (agent), shown on every rewrite
    if notes.exists():
        lines += ["", notes.read_text().rstrip()]
    lines += [
        "",
        "Logs: logs/<job>.log, logs/worker_gpuN.log, logs/watchdog.log. Results: "
        "eval/final/<read>/ (rows_plain_B64.jsonl, summary.json, chunks.json). Low-RL "
        "compute: lowrl/compute.tsv.",
        "",
    ]
    text = "\n".join(lines)
    tmp = H / "STATUS.md.tmp"
    tmp.write_text(text)
    os.replace(tmp, H / "STATUS.md")
    return text


# ---------------------------------------------------------------- the worker
def claim_gpu(g: int) -> Path:
    lk = H / "locks" / f"gpu{g}"
    try:
        lk.mkdir()
    except FileExistsError:
        try:
            pid = int((lk / "pid").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"GPU {g} is claimed by worker pid {pid} ({lk}): refusing") from None
        raise SystemExit(
            f"stale lock {lk} (pid {pid}): check its worker log, then rmdir it"
        ) from None
    (lk / "pid").write_text(f"{os.getpid()} {utc()}\n")
    return lk


def worker(g: int, slots0: int, free_for: float) -> None:
    lk = claim_gpu(g)
    say(f"worker GPU {g}: start (pid {os.getpid()}, slots {slots0}, free_for {free_for} min)")
    running: dict[str, tuple[subprocess.Popen[bytes], dict[str, Any], Any]] = {}
    # jobs an earlier worker of this GPU launched and left running (a restart): polled by pid
    adopted: dict[str, tuple[int, dict[str, Any]]] = {}
    with Locked():
        for j in jobs():
            st = state(j["id"]) or {}
            here = st.get("host", os.uname().nodename) == os.uname().nodename
            if st.get("status") == "running" and st.get("gpu") == g and st.get("pid") and here:
                adopted[j["id"]] = (int(st["pid"]), j)
                say(f"worker GPU {g}: adopting {j['id']} (pid {st['pid']})")
    free_since: float | None = None
    last_status = 0.0
    stop_seen = False

    def slots() -> int:
        p = Q / f"slots_gpu{g}"
        try:
            return max(1, int(p.read_text().strip()))
        except (OSError, ValueError):
            return slots0

    def finish(jid: str, j: dict[str, Any], rc: int | None) -> None:
        st = state(jid) or {}
        ok = (rc == 0 or rc is None) and finished_ok(j)
        st.update(
            status="done" if ok else "failed",
            rc=rc,
            end=time.time(),
            wall_s=time.time() - st.get("start", time.time()),
        )
        if not ok:
            st["why"] = (
                "exit 124: hard timeout"
                if rc == 124
                else (f"exit {rc}" + ("" if finished_ok(j) else ", no output"))
            )
        tries_f = ST / f"{jid}.tries"
        tries = int(tries_f.read_text()) if tries_f.exists() else 0
        with Locked():
            if not ok and rc != 124 and tries < 1:  # one automatic retry (e.g. a CUDA OOM)
                tries_f.write_text(f"{tries + 1}\n")
                (ST / f"{jid}.json").rename(ST / f"{jid}.failed{tries + 1}.json")
                say(f"{jid}: FAILED exit {rc} ({st['why']}): requeued for one retry")
                return
            write_state(jid, st)
        say(
            f"{jid}: {'done' if ok else 'FAILED'} exit {rc} in {st['wall_s'] / 60:.1f} min"
            + (f": {tail(LOGS / (jid + '.log'))}" if j["kind"] == "read" else "")
        )
        if ok and j["kind"] == "read" and not j.get("group"):
            mark_ready(Path(j["out"]).name)  # a single read, written to eval/final itself
        if ok:
            try:
                with Locked():
                    maybe_merge(j, jobs())
            except Exception as e:  # a merge problem must not stop the queue
                say(f"merge after {jid} failed: {e!r}")

    def reap() -> bool:
        changed = False
        for jid, (pid, j) in list(adopted.items()):
            if not Path(f"/proc/{pid}").exists():
                del adopted[jid]
                changed = True
                finish(jid, j, None)
        for jid, (proc, j, logf) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            del running[jid]
            changed = True
            finish(jid, j, rc)
        return changed

    def next_job() -> dict[str, Any] | None:
        """The first job with no state file whose inputs exist (called under the lock)."""
        for j in jobs():
            if state(j["id"]) is None and not missing_needs(j):
                if j.get("gpus") and g not in j["gpus"]:
                    continue
                return j
        return None

    def blocked_forever() -> bool:
        """Every pending job waits for inputs that no pending or running train job will make."""
        js = jobs()
        pend = [j for j in js if state(j["id"]) is None]
        if not pend:
            return True
        trains = [
            j
            for j in js
            if j["kind"] == "train"
            and (state(j["id"]) is None or (state(j["id"]) or {}).get("status") == "running")
        ]
        return all(missing_needs(j) for j in pend) and not trains

    try:
        while True:
            if reap():
                last_status = 0.0
            now = time.time()
            if now - last_status > 300:
                with Locked():
                    write_status()
                last_status = now
            stop = (Q / "STOP").exists() or (Q / f"STOP_gpu{g}").exists()
            if stop and not stop_seen:
                say(f"worker GPU {g}: STOP file seen: no new launches")
            stop_seen = stop
            if stop:
                if not running and not adopted:
                    break
                time.sleep(POLL)
                continue
            ours = [p.pid for p, _, _ in running.values()] + [p for p, _ in adopted.values()]
            n_run = len(running) + len(adopted)
            ok, why = gate(g, ours)
            if not ok:
                if free_since is not None:
                    say(f"worker GPU {g}: gate closed: {why}")
                free_since = None
                time.sleep(POLL)
                continue
            if free_since is None:
                free_since = now
                say(f"worker GPU {g}: gate open: {why}")
            if not n_run and now - free_since < 60 * free_for:
                time.sleep(POLL)  # a GPU's first launch: free for --free_for minutes first
                continue
            # nothing starts next to an exclusive job, nor next to any read that was launched at
            # --batch 1024 (a flex chunk started while the GPU had one slot)
            here = [j for _, j, _ in running.values()] + [j for _, j in adopted.values()]
            excl = any(
                exclusive(j)
                or (j["kind"] == "read" and (state(j["id"]) or {}).get("batch") == 1024)
                for j in here
            )
            if excl or n_run >= slots():
                time.sleep(POLL)
                continue
            launched = False
            with Locked():
                j = next_job()
                if j is not None and exclusive(j) and n_run and not j.get("reuse"):
                    j = None  # it runs alone (exclusive): wait for this GPU's jobs to end
                if j is not None:
                    ok, why = gate(g, ours)  # again, right before the launch
                    if ok and j.get("reuse") and not finished_ok(j):
                        # a reuse job never runs eval and never moves its (another job's) dir
                        write_state(
                            j["id"],
                            {
                                "status": "failed",
                                "gpu": g,
                                "why": f"reuse: no finished read in {j['out']}",
                            },
                            create=True,
                        )
                        say(f"{j['id']}: FAILED: reuse source {j['out']} not finished")
                        j, ok = None, False
                    if ok and j is not None:
                        done_why = prepare(j)
                        if done_why:
                            write_state(
                                j["id"],
                                {
                                    "status": "done",
                                    "gpu": g,
                                    "start": now,
                                    "end": now,
                                    "wall_s": 0,
                                    "why": done_why,
                                },
                                create=True,
                            )
                            say(f"{j['id']}: already done ({done_why})")
                            try:  # (a reused chunk can be a group's last)
                                maybe_merge(j, jobs())
                            except Exception as e:
                                say(f"merge after {j['id']} failed: {e!r}")
                        else:
                            if not write_state(
                                j["id"],
                                {
                                    "status": "running",
                                    "gpu": g,
                                    "start": time.time(),
                                    "host": os.uname().nodename,
                                },
                                create=True,
                            ):
                                j = None
                            if j is not None:
                                launched = True
            if j is None and not n_run:
                with Locked():
                    if blocked_forever():
                        say(f"worker GPU {g}: nothing left to run")
                        break
                time.sleep(POLL)
                continue
            if not launched:
                time.sleep(2 if j is not None else POLL)
                continue
            log = LOGS / f"{j['id']}.log"
            logf = open(log, "ab")  # noqa: SIM115 (open while the job runs; closed by reap)
            # two reads on a GPU at --batch 1024 run out of memory (24 GB): a chunk that may
            # (flex_batch) runs at 512 while the GPU's slots are 2 or more
            batch = 512 if j.get("flex_batch") and slots() >= 2 else 1024
            cmd = read_cmd(j, g, batch) if j["kind"] == "read" else train_cmd(j, g)
            env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(g)}
            logf.write(f"{utc()} GPU {g} cwd {WT}\n$ {' '.join(cmd)}\n".encode())
            logf.flush()
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=WT,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except OSError as e:
                logf.close()
                with Locked():
                    write_state(j["id"], {"status": "failed", "gpu": g, "why": f"launch: {e!r}"})
                say(f"{j['id']}: launch failed: {e!r}")
                continue
            running[j["id"]] = (proc, j, logf)
            with Locked():
                st = state(j["id"]) or {}
                st.update(pid=proc.pid, cmd=cmd, batch=batch if j["kind"] == "read" else None)
                write_state(j["id"], st)
            wl = open(LOGS / "watchdog.log", "a")  # noqa: SIM115 (handed to the watchdog)
            subprocess.Popen(
                [
                    sys.executable,
                    str(WATCHDOG),
                    f"{j['id']}:{proc.pid}:{g}:{log}:{max(0.05, j['est_min'] / 60):.3f}",
                ],
                stdout=wl,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            wl.close()
            say(
                f"{j['id']}: launched on GPU {g} (pid {proc.pid}, est {j['est_min']} min, "
                f"timeout {j['timeout_s'] // 60} min, batch {batch}; {len(running)} running here)"
            )
            time.sleep(5)
    finally:
        for jid, (proc, _, _) in running.items():  # an exception: let the jobs finish alone
            say(
                f"worker GPU {g}: exiting with {jid} still running (pid {proc.pid}; adopted by "
                "the next worker of this GPU)"
            )
        with Locked():
            write_status()
        (lk / "pid").unlink(missing_ok=True)
        lk.rmdir()
        say(f"worker GPU {g}: done")


def tail(p: Path) -> str:
    try:
        xs = [x for x in p.read_text(errors="replace").splitlines() if x.startswith("[")]
        return " ".join(xs)[:200]
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("gpu", type=int)
    w.add_argument("--slots", type=int, default=1)
    w.add_argument("--free_for", type=float, default=0.0, help="minutes free before a first job")
    sub.add_parser("status")
    m = sub.add_parser("merge")
    m.add_argument("group")
    a = ap.parse_args()
    for d in (ST, LOGS, FINAL, CHUNKS, H / "locks", H / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    if a.cmd == "worker":
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # runs the cleanup; jobs go on
        worker(a.gpu, a.slots, a.free_for)
    elif a.cmd == "status":
        with Locked():
            print(write_status())
    elif a.cmd == "merge":
        with Locked():
            js = jobs()
            print(merge_group(a.group, [j for j in js if j.get("group") == a.group]))


if __name__ == "__main__":
    main()
