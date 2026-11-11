"""Gated launches of the author-tag experiment, one job per GPU on this host. The orchestrator runs these; nothing here
launches by itself.

    python tag_launch.py pretrain --gpu N --seeds 0[,1,2]         [--dry-run]   # (a) the tagged bases
    python tag_launch.py read     --gpu N [--seeds 0,1,2] [--tags A,B,none] [--dry-run]   # (b) base reads + tag check
    python tag_launch.py rl       --gpu N --line tB_s0 [--seg K --resume CKPT] [--dry-run]  # (c) tB_s*, (d) tA_s*
    python tag_launch.py status                                                         # locks, jobs, line views

Every launch, in code, before anything touches the GPU (exit 3 when a rule fails):
  * GPU N on this host has no process of anyone (gpu.py, the knockout's) and under 500 MiB in use;
  * serrano: GPU N's power limit is at most 200 W (the PSU rule);
  * the lab census (serrano, ynez, temescal via gpu.py): after this launch at most 4 GPUs hold a process of ours and
    at least 2 of the lab's GPUs hold no process of anyone; temescal at most 2 of ours. An unreadable host refuses
    the launch unless --allow_unreadable (its GPUs then count as busy, and as not ours);
  * the lock AT/locks/<host>_gpu<N>.lock (mkdir) is free; the job holds it until it exits.
The job is detached (nohup setsid) with its progress log AT/logs/<job>_<UTC>.log (each step's start, end, return code
and seconds) and status AT/logs/<job>_<UTC>.status.json; every step runs under `timeout`; the job stops at the first
step that fails. The watchdog (STALL / EXIT / OVERRUN, think/runs/watchdog.py) is attached when present.
`--dry-run` prints the plan (steps, commands, timeouts, time estimates), the gate's verdict and the census, and
writes nothing.
"""

# ruff: noqa: E501, SIM115
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402
import tag_measure as TM  # noqa: E402

G = TM._ko("gpu")
HOST = os.uname().nodename.split(".")[0]
LAB = ("serrano", "ynez", "temescal")
MAX_OURS, MIN_FREE = 4, 2
MAX_OURS_HOST = {"serrano": 5, "ynez": 4, "temescal": 2}
WATCHDOG = Path("/scratch/cowifornia/nd-rl/think/runs/watchdog.py")
HERE = T.HERE

# Time estimates (hours) from the knockout logs (log.md, numbers.md) and earlier base trainings on these GPUs.
EST = {
    # 24,000 best-recipe steps took 1,489 / 1,701 s on serrano and 967 s on ynez (knockout_grpo base_d1_s*), plus
    # about 1.5 min of loading and 1 min of compilation: 0.45-0.55 h per seed of 24,028-25,607 steps
    "pretrain": 0.55,
    "sample": 0.05,
    # pc read: about 4 min per base on serrano GPU 0 (propensity_capability log, k 32 / k_f 12, 346 theorems) x 3 tags
    "read_gpu_per_base_tag": 0.08,
    "rows": 0.1,
    "check": 0.1,
    # a line: the 1 % lines crossed and held by 1.2-1.9 GPU-h (x 0.35-0.69); the lines that never crossed ran
    # 0.30-0.50 x_dan per GPU-h, so x 1.5 is 3-5 GPU-h
    "rl_cross": 2.0,
    "rl_cap": 5.0,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ the gate
def local_gpu(gpu: int) -> dict:
    q = ["nvidia-smi", "--query-gpu=index,memory.used,power.limit", "--format=csv,noheader,nounits"]
    out = subprocess.run(q, capture_output=True, text=True, timeout=30).stdout
    for line in out.strip().splitlines():
        i, mem, pl = (x.strip() for x in line.split(","))
        if int(i) == gpu:
            return {"mem_mib": int(float(mem)), "power_limit_w": float(pl)}
    raise SystemExit(f"GPU {gpu} not on {HOST}")


def census() -> dict:
    out = {}
    for h in LAB:
        g = G.gpus(h)
        out[h] = None if g is None else {i: sorted(u) for i, u in g.items()}
    return out


def gate(gpu: int, allow_unreadable: bool = False) -> tuple[bool, list[str], dict]:
    """(ok, reasons it is not, the census) for launching one job of ours on GPU `gpu` of this host."""
    why: list[str] = []
    cen = census()
    me = cen.get(HOST)
    if me is None:
        why.append(f"{HOST} itself unreadable")
    elif gpu not in me:
        why.append(f"no GPU {gpu} on {HOST}")
    elif me[gpu]:
        why.append(f"GPU {gpu} has processes of {me[gpu]}")
    if me is not None and gpu in me:
        info = local_gpu(gpu)
        if info["mem_mib"] >= 500:
            why.append(f"GPU {gpu}: {info['mem_mib']} MiB in use")
        if HOST == "serrano" and info["power_limit_w"] > 200:
            why.append(f"serrano GPU {gpu}: power limit {info['power_limit_w']:.0f} W > 200 W")
    ours = free = 0
    ours_host = {}
    for h, g in cen.items():
        if g is None:
            if not allow_unreadable:
                why.append(
                    f"{h} unreadable (pass --allow_unreadable to count its GPUs as busy, not ours)"
                )
            continue
        ours_host[h] = sum(1 for u in g.values() if G.ME in u)
        ours += ours_host[h]
        free += sum(1 for u in g.values() if not u)
    if ours + 1 > MAX_OURS:
        why.append(f"ours would be {ours + 1} > {MAX_OURS}")
    if free - 1 < MIN_FREE:
        why.append(f"lab GPUs free after launch would be {free - 1} < {MIN_FREE}")
    if ours_host.get(HOST, 0) + 1 > MAX_OURS_HOST.get(HOST, 4):
        why.append(f"{HOST}: ours would exceed {MAX_OURS_HOST.get(HOST)}")
    lock = T.AT / "locks" / f"{HOST}_gpu{gpu}.lock"
    if lock.exists():
        why.append(
            f"lock held: {lock} ({(lock / 'pid').read_text() if (lock / 'pid').exists() else '?'})"
        )
    return not why, why, {"census": cen, "ours": ours, "free": free}


# ------------------------------------------------------------------------------------------------ plans
def step(name: str, argv: list[str], hours: float, est: float, env: dict | None = None) -> dict:
    return {
        "name": name,
        "argv": argv,
        "timeout_s": int(hours * 3600),
        "est_h": est,
        "env": env or {},
    }


def plan(a: argparse.Namespace) -> tuple[str, list[dict]]:
    py, h = T.PY, str(HERE)
    if a.cmd == "pretrain":
        seeds = [int(s) for s in a.seeds.split(",")]
        steps = []
        for s in seeds:
            out = T.base_path(s)
            if out.exists():
                raise SystemExit(f"{out} exists: a base is never overwritten")
            steps.append(step(f"pretrain_s{s}", [py, f"{h}/tag_pretrain.py", "--seed", str(s), "--out", str(out)],
                              1.5, EST["pretrain"]))  # fmt: skip
            steps.append(step(f"sample_s{s}", [py, f"{h}/tag_sample.py", "--ckpt", str(out), "--device", "cuda",
                                               "--n", "64", "--k", "4", "--max_lines", "6",
                                               "--out", str(out.with_suffix(".sample.json"))], 0.5, EST["sample"]))  # fmt: skip
        return f"pretrain_s{'-'.join(map(str, seeds))}", steps
    if a.cmd == "read":
        seeds, tags = [int(s) for s in a.seeds.split(",")], a.tags.split(",")
        for s in seeds:
            assert T.base_path(s).exists() or a.dry_run, f"no base {T.base_path(s)}"
        n = len(seeds) * len(tags)
        steps = [
            step("pc_gpu", [py, f"{h}/tag_read.py", "gpu", "--seeds", a.seeds, "--tags", a.tags, "--gpu", str(a.gpu)],
                 max(1.0, 2.5 * n * EST["read_gpu_per_base_tag"]), n * EST["read_gpu_per_base_tag"]),
            step("pc_rows", [py, f"{h}/tag_read.py", "rows", "--seeds", a.seeds, "--tags", a.tags], 1.0, EST["rows"],
                 {"CUDA_VISIBLE_DEVICES": ""}),
        ]  # fmt: skip
        for s in seeds:
            steps.append(step(f"check_s{s}", [py, f"{h}/tag_check.py", "--ckpt", str(T.base_path(s)), "--tags", a.tags,
                                              "--out", str(T.AT / "reads" / "check" / f"tagAB_s{s}.json")],
                              0.75, EST["check"]))  # fmt: skip
        return f"read_s{'-'.join(map(str, seeds))}", steps
    if a.cmd == "rl":
        L = T.LINES[a.line]
        assert T.base_path(L["seed"]).exists() or a.dry_run, f"no base {T.base_path(L['seed'])}"
        argv = [py, f"{h}/tag_line.py", "--line", a.line, "--gpu", str(a.gpu), "--seg", str(a.seg)]
        if a.resume:
            argv += ["--resume", a.resume]
        return f"rl_{a.line}_seg{a.seg}", [step(f"line_{a.line}", argv, 8.0, EST["rl_cap"])]
    raise SystemExit(a.cmd)


# ------------------------------------------------------------------------------------------------ the job
def job_env(gpu: int) -> dict:
    e = dict(os.environ)
    e.update(ND_OFFLINE="1", ND_REGISTRY_SYNC="0", PYTHONDONTWRITEBYTECODE="1", LEAN_GATE_WORKERS="8",
             OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_DEVICE_ORDER="PCI_BUS_ID",
             CUDA_VISIBLE_DEVICES=str(gpu), KD=str(T.KD), AT=str(T.AT))  # fmt: skip
    e.pop(T.ENV, None)  # every step declares its own tag
    return e


def run_job(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    gpu, log, status = spec["gpu"], Path(spec["log"]), Path(spec["status"])
    lock = T.AT / "locks" / f"{HOST}_gpu{gpu}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        with open(log, "a") as f:
            f.write(f"{now()} {msg}\n")

    try:
        os.mkdir(lock)
    except FileExistsError:
        say(f"LOCK held: {lock}; not running")
        return 3
    (lock / "pid").write_text(f"{HOST} {os.getpid()} {spec['job']} {now()}")
    st = {
        "job": spec["job"],
        "host": HOST,
        "gpu": gpu,
        "pid": os.getpid(),
        "start": now(),
        "steps": [],
    }
    rc = 0
    try:
        if not G.free_for_us(HOST, gpu):
            say(f"GATE: GPU {gpu} has another user's process at job start")
            return 3
        for s in spec["steps"]:
            env = {**job_env(gpu), **s["env"]}
            say(f"step {s['name']}: start (timeout {s['timeout_s']} s): {shlex.join(s['argv'])}")
            t0 = time.time()
            with open(log, "a") as f:
                p = subprocess.run(["timeout", "--kill-after=300", str(s["timeout_s"]), *s["argv"]], env=env,
                                   stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))  # fmt: skip
            rc = p.returncode
            st["steps"].append(
                {"name": s["name"], "rc": rc, "secs": round(time.time() - t0, 1), "end": now()}
            )
            status.write_text(json.dumps(st, indent=1))
            say(f"step {s['name']}: end rc {rc} ({'timeout' if rc == 124 else 'ok' if rc == 0 else 'failed'}) "
                f"in {time.time() - t0:.0f} s")  # fmt: skip
            if rc != 0:
                break
    finally:
        st["end"], st["rc"] = now(), rc
        status.write_text(json.dumps(st, indent=1))
        try:
            (lock / "pid").unlink()
            os.rmdir(lock)
        except OSError:
            pass
        say(f"job end rc {rc}; lock released")
    return rc


def launch(a: argparse.Namespace) -> None:
    job, steps = plan(a)
    ok, why, info = gate(a.gpu, a.allow_unreadable)
    est = sum(s["est_h"] for s in steps)
    cap = sum(s["timeout_s"] for s in steps) / 3600
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = T.AT / "logs" / f"{job}_{stamp}.log"
    if a.dry_run:
        print(f"== dry run on {HOST}, GPU {a.gpu}: job {job}; nothing is launched or written ==")
        for s in steps:
            print(f"  step {s['name']}: timeout {s['timeout_s'] / 3600:.2f} h, estimate {s['est_h']:.2f} h"
                  f"{' env ' + json.dumps(s['env']) if s['env'] else ''}\n    {shlex.join(s['argv'])}")  # fmt: skip
        print(f"  total estimate {est:.2f} h (hard cap {cap:.1f} h); log would be {log}")
        print(f"  gate: {'PASS' if ok else 'FAIL'} {why}")
        print(
            f"  census: ours {info['ours']}, lab free {info['free']}: {json.dumps(info['census'])}"
        )
        return
    if not ok:
        print(f"GATE FAIL: {why}")
        sys.exit(3)
    log.parent.mkdir(parents=True, exist_ok=True)
    spec = {"job": job, "gpu": a.gpu, "log": str(log), "status": str(log.with_suffix(".status.json")),
            "steps": steps, "argv": sys.argv, "census": info}  # fmt: skip
    sp = log.with_suffix(".spec.json")
    sp.write_text(json.dumps(spec, indent=1))
    with open(log, "a") as f:
        f.write(f"{now()} launch {job} on {HOST} GPU {a.gpu}; estimate {est:.2f} h, cap {cap:.1f} h; "
                f"census ours {info['ours']} free {info['free']}\n")  # fmt: skip
    p = subprocess.Popen(["nohup", T.PY, str(HERE / "tag_launch.py"), "_job", str(sp)],
                         stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, cwd=str(HERE))  # fmt: skip
    msg = f"job {job} PID {p.pid}, log {log}"
    if WATCHDOG.exists():
        w = subprocess.Popen(["nohup", T.PY, str(WATCHDOG), f"{job}:{p.pid}:{a.gpu}:{log}:{est:.2f}"],
                             stdout=open(T.AT / "logs" / "watchdog.log", "a"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        msg += f"; watchdog PID {w.pid} (AT/logs/watchdog.log)"
    print(msg)


def status() -> None:
    for lk in sorted((T.AT / "locks").glob("*.lock")):
        print("lock", lk.name, (lk / "pid").read_text() if (lk / "pid").exists() else "?")
    for sp in sorted((T.AT / "logs").glob("*.status.json"))[-10:]:
        s = json.loads(sp.read_text())
        print(sp.name, s.get("rc"), [(x["name"], x["rc"], x["secs"]) for x in s["steps"]])
    for line in T.LINES:
        if TM.segs_of(line):
            v = TM.line_view(line)
            print(f"{line}: it {v['last_iter']} x_dan {v['x_dan']:.3f} measure {[m['cum'] for m in v['matched']]} "
                  f"cross {v['cross_dan']} stop {TM.stop_reason(v)}")  # fmt: skip


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "_job":
        sys.exit(run_job(Path(sys.argv[2])))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("pretrain", "read", "rl", "status"))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument("--line", choices=sorted(T.LINES), default=None)
    ap.add_argument("--seg", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--allow_unreadable", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
        return
    assert a.gpu is not None, "--gpu N (nvidia-smi index on this host)"
    assert a.cmd != "rl" or a.line, "--line"
    launch(a)


if __name__ == "__main__":
    main()
