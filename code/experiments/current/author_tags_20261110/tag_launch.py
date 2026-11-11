"""Gated launches of the author-tag experiment, one job per GPU on this host. The orchestrator runs these; nothing here
launches by itself.

    python tag_launch.py pretrain --gpu N --seeds 0[,1,2] [--variant p10|p50|p100] [--dry-run]   # (a) tagged bases
    python tag_launch.py read     --gpu N [--seeds 0,1,2] [--tags A,B,none] [--variant pNN] [--dry-run]  # (b) base reads + tag check
    python tag_launch.py rl       --gpu N --line tB_s0 [--seg K --resume CKPT] [--dry-run]  # (c) tB_s*, (d) tA_s*
                                         (variant lines: --line tB_p10_s0, tA_p10_s1, ...)
    python tag_launch.py check    --gpu N --bases tagAB_p10_s0[,...] [--tags A,B,none] [--dry-run]  # tag_check only
    python tag_launch.py status                                                         # locks, jobs, line views

`pretrain --variant pNN` trains on a tie-strength variant's corpus (tagcorpus.py --route-share) and writes
AT/bases/stage1_tagAB_<variant>_s<seed>.pt (+ .tag.json), same recipe and step count as the default bases; without
--variant, the default corpus and names. `check --bases` runs only tag_check.py on each named base
(`tagAB[_<variant>]_s<seed>`, the checkpoint's stem without `stage1_`) on the corpus it was trained on and writes
AT/reads/check/<base>.json; `tag_gate.py` reads those. Neither ever overwrites a base or a check.

`read [--variant pNN]` reads the bases `tagAB[_<variant>]_s<seed>` under each tag: tag_read.py gpu, then rows (CPU),
into tag_core.read_root(variant) (AT/reads/pc for the default bases, AT/reads/pc_<variant> for a variant's), then
tag_check.py on each base that has no AT/reads/check/<base>.json yet (a base that has one is skipped, never re-run).
It refuses, before the gate, a missing base and a read whose results already exist. `rl --line` takes the default lines
(tB_s0 ... tA_s2) and the variant lines t<A|B>_<variant>_s<seed> (tag_core.LINES); it refuses, before the gate, a line
whose base or .tag.json is missing or of another variant, a segment folder that already exists, a later segment
without --resume, and a --resume that is not one of this line's own rolling/alarm checkpoints of an earlier segment.

Every launch, in code, before anything touches the GPU (exit 3 when a rule fails):
  * GPU N on this host has no process of anyone (gpu.py, the knockout's) and under 500 MiB in use;
  * serrano: GPU N's power limit is at most 200 W (the PSU rule);
  * the lab census (serrano, ynez, temescal via gpu.py): after this launch at most 4 GPUs hold a process of ours and
    at least 2 of the lab's GPUs hold no process of anyone; temescal at most 2 of ours. An unreadable host refuses
    the launch unless --allow_unreadable (its GPUs then count as busy, and as not ours);
  * the lock AT/locks/<host>_gpu<N>.lock (mkdir) is free; the job holds it until it exits.
The job is detached (nohup, in its own session through Popen's start_new_session; no `setsid`, which would fork and
hand the watchdog a PID that has already exited) with its progress log AT/logs/<job>_<UTC>.log (each step's start,
end, return code and seconds) and status AT/logs/<job>_<UTC>.status.json; every step runs under `timeout`; the job
stops at the first step that fails. The watchdog (STALL / EXIT / OVERRUN, think/runs/watchdog.py) is attached when
present, with the job's own PID.
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


def preflight(a: argparse.Namespace) -> tuple[list[str], list[str]]:
    """(problems, notes) for `read` and `rl`: a real launch refuses on any problem, before the gate and before
    anything is written; `--dry-run` prints both."""
    problems: list[str] = []
    notes: list[str] = []
    if a.cmd == "read":
        seeds, tags, v = [int(s) for s in a.seeds.split(",")], a.tags.split(","), a.variant
        for s in seeds:
            if not T.base_path(s, v).exists():
                problems.append(
                    f"no base {T.base_path(s, v)} (train it first: {T.pretrain_hint(s, v)})"
                )
        root = T.read_root(v)
        done = [
            c
            for c in (T.read_id(s, t, v) for s in seeds for t in tags)
            if (root / "results" / f"{c}.jsonl").exists()
        ]
        if done:
            problems.append(f"{root / 'results'} already holds {done}: a read is never overwritten")
        notes.append(f"reads -> {root} (ids {T.read_id(seeds[0], tags[0], v)}, ...)")
        for s in seeds:
            c = T.check_path(T.base_name(s, v))
            if c.exists():
                notes.append(f"no check step for {T.base_name(s, v)}: {c} exists (never re-run)")
    elif a.cmd == "rl":
        L = T.LINES[a.line]
        try:
            side = T.check_line_base(a.line)
            notes.append(f"base {T.line_base(a.line)}, D {side['train_tokens']:,} tokens (its .tag.json), "
                         f"ladder {L['ladder']}, run folder {T.AT / 'runs' / a.line}")  # fmt: skip
        except SystemExit as e:
            problems.append(str(e))
        problems += TM.segment_problems(a.line, a.seg, a.resume)
    return problems, notes


def plan(a: argparse.Namespace) -> tuple[str, list[dict]]:
    py, h = T.PY, str(HERE)
    if a.cmd in ("read", "rl") and not a.dry_run:
        problems, _ = preflight(a)
        if problems:
            raise SystemExit("refused (nothing taken or written): " + "; ".join(problems))
    if a.cmd == "pretrain":
        seeds = [int(s) for s in a.seeds.split(",")]
        v = a.variant
        if v is not None:
            for x in (T.corpus_path(v), T.manifest_path(v)):
                if not x.exists():
                    raise SystemExit(
                        f"no {x}: build it with tagcorpus.py --route-share {T.VARIANTS[v]}"
                    )
        steps = []
        for s in seeds:
            out = T.base_path(s, v)
            for x in (out, out.with_suffix(".tag.json"), Path(f"{out}.args.json")):
                if x.exists():
                    raise SystemExit(f"{x} exists: a base is never overwritten")
            tag = f"{v}_s{s}" if v else f"s{s}"
            argv = [py, f"{h}/tag_pretrain.py", "--seed", str(s), "--out", str(out)]
            if v is not None:
                argv += ["--variant", v]
            steps.append(step(f"pretrain_{tag}", argv, 1.5, EST["pretrain"]))
            steps.append(step(f"sample_{tag}", [py, f"{h}/tag_sample.py", "--ckpt", str(out), "--device", "cuda",
                                                "--n", "64", "--k", "4", "--max_lines", "6",
                                                "--out", str(out.with_suffix(".sample.json"))], 0.5, EST["sample"]))  # fmt: skip
        return f"pretrain_{v + '_' if v else ''}s{'-'.join(map(str, seeds))}", steps
    if a.cmd == "check":
        steps, short = [], []
        for name in a.bases.split(","):
            v, s = T.parse_base_name(name)
            base, ckpt = T.base_name(s, v), T.base_path(s, v)
            if not ckpt.exists() and not a.dry_run:
                raise SystemExit(f"no base {ckpt}")
            out = T.AT / "reads" / "check" / f"{base}.json"
            if out.exists():
                raise SystemExit(f"{out} exists: a check is never overwritten")
            steps.append(step(f"check_{base}", [py, f"{h}/tag_check.py", "--ckpt", str(ckpt), "--tags", a.tags,
                                                "--corpus", str(T.corpus_path(v)), "--out", str(out)],
                              0.75, EST["check"]))  # fmt: skip
            short.append(base.removeprefix("tagAB_"))
        if len({x["name"] for x in steps}) != len(steps):
            raise SystemExit(f"a base named twice: {a.bases}")
        return f"check_{'-'.join(short)}", steps
    if a.cmd == "read":
        seeds, v = [int(s) for s in a.seeds.split(",")], a.variant
        n = len(seeds) * len(a.tags.split(","))
        extra = ["--variant", v] if v is not None else []
        steps = [
            step("pc_gpu", [py, f"{h}/tag_read.py", "gpu", "--seeds", a.seeds, "--tags", a.tags, "--gpu", str(a.gpu), *extra],
                 max(1.0, 2.5 * n * EST["read_gpu_per_base_tag"]), n * EST["read_gpu_per_base_tag"]),
            step("pc_rows", [py, f"{h}/tag_read.py", "rows", "--seeds", a.seeds, "--tags", a.tags, *extra], 1.0, EST["rows"],
                 {"CUDA_VISIBLE_DEVICES": ""}),
        ]  # fmt: skip
        for s in seeds:
            out = T.check_path(T.base_name(s, v))
            if out.exists():
                continue  # a check is never re-run (preflight notes it)
            argv = [py, f"{h}/tag_check.py", "--ckpt", str(T.base_path(s, v)), "--tags", a.tags]
            argv += ["--corpus", str(T.corpus_path(v))] if v is not None else []
            steps.append(step(f"check_s{s}" if v is None else f"check_{v}_s{s}", [*argv, "--out", str(out)],
                              0.75, EST["check"]))  # fmt: skip
        return f"read_{v + '_' if v else ''}s{'-'.join(map(str, seeds))}", steps
    if a.cmd == "rl":
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
        problems, notes = preflight(a)
        for x in notes:
            print(f"  note: {x}")
        for x in problems:
            print(f"  A REAL LAUNCH REFUSES: {x}")
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
    ap.add_argument("cmd", choices=("pretrain", "read", "rl", "check", "status"))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument(
        "--variant",
        choices=sorted(T.VARIANTS),
        default=None,
        help="pretrain / read: a tie-strength variant (rl: the variant is in the line name)",
    )
    ap.add_argument(
        "--bases",
        default=None,
        help="check: base names, tagAB[_<variant>]_s<seed>, comma-separated",
    )
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
    assert a.cmd != "check" or a.bases, "--bases"
    assert a.variant is None or a.cmd in ("pretrain", "read"), (
        "--variant is for pretrain and read (check takes base names, rl line names)"
    )
    launch(a)


if __name__ == "__main__":
    main()
