"""Run schedule jobs on the allowed GPUs, each inside `runner.gpu.lease` (lock + ledger).

    NDBENCH_ALLOWED_GPUS=0,1,2,3,4 code/.venv/bin/python \\
        code/experiments/current/benchmarking_20260930/run_schedule.py \\
        --schedule <yaml> --jobs G0,G1 [--gpus 0,1,2,3,4] [--scale 1.0] [--dry-run]

The benchmark panel (`code/ndbench/panel.yaml`, see `code/ndbench/README.md`) adds cost tiers and a
`{model}` placeholder, so the cheap benchmarks can run alone on any registered model:

    NDBENCH_ALLOWED_GPUS=4 code/.venv/bin/python \\
        code/experiments/current/benchmarking_20260930/run_schedule.py \\
        --schedule code/ndbench/panel.yaml --model <label> --max-tier cheap --budget-h 0.5 --dry-run

`--max-tier` selects every job at or below that tier (cpu < cheap < moderate < expensive); `--jobs`
then narrows the selection further. `cpu` jobs run without a GPU lease and with no visible GPU.
Jobs with `scope: pair` (the interp jobs; their configs name a base/rl pair) run only with
`--include-pair`. `--budget-h` caps the GPU time this invocation may spend, instead of the
schedule's `cap_h` on the whole ledger.

Run from the worktree root. `--jobs` takes schedule ids or job names; jobs start in schedule order,
one per GPU at a time (one worker thread per GPU in `--gpus`, default: `NDBENCH_ALLOWED_GPUS`, else
4). Each job runs as a subprocess with `CUDA_VISIBLE_DEVICES=<its GPU>` set at launch; this driver
never imports torch. Before each job the plan's stop rule is applied, counting jobs still running:
spent + in-flight estimates + 1.5 x estimate <= cap. A job is stopped at 3 x its estimate (the
runner stores results as it goes and resumes, so a stopped job is partial, not lost). Each job's
output goes to out/logs/<job>.log.
"""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from code.ndbench.runner.gpu import allowed_gpus, lease  # noqa: E402
from code.ndbench.store import Store  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/bench/out")
PY = str(ROOT / "code" / ".venv" / "bin" / "python")
_lock = threading.Lock()
_inflight_s: dict[str, float] = {}
TIERS = ("cpu", "cheap", "moderate", "expensive")


def _log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


def _worker(
    gpu: int,
    jobs: queue.Queue[dict[str, object]],
    store: Store,
    args: argparse.Namespace,
    cap_h: float,
    spent0_s: float,
) -> None:
    while True:
        try:
            j = jobs.get_nowait()
        except queue.Empty:
            return
        name = str(j["job"])
        est_s = float(j["est_min"]) * 60 * args.scale  # type: ignore[arg-type]
        cpu_only = j.get("tier") == "cpu"
        with _lock:
            spent_s = store.ledger_total_s() - spent0_s
            busy_s = sum(_inflight_s.values())
            if not cpu_only and (spent_s + busy_s + 1.5 * est_s) / 3600.0 > cap_h:
                print(
                    f"SKIP {name}: spent {spent_s / 3600:.2f} h + in-flight "
                    f"{busy_s / 3600:.2f} h + 1.5 x {est_s / 3600:.2f} h > cap {cap_h} h",
                    flush=True,
                )
                continue
            if not cpu_only:
                _inflight_s[name] = est_s
        cmd = shlex.split(str(j["command"]))
        if cmd[0] == "python":
            cmd[0] = PY
        where = "CPU" if cpu_only else f"GPU {gpu}"
        _log(f"RUN  {name} [{j.get('tier', '-')}] on {where}, est {est_s / 60:.1f} min")
        if args.dry_run:
            _log("     " + " ".join(cmd))
            with _lock:
                _inflight_s.pop(name, None)
            continue
        log = OUT / "logs" / f"{name}.log"
        t0 = time.time()
        if cpu_only:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            with log.open("a") as fh:
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                fh.write(f"# {stamp} CPU {' '.join(cmd)}\n")
                fh.flush()
                rc = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT
                ).returncode
            _log(f"DONE {name} on CPU rc={rc} in {(time.time() - t0) / 60:.1f} min")
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        rc: object = "error"
        try:
            with (
                lease(name, est_s, store, gpu=gpu, wait_s=600, check_env=False),
                log.open("a") as fh,
            ):
                fh.write(
                    f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} GPU {gpu} "
                    f"{' '.join(cmd)}\n"
                )
                fh.flush()
                try:
                    rc = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        timeout=3 * est_s + 60,
                    ).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
        finally:
            with _lock:
                _inflight_s.pop(name, None)
        _log(f"DONE {name} on GPU {gpu} rc={rc} in {(time.time() - t0) / 60:.1f} min")


def select_jobs(
    jobs: list[dict[str, object]],
    *,
    want: set[str] | None,
    max_tier: str | None,
    model: str | None,
    include_pair: bool,
) -> list[dict[str, object]]:
    """The schedule's jobs, in order, filtered by `--jobs`/`--max-tier`/scope, with `{model}`
    and `{out}` filled in. A job that names `{model}` needs `--model`; others run as written."""
    out: list[dict[str, object]] = []
    for j in jobs:
        tier = j.get("tier")
        if max_tier is not None:
            if tier not in TIERS:
                raise SystemExit(f"job {j['job']} has no valid tier ({tier!r}); tiers: {TIERS}")
            if TIERS.index(str(tier)) > TIERS.index(max_tier):
                continue
        if j.get("scope") == "pair" and not include_pair:
            continue
        templated = "{model}" in str(j["job"]) or "{model}" in str(j["command"])
        if templated and model is None:
            raise SystemExit(f"job {j['job']} needs --model")
        fill = {"model": model or "", "out": str(OUT)}
        j = {**j, "job": str(j["job"]).format(**fill), "command": str(j["command"]).format(**fill)}
        if want is not None and not ({str(j["id"]), str(j["job"])} & want):
            continue
        out.append(j)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=str(Path(__file__).with_name("schedule.yaml")))
    ap.add_argument(
        "--jobs",
        default=None,
        help="comma-separated schedule ids or job names (required unless --max-tier is given)",
    )
    ap.add_argument("--max-tier", choices=TIERS, default=None, dest="max_tier")
    ap.add_argument("--model", default=None, help="fills {model} in panel jobs")
    ap.add_argument("--include-pair", action="store_true", dest="include_pair")
    ap.add_argument(
        "--budget-h",
        type=float,
        default=None,
        dest="budget_h",
        help="GPU-hours this invocation may spend (default: the schedule's cap_h on the ledger)",
    )
    ap.add_argument("--gpus", default=None, help="GPUs to use (default: NDBENCH_ALLOWED_GPUS)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every est_min")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    allowed = allowed_gpus()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(allowed)
    bad = [g for g in gpus if g not in allowed]
    if bad:
        sys.exit(f"refusing: GPUs {bad} not in NDBENCH_ALLOWED_GPUS={allowed}")
    if args.jobs is None and args.max_tier is None:
        sys.exit("give --jobs, --max-tier, or both")
    sched = yaml.safe_load(Path(args.schedule).read_text())
    want = set(args.jobs.split(",")) if args.jobs else None
    selected = select_jobs(
        sched["jobs"],
        want=want,
        max_tier=args.max_tier,
        model=args.model,
        include_pair=args.include_pair,
    )
    gpu_min = sum(float(j["est_min"]) for j in selected if j.get("tier") != "cpu") * args.scale  # type: ignore[arg-type]
    print(f"{len(selected)} jobs selected, GPU estimate {gpu_min:.0f} min", flush=True)
    q: queue.Queue[dict[str, object]] = queue.Queue()
    for j in selected:
        q.put(j)
    store = Store(OUT)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if args.budget_h is not None:
        cap_h, spent0_s = args.budget_h, store.ledger_total_s()
    else:
        cap_h, spent0_s = float(sched.get("cap_h", 9.5)), 0.0
    threads = [
        threading.Thread(target=_worker, args=(g, q, store, args, cap_h, spent0_s)) for g in gpus
    ]
    for t in threads:
        t.start()
        time.sleep(2)  # stagger lock acquisition and model loading
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
