"""Run schedule jobs on the allowed GPUs, each inside `runner.gpu.lease` (lock + ledger).

    NDBENCH_ALLOWED_GPUS=0,1,2,3,4 code/.venv/bin/python \\
        code/experiments/current/benchmarking_20260930/run_schedule.py \\
        --schedule <yaml> --jobs G0,G1 [--gpus 0,1,2,3,4] [--scale 1.0] [--dry-run]

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


def _log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


def _worker(
    gpu: int,
    jobs: queue.Queue[dict[str, object]],
    store: Store,
    args: argparse.Namespace,
    cap_h: float,
) -> None:
    while True:
        try:
            j = jobs.get_nowait()
        except queue.Empty:
            return
        name = str(j["job"])
        est_s = float(j["est_min"]) * 60 * args.scale  # type: ignore[arg-type]
        with _lock:
            spent_s = store.ledger_total_s()
            busy_s = sum(_inflight_s.values())
            if (spent_s + busy_s + 1.5 * est_s) / 3600.0 > cap_h:
                print(
                    f"SKIP {name}: spent {spent_s / 3600:.2f} h + in-flight "
                    f"{busy_s / 3600:.2f} h + 1.5 x {est_s / 3600:.2f} h > cap {cap_h} h",
                    flush=True,
                )
                continue
            _inflight_s[name] = est_s
        cmd = shlex.split(str(j["command"]))
        if cmd[0] == "python":
            cmd[0] = PY
        _log(f"RUN  {name} on GPU {gpu}, est {est_s / 60:.1f} min")
        if args.dry_run:
            _log("     " + " ".join(cmd))
            with _lock:
                _inflight_s.pop(name, None)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = OUT / "logs" / f"{name}.log"
        t0 = time.time()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=str(Path(__file__).with_name("schedule.yaml")))
    ap.add_argument("--jobs", required=True, help="comma-separated schedule ids or job names")
    ap.add_argument("--gpus", default=None, help="GPUs to use (default: NDBENCH_ALLOWED_GPUS)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every est_min")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    allowed = allowed_gpus()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(allowed)
    bad = [g for g in gpus if g not in allowed]
    if bad:
        sys.exit(f"refusing: GPUs {bad} not in NDBENCH_ALLOWED_GPUS={allowed}")
    sched = yaml.safe_load(Path(args.schedule).read_text())
    want = set(args.jobs.split(","))
    q: queue.Queue[dict[str, object]] = queue.Queue()
    for j in sched["jobs"]:
        if j["id"] in want or j["job"] in want:
            q.put(j)
    store = Store(OUT)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cap_h = float(sched.get("cap_h", 9.5))
    threads = [threading.Thread(target=_worker, args=(g, q, store, args, cap_h)) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(2)  # stagger lock acquisition and model loading
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
