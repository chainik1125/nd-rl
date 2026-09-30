"""Run schedule.yaml jobs one at a time on GPU 4, each inside `runner.gpu.lease` (lock + ledger).

    CUDA_VISIBLE_DEVICES=4 code/.venv/bin/python \
        code/experiments/current/benchmarking_20260930/run_schedule.py \
        --jobs G0 [--scale 1.0] [--dry-run]

Run from the worktree root. `--jobs` takes schedule ids (G0, G1, ...) or job names; jobs run in
schedule order. Before each job the plan's stop rule is applied (`budget_guard`: spent + 1.5 x
estimate <= cap); a job is stopped at 3 x its estimate (the runner stores chunks as it goes, so a
stopped job is `partial`, not lost). Each job's stdout/stderr goes to out/logs/<job>.log.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from code.ndbench.runner.gpu import BudgetExceeded, budget_guard, lease  # noqa: E402
from code.ndbench.store import Store  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/bench/out")
PY = str(ROOT / "code" / ".venv" / "bin" / "python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=str(Path(__file__).with_name("schedule.yaml")))
    ap.add_argument("--jobs", required=True, help="comma-separated schedule ids or job names")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every est_min (from G0)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        sys.exit("refusing: launch with CUDA_VISIBLE_DEVICES=4 set in the environment")
    sched = yaml.safe_load(Path(args.schedule).read_text())
    want = set(args.jobs.split(","))
    jobs = [j for j in sched["jobs"] if j["id"] in want or j["job"] in want]
    store = Store(OUT)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for j in jobs:
        est_s = float(j["est_min"]) * 60 * args.scale
        cmd = shlex.split(j["command"])
        if cmd[0] == "python":
            cmd[0] = PY
        try:
            left = budget_guard(store, est_s, cap_h=float(sched.get("cap_h", 9.5)))
        except BudgetExceeded as e:
            print(f"SKIP {j['job']}: {e}", flush=True)
            continue
        print(f"RUN  {j['job']} est {est_s / 60:.1f} min, budget left {left:.2f} h", flush=True)
        if args.dry_run:
            print("     " + " ".join(cmd), flush=True)
            continue
        log = OUT / "logs" / f"{j['job']}.log"
        t0 = time.time()
        with lease(j["job"], est_s, store), log.open("a") as fh:
            fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
            fh.flush()
            try:
                rc = subprocess.run(
                    cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=3 * est_s + 60
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"DONE {j['job']} rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
