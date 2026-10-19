"""Time limits for the 16-18-line band (L4): re-certify `run.py generate`'s search timeouts with
long limits and record how long each search and necessity check takes.

    PFG_MIN_L=16 PFG_MAX_L=18 PFG_SEARCH_SECS=240 PFG_NEC_SECS=120 python -m \
        code.experiments.current.thinking_traces_20261005.curriculum.calib_L4 \
        --source .../curriculum/L4/gen --out .../curriculum/L4/calib/calib_L4.jsonl --workers 16

L2r / L3r use `PFG_SEARCH_SECS=12` and `PFG_NEC_SECS=14.5` (chosen to match serrano's timeout
rates at 7-12 lines, `calib_timeouts.py`). At 13-15 lines the shortest-proof search already takes
a median 7.0 s (p90 10.1 s) for L3r's certified 15-line rows, and at 16 lines and above almost
every candidate whose shortest proof is that long times out at 12 s: seeds 0-1M (default
weights) gave 0 certified rows. This rebuilds every `search_timeout` reject of a generate run
(its seed and the run's recorded recipe weights, checked against the stored class), runs
`pfgates.certify` with the limits from the environment, and writes one line per candidate with
the status, the shortest length, the search time and each necessity check's verdict and time.
From those times one can read off, for any pair of limits, which candidates would certify, which
is how the L4 limits were chosen (`build_L4.sh`, header).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import regen as RG
from code.experiments.current.thinking_traces_20261005.hard712 import pfgates as GT
from code.experiments.current.thinking_traces_20261005.hard712 import run as RN


def one(job: tuple[dict[str, Any], Any]) -> dict[str, Any]:
    c = RG.rebuild(job)
    if "prompt" not in c:
        return {"seed": job[0]["seed"], "pkey": job[0]["pkey"], "status": c["status"]}
    t0 = time.time()
    r = RN.certify_safe(c)
    keep = ("seed", "recipe", "pkey", "gen_lines", "status", "min_lines", "t_search", "families")
    out = {k: r.get(k) for k in keep}
    out["necessity"] = r.get("necessity")
    out["t_total"] = round(time.time() - t0, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="a run.py generate output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", nargs="+", default=["search_timeout"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_n", type=int, default=0, help="at most this many candidates (0: all)")
    a = ap.parse_args()
    work = Path(a.source) / "work"
    rw = json.loads((work / "progress.json").read_text()).get("round_weights", {})
    starts = sorted(int(k) for k in rw)
    rej = []
    with open(work / "rejects.jsonl") as f:  # read only: a running generate may be appending
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in a.status:
                rej.append(r)
    if a.max_n:
        rej = rej[: a.max_n]
    done: set[int] = set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {r["seed"] for r in RN.jl(out)}
    jobs = [
        (r, RG.weights_from_rounds(r["seed"], starts, rw)) for r in rej if r["seed"] not in done
    ]
    RN.log(
        f"calib: {len(jobs)} candidates (search {GT.SEARCH_SECS}s, necessity {GT.NEC_SECS}s, "
        f"band {GT.MIN_L}-{GT.MAX_L}, gap {GT.MAX_GAP})"
    )
    t0 = time.time()
    with (
        mp.get_context("spawn").Pool(
            a.workers, initializer=RN._init, initargs=(RN.MEM_GB,), maxtasksperchild=50
        ) as pool,
        open(out, "a") as f,
    ):
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                RN.log(f"calib {i}/{len(jobs)} {time.time() - t0:.0f}s")
    RN.log(f"calib done: {len(jobs)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
