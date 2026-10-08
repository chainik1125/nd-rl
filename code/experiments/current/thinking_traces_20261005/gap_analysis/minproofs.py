"""Shortest proofs for the gap analysis: textbook72 and matched generator samples (CPU only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    G=code/experiments/current/thinking_traces_20261005/gap_analysis
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 code/.venv/bin/python \
        $G/minproofs.py --set textbook --workers 8

Search: `test_21_30/fastlen.fast_minlen` (the fork's `minlen.Search` with a cross-budget memo:
iterative deepening on the total line count over the 14 rules, intermediate formulas restricted
to S = subformulas of the theorem, their negations and double negations; every proof found is
re-checked with nd_verify). The first proof found is a shortest proof *in S*.

* textbook (dev58 + train14): bound = reference_lines (dev) and then +2 steps up to 16 if
  nothing is found; train14 (no reference length) searches to 16. --tb_secs per problem.
* generator sets (pool / h12test / k12): bound = the generator proof's length, with the
  generator proof's formulas added to S (`extra`), so a proof of at most that length is always in
  the space. On timeout the generator proof is kept (`src` = "gen").

Each worker has an address-space cap (--mem_gb) so a blow-up fails one problem, not the box.
Output: one JSON row per problem in runs/gap_analysis/minproofs_<set>.jsonl (appended; problems
already present are skipped, so the script can be resumed).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "test_21_30"))

import common as C  # noqa: E402
from fastlen import fast_minlen  # noqa: E402

MEM_GB = 3.5


def _init(mem_gb: float) -> None:
    b = int(mem_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (b, b))


def _work(job: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    out = {k: job[k] for k in ("set", "name", "prompt", "ref_lines")}
    try:
        extra = ()
        if job.get("gen_proof"):
            extra = tuple(ln["f"] for ln in C.parse_proof(job["gen_proof"]))
        res = {
            "min_lines_ub": None,
            "proof": None,
            "timeout": False,
            "bound": None,
            "lb": None,
            "error": "skipped",
        }
        for bound in job["bounds"]:
            r = fast_minlen(job["prompt"], bound=bound, time_limit=job["secs"], extra=extra)
            res = r
            if r["min_lines_ub"] is not None or r["timeout"] or r.get("error"):
                break
        out.update(
            min_lines=res["min_lines_ub"],
            proof=res["proof"] if res["min_lines_ub"] is not None else None,
            timeout=res["timeout"],
            bound=res["bound"],
            lb=res.get("lb"),
            error=res.get("error"),
        )
    except MemoryError:
        out.update(min_lines=None, proof=None, timeout=False, error="memory")
    if out["proof"] is None and job.get("gen_proof"):
        out["proof"] = job["gen_proof"]
        out["src"] = "gen"
    else:
        out["src"] = "search" if out["proof"] else None
    out["secs"] = round(time.time() - t0, 2)
    return out


def jobs_for(which: str, n: int, secs: float, tb_secs: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    jobs = []
    if which == "textbook":
        for fn, role in (("textbook_dev.jsonl", "dev58"), ("textbook_train.jsonl", "train14")):
            for r in C.jl(C.TB / fn):
                rl = r.get("reference_lines")
                if rl is not None:
                    bounds = sorted({min(rl, 16), *(b for b in (rl + 2, rl + 4) if b <= 16)})
                    if rl > 16:
                        bounds = []  # out of exact-search reach: theorem features only
                else:
                    bounds = [10, 12, 14, 16]
                jobs.append(
                    dict(
                        set=role,
                        name=r["name"],
                        prompt=r["prompt"],
                        ref_lines=rl,
                        bounds=bounds,
                        secs=tb_secs,
                    )
                )
        return jobs
    if which == "pool":
        rows = C.jl(C.POOL)
        key = "gen_proof"
    elif which == "h12test":
        rows = C.jl(C.H12TEST)
        key = "reference_proof"
    elif which == "k12":
        rows = [r for r in C.k12_sample() if 7 <= r["n_lines"] <= 12]
        key = "proof"
    else:
        raise SystemExit(which)
    if n and n < len(rows):
        rows = rng.sample(rows, n)
    for i, r in enumerate(rows):
        L = r.get("n_lines") or r.get("reference_lines")
        jobs.append(
            dict(
                set=which,
                name=r.get("name") or f"k12_{i}_{r['key'][:40]}",
                prompt=r["prompt"],
                ref_lines=L,
                bounds=[L],
                secs=secs,
                gen_proof=r[key],
            )
        )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["textbook", "pool", "h12test", "k12"])
    ap.add_argument("--n", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--secs", type=float, default=60.0, help="time limit per generator problem")
    ap.add_argument("--tb_secs", type=float, default=600.0, help="time limit per textbook problem")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mem_gb", type=float, default=MEM_GB)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    out = C.OUT / f"minproofs_{a.set}.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(x)["name"] for x in f}
    jobs = [j for j in jobs_for(a.set, a.n, a.secs, a.tb_secs, a.seed) if j["name"] not in done]
    print(f"{a.set}: {len(jobs)} jobs ({len(done)} already done)", flush=True)
    t0 = time.time()
    with (
        mp.Pool(a.workers, initializer=_init, initargs=(a.mem_gb,), maxtasksperchild=50) as pool,
        open(out, "a") as fo,
    ):
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if (i + 1) % 100 == 0 or a.set == "textbook":
                print(
                    f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s {r['name']} L={r['min_lines']} "
                    f"ref={r['ref_lines']} to={r['timeout']} {r['secs']}s",
                    flush=True,
                )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
