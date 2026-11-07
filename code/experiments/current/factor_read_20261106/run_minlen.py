"""Reference proofs, step 1: Dan's `minlen.minlen` on the 106 strict σ theorems (CPU only).

    python run_minlen.py --out OUT.jsonl [--time 60] [--slack 2] [--procs 8] [--only OLD.jsonl]

The bound is the theorem's `n_lines` (Dan's label) + `slack`, because the shortest proof is what
the search returns and it stops at the first length that has one. With `--only OLD.jsonl` it reruns
only the theorems OLD left without a proof. Each process is capped at 5 GB of address space so a
memo blow-up ends that theorem, not the machine. Proof text stays in OUT, under /scratch.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

KD = Path(os.environ.get("KD", "/scratch/cowifornia/nd-rl/knockout_danbase"))
DAN = KD / "dan_code_13e41bbd"
sys.path.insert(0, str(DAN))
sys.path.insert(0, str(DAN / "coe"))

GB = 2**30


def work(a: tuple) -> dict:
    rec, bound, tl = a
    from minlen import minlen  # type: ignore[import-not-found]

    resource.setrlimit(resource.RLIMIT_AS, (5 * GB, 5 * GB))
    t0 = time.time()
    try:
        r = minlen(rec["prompt"], bound, tl)
    except MemoryError:
        r = {"min_lines_ub": None, "proof": None, "timeout": False, "memory": True}
    r["secs"] = time.time() - t0
    r.update(name=rec["name"], prompt=rec["prompt"], schema=rec.get("schema"),
             n_lines=rec.get("n_lines"), bound=bound)  # fmt: skip
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    strict = (DAN / "data/coe/sigma_heldout_strict.txt").read_text().split()
    by = {}
    for line in open(DAN / "data/coe/transfer_track832.jsonl"):
        t = json.loads(line)
        by[t["name"]] = t
    names = strict
    if a.only:
        have = {json.loads(x)["name"] for x in open(a.only) if json.loads(x).get("proof")}
        names = [n for n in strict if n not in have]
    jobs = [(by[n], by[n]["n_lines"] + a.slack, a.time) for n in names]
    t0 = time.time()
    with mp.Pool(a.procs, maxtasksperchild=1) as pool, open(a.out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(work, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{i + 1}/{len(jobs)} {r['name']} {r['schema']} n_lines {r['n_lines']} "
                  f"got {r.get('min_lines_ub')} timeout {r.get('timeout')} {r['secs']:.0f}s "
                  f"[{time.time() - t0:.0f}s]", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
