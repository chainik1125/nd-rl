"""Candidate theorems for the hard RL pool and the near-OOD band, from Dan's generator, unchanged.

    python gen_candidates.py --out /scratch/cowifornia/nd-rl/think/runs/pools/raw/cands.jsonl \
        --min 7 --max 20 --per_len 6000 --workers 8 --seed 52000

The generator is the fork's `gen.sample_one(long=True)` with the take-home knobs (max_prem 3,
max_depth 3): the strict long generator that built the ladder's `rl_targets.jsonl` and
`transfer_long.jsonl` (`make_coverage_sets.py gen --long`).  Only the output filter differs:
generated (reference) length in [--min, --max], contradictory-premise theorems dropped (as the
ladder did), at most --per_len distinct renaming classes per length.  Every kept proof passes
`nd_verify.verify_text`.  The fork is imported read-only (no bytecode is written into it).

Worker i uses seed + 1000 * i, so worker streams never overlap (long-pool-2's lesson: seeds spaced
by 1 repeat streams).  Output rows: {thm, prompt, proof, n_lines, rules, n_prem, gen_last_rule,
n_lazy_prem, key, seed}; `key` is the fork's renaming-class key (`gen.canon_key`), deduplicated
across workers.  A JSON summary goes to <out>.summary.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import sys
import time
from typing import Any

FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"


def _fork() -> tuple[Any, Any, Any, Any, Any]:
    sys.dont_write_bytecode = True
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    from gen import Fail, Gen, canon_key, sample_one  # type: ignore[import-not-found]
    from nd_verify import verify_text  # type: ignore[import-not-found]

    return Gen, sample_one, Fail, canon_key, verify_text


def worker(args: tuple[int, int, int, int, int, str]) -> dict[str, Any]:
    seed, tries, mn, mx, per_len, out = args
    Gen, sample_one, Fail, canon_key, verify_text = _fork()
    rng = random.Random(seed)
    g = Gen(rng, max_prem=3, max_depth=3)
    seen: set[str] = set()
    per = collections.Counter()
    st = collections.Counter()
    t0 = time.time()
    with open(out, "w") as fo:
        for _ in range(tries):
            st["tries"] += 1
            try:
                r = sample_one(g, rng, True)
            except (Fail, RecursionError):
                st["fail"] += 1
                continue
            L = r["n_lines"]
            if not mn <= L <= mx:
                st["len_out"] += 1
                continue
            if r["contra_prem"]:
                st["contra"] += 1
                continue
            if per[L] >= per_len:
                st["len_full"] += 1
                continue
            key = canon_key(r["thm"])
            if key in seen:
                st["dup"] += 1
                continue
            ok, _, nl = verify_text(r["prompt"] + " " + r["proof"])
            if not ok:
                st["verifier_reject"] += 1
                continue
            assert nl == L
            seen.add(key)
            per[L] += 1
            r.pop("text", None)
            fo.write(json.dumps({**r, "key": key, "seed": seed}) + "\n")
            if all(per[x] >= per_len for x in range(mn, mx + 1)):
                break
    return {"seed": seed, "per_len": dict(per), "stats": dict(st), "secs": time.time() - t0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=7)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--per_len", type=int, default=6000, help="per worker, per generated length")
    ap.add_argument("--tries", type=int, default=3_000_000, help="per worker")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=52000)
    a = ap.parse_args(argv)
    jobs = [
        (a.seed + 1000 * i, a.tries, a.min, a.max, a.per_len, f"{a.out}.w{i}")
        for i in range(a.workers)
    ]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(worker, jobs)
    _, _, _, _, _ = _fork()
    seen: set[str] = set()
    per = collections.Counter()
    dup = 0
    with open(a.out, "w") as fo:
        for (_, _, _, _, _, path), _r in zip(jobs, res, strict=True):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["key"] in seen:
                        dup += 1
                        continue
                    seen.add(r["key"])
                    per[r["n_lines"]] += 1
                    fo.write(line)
    summary = {
        "args": vars(a),
        "generator": "fork gen.sample_one(long=True), Gen(max_prem=3, max_depth=3); fork 4dcd0e4f",
        "distinct_classes": len(seen),
        "cross_worker_dups": dup,
        "per_len": dict(sorted(per.items())),
        "workers": res,
        "secs": time.time() - t0,
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("distinct_classes", "per_len", "secs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
