"""Stage 3: bounded minimal-length search on composites (lower bounds, shortcut detection).

    python minimality.py --in work/comps.jsonl --out work/comps_min.jsonl --bound 14 --procs 16 \
        --stages 10:5,12:30,14:120

For each composite, `fastlen` (the fork's `minlen` rules and search space, cross-budget memo)
searches for a proof shorter than the reference, by staged iterative deepening up to min(--bound,
lines - 1). The search space is the fork's (subformulas of the theorem, their negations and double
negations) plus every formula on the reference proof (`--extra ref`, default), so the reference's
own route is inside it. Output fields:
  min_found length of a shorter proof found (nd_verify-checked; `min_proof` holds it), else null lb
  largest total length certified to have no proof in that space (the minimal length is >= lb + 1)
  timeout the last stage timed out (lb is then the previous stage's certificate) exact min_found is
  not None or lb == lines - 1 (the reference is minimal in the space)
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def work(args):
    rec, stages, bound, use_extra = args
    from fastlen import fast_minlen
    from ndproof import parse

    p = parse(rec["prompt"], rec["proof"])
    extra = [ln.f for ln in p.lines] if use_extra else ()
    n = len(p.lines)
    top = min(bound, n - 1)
    lb = len(p.prem)
    t0 = time.time()
    out = dict(min_found=None, min_proof=None, lb=lb, timeout=False, bound=top)
    for b, tl in stages:
        b = min(b, top)
        if b <= lb:
            continue
        r = fast_minlen(rec["prompt"], b, tl, extra=extra, start=lb + 1)
        if r.get("error"):
            out["error"] = r["error"]
            break
        if r["timeout"]:
            out["timeout"] = True
            out["timeout_stage"] = b
            break
        if r["min_lines_ub"] is not None:
            out["min_found"] = r["min_lines_ub"]
            out["min_proof"] = r["proof"]
            out["lb"] = r["min_lines_ub"] - 1
            break
        out["lb"] = r["lb"]
        if b == top:
            break
    out["exact"] = out["min_found"] is not None or out["lb"] >= n - 1
    out["secs"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bound", type=int, default=14)
    ap.add_argument("--stages", default="10:5,12:30,14:120")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--no_extra", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    stages = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in a.stages.split(",")]
    with open(a.inp) as fi:
        recs = [json.loads(line) for line in fi]
    if a.limit:
        recs = recs[: a.limit]
    t0 = time.time()
    st = collections.Counter()
    with mp.Pool(a.procs, maxtasksperchild=50) as pool, open(a.out, "w") as fo:
        jobs = [(r, stages, a.bound, not a.no_extra) for r in recs]
        for i, (r, res) in enumerate(zip(recs, pool.imap(work, jobs, chunksize=1), strict=True)):
            r = dict(r)
            r["minsearch"] = res
            fo.write(json.dumps(r) + "\n")
            st[
                "found_shorter"
                if res["min_found"]
                else ("timeout" if res["timeout"] else "none_shorter")
            ] += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(recs)} {time.time() - t0:.0f}s {dict(st)}", flush=True)
    print(json.dumps(dict(n=len(recs), stats=dict(st), secs=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
