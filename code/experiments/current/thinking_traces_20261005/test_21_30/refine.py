"""Stage 3a: shorten composites by a vocabulary-restricted exact search, then recompute shape
fields.

    python refine.py --in work/comps.jsonl --out work/comps_ref.jsonl --secs 30 --procs 16

For each composite, `fastlen.vocab_shorten` looks for a proof of length <= lines - 1 whose
intermediate formulas (elimination partners, lemmas, reductio hypotheses) are among the reference
proof's own formulas, by one exhaustive branch-and-bound call within --secs. At 12-16 lines this
search found 53 of the 55 shortcuts the full minlen search found (README). If a shorter proof is
found, it replaces the reference (after `ndproof.simplify` and nd_verify) and the search repeats on
it. Fields added: `vocab` = {status: minimal (search completed, nothing shorter) | timeout |
shortened, from, to, secs}; `lines` and the shape fields are recomputed for the final reference.
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
    rec, secs = args
    from compose import shape
    from fastlen import vocab_shorten
    from ndproof import parse, simplify
    from state_env import decompose

    p = parse(rec["prompt"], rec["proof"])
    n0 = len(p.lines)
    t0 = time.time()
    status = "minimal"
    while True:
        left = secs - (time.time() - t0)
        if left <= 1:
            status = "timeout" if status != "shortened" else status
            break
        r = vocab_shorten(rec["prompt"], [ln.f for ln in p.lines], len(p.lines) - 1, left)
        if r.get("error"):
            status = "error"
            break
        if r["timeout"]:
            status = "timeout" if len(p.lines) == n0 else "shortened"
            break
        if r["min_lines_ub"] is None:
            status = "minimal" if len(p.lines) == n0 else "shortened"
            break
        q = simplify(parse(rec["prompt"], r["proof"]))
        if not q.verify()[0] or len(q.lines) >= len(p.lines):
            status = "error"
            break
        p = q
    out = dict(rec)
    out["vocab"] = dict(status=status, frm=n0, to=len(p.lines), secs=round(time.time() - t0, 2))
    if len(p.lines) != n0:
        out["proof_construction"] = rec["proof"]
        out["proof"] = p.body()
        out.update(shape(p, decompose))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    with open(a.inp) as fi:
        recs = [json.loads(line) for line in fi]
    st = collections.Counter()
    t0 = time.time()
    with mp.Pool(a.procs, maxtasksperchild=40) as pool, open(a.out, "w") as fo:
        for i, r in enumerate(pool.imap(work, [(r, a.secs) for r in recs], chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            st[r["vocab"]["status"]] += 1
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(recs)} {time.time() - t0:.0f}s {dict(st)}", flush=True)
    print(json.dumps(dict(n=len(recs), stats=dict(st), secs=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
