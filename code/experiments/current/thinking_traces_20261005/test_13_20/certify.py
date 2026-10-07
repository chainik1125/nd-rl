"""Apply exact-search results to 13-20 candidates: certified, replaced, or bounded.

    python certify.py --in work/cands_lb.jsonl --out work/cands_cert.jsonl

Input rows carry `minsearch` from test_21_30/minimality.py, run with bound = reference lines - 1
(13-17 lines: exhaustive, so the result is exact) or a lower bound (18-20 lines).  For each row:
  * search finished to lines - 1, nothing shorter -> `cert` = "exact" (L_true = reference lines);
  * a shorter proof was found (minimal: the search is iterative deepening) -> the reference becomes
    that proof, `cert` = "exact_replaced" (the construction proof is kept in `proof_construction`);
  * otherwise -> `cert` = "lower_bound" (minimal length >= lb + 1).
`minsearch` is rewritten for replaced rows so that `assemble.py write` keeps them (min_found
cleared, lb = L - 1).  Shape fields are recomputed for replaced proofs.  All in the fork's
`minlen` space plus the reference's formulas.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

T2130 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_21_30")
sys.path.insert(0, T2130)
sys.dont_write_bytecode = True

from compose import shape  # noqa: E402
from ndproof import parse  # noqa: E402
from state_env import decompose  # noqa: E402  (fork, read-only; on sys.path via ndproof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = collections.Counter()
    with open(a.inp) as f, open(a.out, "w") as fo:
        for line in f:
            r = json.loads(line)
            ms = r["minsearch"]
            n = len(r["proof"].split(";")) - 1
            if ms.get("min_found") is not None:
                p = parse(r["prompt"], ms["min_proof"])
                ok, _ = p.verify()
                if not ok:
                    st["replacement_verify_fail"] += 1
                    continue
                r["proof_construction"] = r["proof"]
                r["proof"] = p.body()
                r.update(shape(p, decompose))
                m = len(p.lines)
                r["minsearch"] = dict(ms, min_found=None, min_proof=None, lb=m - 1, replaced_from=n)
                r["cert"] = "exact_replaced"
            elif not ms.get("timeout") and ms.get("lb", 0) >= n - 1:
                r["cert"] = "exact"
            else:
                r["cert"] = "lower_bound"
            st[r["cert"]] += 1
            fo.write(json.dumps(r) + "\n")
    print(json.dumps(dict(st)))


if __name__ == "__main__":
    main()
