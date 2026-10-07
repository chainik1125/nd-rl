"""Stage 1b: one piece pool from the generator runs: drop premises the minimal proof does not use
(exactness is kept, see `ndproof.drop_unused_premises`), de-duplicate by renaming class, flag what
the composer needs, and drop pieces whose class is in any training / eval / pool file (`seal`
index).

    cd nd-rl && python -m code.experiments.current.thinking_traces_20261005.test_21_30.prep_pieces \
        --in work/mains.jsonl,work/lemmas.jsonl --index work/excl_index.pkl \
        --out work/pieces_all.jsonl

Rows: {prompt, proof (minimal, nd_verify-checked), L (exact minimal length in minlen's space), key,
n_prem, last_rule, has_bote, concl_lit, consistent, src, reduced_from (L before dropping unused
premises)}.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ndproof import drop_unused_premises, parse, satisfiable  # noqa: E402
from seal import hits  # noqa: E402

from code.thinking.data import class_keys  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_L", type=int, default=4)
    a = ap.parse_args()
    idx = None
    if a.index:
        with open(a.index, "rb") as f:
            idx = pickle.load(f)
    st = collections.Counter()
    seen = set()
    with open(a.out, "w") as fo:
        for path in a.inp.split(","):
            with open(path) as fh:
                for line in fh:
                    r = json.loads(line)
                    p = parse(r["prompt"], r["proof"])
                    q = drop_unused_premises(p)
                    L = len(q.lines)
                    if len(p.lines) > L:
                        st["reduced"] += 1
                    ok, why = q.verify()
                    if not ok:
                        st["reduced_verify_fail"] += 1
                        continue
                    if a.min_L > L or q.concl in q.prem:
                        st["too_short"] += 1
                        continue
                    prompt = q.prompt()
                    k, pk = class_keys(prompt)
                    if (k, pk) in seen:
                        st["dup"] += 1
                        continue
                    seen.add((k, pk))
                    if idx is not None and hits(idx, prompt):
                        st["excluded"] += 1
                        continue
                    f = q.concl
                    row = dict(
                        prompt=prompt,
                        proof=q.body(),
                        L=L,
                        key=k,
                        n_prem=len(q.prem),
                        last_rule=q.lines[-1].rule,
                        has_bote=any(ln.rule == "BOTE" for ln in q.lines),
                        concl_lit=f[0] == "atom" or (f[0] == "not" and f[1][0] == "atom"),
                        consistent=satisfiable(q.prem),
                        src=os.path.basename(path).split(".")[0] + ":" + r.get("mode", ""),
                        reduced_from=r["L"],
                    )
                    fo.write(json.dumps(row) + "\n")
                    st["kept"] += 1
                    st[f"L{L}"] += 1
    print(json.dumps(dict(st)))


if __name__ == "__main__":
    main()
