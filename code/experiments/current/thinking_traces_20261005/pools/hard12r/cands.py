"""Candidate sets for hard12r.

    C=code.experiments.current.thinking_traces_20261005.pools.hard12r.cands
    python -m $C prior --work runs/pools/hard12_work --index excl_index.pkl \
        --out runs/pools/hard12r/work/cands/prior.jsonl
    python -m $C evalset --in cands/prior.jsonl --out sets/pilot_prior.jsonl [--shards 2]

`prior`: the hard12 candidates that T0_s0 already measured at k 32 and that are in neither
rl_hard12 nor the hard12 test set: the spare in-band targets (`spare_band.jsonl`, T0 2-16 of 32)
and every other measured candidate with T0 n_ok <= --max_t0 (default 1; above the band, 17+ of 32,
is left out: too easy before RL). Records are hard12 candidate records (`assemble12.records`) plus
`t0_n_ok`, `t0_k`, `src` (`spare` / `t0_le1`). Rows whose renaming class or premise-order key is in
the index, or repeats an earlier row, are dropped (counts in <out>.report.json).

`evalset`: the eval-set rows `code.thinking.eval --sets` reads (name, prompt, reference_lines,
family, knob, reference_proof), optionally in round-robin shards <out>_s<i>.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from typing import Any

from code.experiments.current.thinking_traces_20261005.pools.hard12.assemble12 import (
    records,
    t0_runs,
)
from code.experiments.current.thinking_traces_20261005.test_21_30.seal import hits
from code.thinking.data import class_keys, read_jsonl

POOLS = "/scratch/cowifornia/nd-rl/think/runs/pools"


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cmd_prior(a: argparse.Namespace) -> None:
    recs = records(a.work)
    rows: dict[str, dict[str, Any]] = {}
    for d in t0_runs(a.work):
        for row in read_jsonl(f"{d}/rows.jsonl"):
            rows.setdefault(row["name"], row)
    used = {r["name"][4:] for r in read_jsonl(f"{POOLS}/rl_hard12.jsonl")}
    used |= {r["name"][5:] for r in read_jsonl(f"{POOLS}/hard12_test/hard12_test.jsonl")}
    spare = {r["name"][4:] for r in read_jsonl(f"{a.work}/spare_band.jsonl")}
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    st: collections.Counter[str] = collections.Counter()
    seen_k: set[str] = set()
    seen_pk: set[str] = set()
    out = []
    for n in sorted(rows):
        row = rows[n]
        if n not in recs or n in used:
            st["in_hard12" if n in used else "no_record"] += 1
            continue
        if n in spare:
            src = "spare"
        elif row["n_ok"] <= a.max_t0:
            src = f"t0_le{a.max_t0}"
        else:
            st["t0_too_easy_or_unused"] += 1
            continue
        r = recs[n]
        if hits(idx, r["prompt"]):
            st[f"excluded_{src}"] += 1
            continue
        k, pk = class_keys(r["prompt"])
        if k in seen_k or (pk is not None and pk in seen_pk):
            st[f"dup_{src}"] += 1
            continue
        seen_k.add(k)
        if pk is not None:
            seen_pk.add(pk)
        st[f"kept_{src}"] += 1
        out.append({**r, "t0_n_ok": row["n_ok"], "t0_k": row["k"], "src": src})
    write_jsonl(a.out, out)
    rep = {"work": a.work, "index": a.index, "max_t0": a.max_t0, "stats": dict(st)}
    with open(a.out + ".report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))


def eval_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": r["name"],
        "prompt": r["prompt"],
        "reference_lines": r["n_lines"],
        "family": "hard12r_cand",
        "knob": r["n_lines"],
        "reference_proof": r["proof"],
    }


def cmd_evalset(a: argparse.Namespace) -> None:
    rs = [r for p in a.inp for r in read_jsonl(p)]
    names = [r["name"] for r in rs]
    assert len(set(names)) == len(names), "duplicate candidate names"
    ev = [eval_row(r) for r in rs]
    if a.shards == 1:
        write_jsonl(a.out, ev)
    else:
        for s in range(a.shards):
            write_jsonl(a.out.replace(".jsonl", f"_s{s}.jsonl"), ev[s :: a.shards])
    print(json.dumps({"rows": len(ev), "out": a.out, "shards": a.shards}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prior")
    p.add_argument("--work", default=f"{POOLS}/hard12_work")
    p.add_argument("--index", required=True)
    p.add_argument("--max_t0", type=int, default=1)
    p.add_argument("--out", required=True)
    e = sub.add_parser("evalset")
    e.add_argument("--in", dest="inp", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--shards", type=int, default=1)
    a = ap.parse_args(argv)
    {"prior": cmd_prior, "evalset": cmd_evalset}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
