"""Renaming-class exclusion and dedupe for hard12 (keys: `code.thinking.data.class_keys`, as
`data.EvalIndex`).

    X=code.experiments.current.thinking_traces_20261005.pools.hard12.exclude
    python -m $X index --out runs/pools/hard12_work/excl_index.pkl
    python -m $X filter --index excl_index.pkl --in cands.jsonl --out cands_x.jsonl \
        [--seen other.jsonl ...]

A row is dropped when its atom-renaming class key (the fork's `canon_key`) or its
premise-order-invariant key (ndbench `pkey`) is in an indexed file, or in a `--seen` file, or
repeats an earlier row of the input. The index reuses `test_21_30/seal.py`'s `build_index` / `hits`
over `exclusion_files()` below:
  * K12 train, the K12 held-out set and the cap-6 Stage-1 set (`bench/dan_ckpts/data/`);
  * every jsonl under the fork's data/ (ladder + take-home rl_targets, transfer, transfer_long*, p2,
    r1-r3, bs, gt, lj, ..., and textbook72 dev + train under eval_only/);
  * the OOD sets (`think/ood_check/sets/`);
  * the existing hard pools `runs/pools/rl_hard*.jsonl` (rl_hard, rl_hard_filtered; never this pool,
    rl_hard12);
  * `runs/pools/test_21_30/test_21_30.jsonl` when it exists.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import sys

from code.experiments.current.thinking_traces_20261005.test_21_30.seal import build_index, hits
from code.thinking.data import class_keys

FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
THINK = "/scratch/cowifornia/nd-rl/think"
OWN = {"rl_hard12.jsonl"}


def exclusion_files() -> list[str]:
    fs = [
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl",
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl",
        "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_depth3_f0_a1.jsonl",
    ]
    fs += sorted(glob.glob(f"{FORK}/data/**/*.jsonl", recursive=True))
    fs += sorted(glob.glob(f"{THINK}/ood_check/sets/*.jsonl"))
    fs += [
        f
        for f in sorted(glob.glob(f"{THINK}/runs/pools/rl_hard*.jsonl"))
        if os.path.basename(f) not in OWN
    ]
    fs += [f"{THINK}/runs/pools/test_21_30/test_21_30.jsonl"]
    return [f for f in fs if os.path.exists(f)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("index")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("filter")
    a2.add_argument("--index", required=True)
    a2.add_argument("--in", dest="inp", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--seen", nargs="*", default=[], help="further files whose classes are dropped")
    a = ap.parse_args(argv)
    if a.cmd == "index":
        idx = build_index(exclusion_files())
        with open(a.out, "wb") as f:
            pickle.dump(idx, f)
        print(
            f"classes {len(idx['keys'])}, pkeys {len(idx['pkeys'])}, files {len(idx['manifest'])}"
        )
        return 0
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    seen_k: set[str] = set()
    seen_pk: set[str] = set()
    for p in a.seen:
        with open(p) as f:
            for line in f:
                k, pk = class_keys(json.loads(line)["prompt"])
                seen_k.add(k)
                if pk is not None:
                    seen_pk.add(pk)
    st: collections.Counter[str] = collections.Counter()
    per: collections.Counter[str] = collections.Counter()
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            st["in"] += 1
            h = hits(idx, r["prompt"])
            if h:
                st["excluded"] += 1
                per.update(os.path.basename(x) for x in h)
                continue
            k, pk = class_keys(r["prompt"])
            if k in seen_k or (pk is not None and pk in seen_pk):
                st["dup"] += 1
                continue
            seen_k.add(k)
            if pk is not None:
                seen_pk.add(pk)
            st["kept"] += 1
            fo.write(line)
    rep = {
        "in": a.inp,
        "out": a.out,
        "seen": a.seen,
        "stats": dict(st),
        "excluded_by_file": dict(per.most_common()),
    }
    with open(a.out + ".report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
