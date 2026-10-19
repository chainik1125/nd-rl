"""Length-stratified calibration subsample of a test set (equal count per reference length, seeded).

python subsample.py --set test_13_20.jsonl --out calib600.jsonl --n 600 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    with open(a.set) as f:
        for line in f:
            r = json.loads(line)
            by[r["reference_lines"]].append(r)
    rng = random.Random(a.seed)
    lens = sorted(by)
    per = a.n // len(lens)
    out = []
    for L in lens:
        xs = by[L][:]
        rng.shuffle(xs)
        out += xs[:per]
    with open(a.out, "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    per_len = collections.Counter(r["reference_lines"] for r in out)
    print(json.dumps(dict(n=len(out), per_len=dict(per_len))))


if __name__ == "__main__":
    main()
