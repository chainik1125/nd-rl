"""Exclusion index for test_13_20: every file test_21_30 was sealed against, plus the hard12 pools
and test sets, the other RL-pool candidate files now under runs/pools/, and the 21-30 test set.

    cd nd-rl && python -m code.experiments.current.thinking_traces_20261005.test_13_20.excl \
        --out /scratch/cowifornia/nd-rl/think/runs/pools/test_13_20/work/excl_index.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

from code.experiments.current.thinking_traces_20261005.test_21_30.seal import (
    build_index,
    exclusion_files,
)

THINK = "/scratch/cowifornia/nd-rl/think"
EXTRA = [
    f"{THINK}/runs/pools/rl_hard12.jsonl",
    f"{THINK}/runs/restart2_hard12/hard12_test_1k.jsonl",
    f"{THINK}/runs/pools/test_21_30/test_21_30.jsonl",
]
EXTRA_GLOBS = [
    f"{THINK}/runs/pools/*.jsonl",
    f"{THINK}/runs/pools/hard12_test/*.jsonl",
    f"{THINK}/runs/pools/hard12_work/*.jsonl",
    f"{THINK}/runs/restart2_hard12/*.jsonl",
]


def files() -> list[str]:
    fs = list(exclusion_files()) + EXTRA
    for g in EXTRA_GLOBS:
        fs += sorted(glob.glob(g))
    out = []
    for f in fs:
        if os.path.exists(f) and f not in out and not f.endswith(".keys.jsonl"):
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = build_index(files())
    with open(a.out, "wb") as f:
        pickle.dump(idx, f)
    print(f"classes {len(idx['keys'])}, pkeys {len(idx['pkeys'])}, files {len(idx['manifest'])}")


if __name__ == "__main__":
    main()
