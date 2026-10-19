"""Exclusion index for hard712: every file test_13_20 was sealed against (`test_13_20/excl.py`,
itself `test_21_30/seal.exclusion_files()` plus the hard12 pools and test sets), plus the sets
built since and every other prompt-bearing file under think/.

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.hard712.excl \
        --out /scratch/cowifornia/nd-rl/think/runs/pools/hard712/work/excl_index.pkl

Keys are `code.thinking.data.class_keys` (atom-renaming class with premise order kept, and the
premise-order-invariant `pkey`), so a candidate is excluded when any renaming or premise
permutation of it is in an indexed file. Covered (`files()`):

* test_13_20's 120 files: K12 train + held-out + the cap-6 Stage-1 set; every jsonl under the
  fork's data/ (ladder and take-home pools, transfer sets, p2 / r1-r3 / bs / gt / lj, and
  **textbook72 dev + train** under eval_only/); the OOD sets; the K12-prompt trace files; the RL
  pool candidate files and pools under runs/pools/ (rl_hard*, cands*, long*, short*); the hard12
  test sets (hard12_test, **hard12_test_1k**) and work files; **test_21_30**;
* added here: **test_13_20** and its calibration set, all of runs/pools/test_21_30/, the hard12r
  candidate and pilot sets, the ood_eval_20261005 sets, every trace file under traces/ (dev,
  graded, audit), and the trace-generation chunks under gen_runs/.

The hard712 outputs themselves (runs/pools/hard712/) are never indexed.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

from code.experiments.current.thinking_traces_20261005.test_13_20.excl import files as files_1320
from code.experiments.current.thinking_traces_20261005.test_21_30.seal import build_index

THINK = "/scratch/cowifornia/nd-rl/think"
TEXTBOOK = f"{THINK}/nd-takehome/data/eval_only/textbook72"
EXTRA_GLOBS = [
    f"{THINK}/runs/pools/test_13_20/*.jsonl",
    f"{THINK}/runs/pools/test_21_30/*.jsonl",
    f"{THINK}/runs/pools/hard12r/work/cands/*.jsonl",
    f"{THINK}/runs/pools/hard12r/work/sets/*.jsonl",
    f"{THINK}/ood_eval_20261005/sets/*.jsonl",
    f"{THINK}/traces/*.jsonl",
    f"{THINK}/traces/*/*.jsonl",
    f"{THINK}/gen_runs/**/*.jsonl",
]
REQUIRED = [
    f"{TEXTBOOK}/textbook_dev.jsonl",
    f"{TEXTBOOK}/textbook_train.jsonl",
    f"{THINK}/runs/restart2_hard12/hard12_test_1k.jsonl",
    f"{THINK}/runs/pools/hard12_test/hard12_test.jsonl",
    f"{THINK}/runs/pools/test_13_20/test_13_20.jsonl",
    f"{THINK}/runs/pools/test_21_30/test_21_30.jsonl",
    f"{THINK}/runs/pools/rl_hard12.jsonl",
]


def files() -> list[str]:
    fs = list(files_1320())
    for g in EXTRA_GLOBS:
        fs += sorted(glob.glob(g, recursive=True))
    fs += REQUIRED
    out: list[str] = []
    for f in fs:
        if (
            os.path.exists(f)
            and f not in out
            and not f.endswith(".keys.jsonl")
            and "/runs/pools/hard712/" not in f
        ):
            out.append(f)
    missing = [f for f in REQUIRED if f not in out]
    assert not missing, missing
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
