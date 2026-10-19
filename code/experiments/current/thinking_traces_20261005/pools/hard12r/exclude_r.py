"""Renaming-class exclusion index for hard12r: hard12's index files plus hard12 itself and the
newer sealed sets.

    X=code.experiments.current.thinking_traces_20261005.pools.hard12r.exclude_r
    python -m $X index --out runs/pools/hard12r/work/excl_index.pkl
    python -m $X filter --index excl_index.pkl --in cands.jsonl --out cands_x.jsonl [--seen ...]

`index` indexes `hard12.exclude.exclusion_files()` (K12 train / held-out / cap-6, all fork data
including textbook72, the OOD sets, rl_hard*.jsonl, test_21_30.jsonl) plus `EXTRA` below:
rl_hard12.jsonl (now excluded: it is the previous pool), the hard12 test set and its 1k subset,
and every top-level jsonl under runs/pools/test_13_20/ and test_21_30/ (calibration and sealed
sets). `filter` is hard12's (`hard12.exclude.main`), unchanged.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

from code.experiments.current.thinking_traces_20261005.pools.hard12 import exclude as h12x
from code.experiments.current.thinking_traces_20261005.test_21_30.seal import build_index

R = "/scratch/cowifornia/nd-rl/think/runs"
EXTRA_FIXED = [
    f"{R}/pools/rl_hard12.jsonl",
    f"{R}/pools/hard12_test/hard12_test.jsonl",
    f"{R}/restart2_hard12/hard12_test_1k.jsonl",
]


def exclusion_files() -> list[str]:
    fs = h12x.exclusion_files() + EXTRA_FIXED
    for d in ("test_13_20", "test_21_30"):
        fs += sorted(
            f
            for f in glob.glob(f"{R}/pools/{d}/*.jsonl")
            if not f.endswith(".keys.jsonl")  # sidecars hold keys, not prompts
        )
    out: list[str] = []
    for f in fs:
        if os.path.exists(f) and f not in out:
            out.append(f)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "index":
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("cmd")
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv)
        idx = build_index(exclusion_files())
        with open(a.out, "wb") as f:
            pickle.dump(idx, f)
        print(
            f"classes {len(idx['keys'])}, pkeys {len(idx['pkeys'])}, files {len(idx['manifest'])}"
        )
        return 0
    return h12x.main(argv)


if __name__ == "__main__":
    sys.exit(main())
