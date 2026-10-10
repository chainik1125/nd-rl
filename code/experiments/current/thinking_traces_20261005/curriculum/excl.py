"""Exclusion index for the length-curriculum pools: hard712's index plus everything built since.

    cd /scratch/cowifornia/nd-rl/think/nd-rl-curr
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
        /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.curriculum.excl \
        --base /scratch/cowifornia/nd-rl/think/runs/pools/hard712/work/excl_index.pkl \
        --out /scratch/cowifornia/nd-rl/think/runs/pools/curriculum/work/excl_index.pkl

Starts from hard712's index (364 files: K12 train + held-out, the cap-6 Stage-1 set, every jsonl
under the fork's data/ including textbook72 dev + train, the OOD sets, rl_hard12 and the other RL
pools, hard12_test_1k, test_13_20, test_21_30, trace files; `hard712/excl.py`) and adds, with the
same class keys (atom renaming, premise permutation):

* hard712's own outputs: `pool.jsonl`, `test_1k.jsonl`, `dev_500.jsonl`, the two RL samples
  `pool_s5k_a/b.jsonl`, and `work/gated.jsonl` (every gated row, so spare rows too);
* every prompt-bearing jsonl under `runs/pools/` one or two levels down and under
  `runs/pools/*/work/` (test_13_20's candidate composites and certification files, hard12r, the
  canary set, ...), except this curriculum's own outputs and hard712's abandoned pilot runs;
* the eval subsets under `runs/eval/matched/sets/`.

Rows reused from hard712 are checked against this index with hard712's own pool / gated / spare
files ignored (`reuse.py`), so the index can serve both the fresh generation and that check.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

from code.experiments.current.thinking_traces_20261005.test_21_30.seal import build_index

R = "/scratch/cowifornia/nd-rl/think/runs"
H712 = f"{R}/pools/hard712"
H712_OWN = [
    f"{H712}/pool.jsonl",
    f"{H712}/test_1k.jsonl",
    f"{H712}/dev_500.jsonl",
    f"{H712}/pool_s5k_a.jsonl",
    f"{H712}/pool_s5k_b.jsonl",
    f"{H712}/work/gated.jsonl",
]
GLOBS = [
    f"{R}/pools/*.jsonl",
    f"{R}/pools/*/*.jsonl",
    f"{R}/pools/*/work/*.jsonl",
    f"{R}/eval/matched/sets/*.jsonl",
]
REQUIRED = [
    *H712_OWN,
    f"{R}/pools/test_13_20/test_13_20.jsonl",
    f"{R}/pools/test_21_30/test_21_30.jsonl",
    f"{R}/pools/rl_hard12.jsonl",
    f"{R}/restart2_hard12/hard12_test_1k.jsonl",
]


def extra_files(have: set[str]) -> list[str]:
    fs = list(H712_OWN)
    for g in GLOBS:
        fs += sorted(glob.glob(g))
    out: list[str] = []
    for f in fs:
        if (
            os.path.exists(f)
            and f not in out
            and f not in have
            and not f.endswith(".keys.jsonl")
            and "/pools/curriculum/" not in f
            and "_abandoned/" not in f
            and not f.endswith("/rejects.jsonl")
        ):
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="hard712's excl_index.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.base, "rb") as f:
        idx = pickle.load(f)
    have = {m["file"] for m in idx["manifest"]}
    add = extra_files(have)
    new = build_index(add)
    for k, fs in new["keys"].items():
        idx["keys"].setdefault(k, set()).update(fs)
    for k, fs in new["pkeys"].items():
        idx["pkeys"].setdefault(k, set()).update(fs)
    idx["manifest"] = idx["manifest"] + new["manifest"]
    files = {m["file"] for m in idx["manifest"]}
    missing = [f for f in REQUIRED if f not in files]
    assert not missing, missing
    with open(a.out, "wb") as f:
        pickle.dump(idx, f)
    print(
        f"classes {len(idx['keys'])}, pkeys {len(idx['pkeys'])}, files {len(idx['manifest'])} "
        f"({len(add)} added)"
    )


if __name__ == "__main__":
    main()
