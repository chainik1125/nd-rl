#!/usr/bin/env python3
"""Size and proof-length histogram of every pool the harness uses, for one fork commit.

  python3 pools.py <fork-checkout>      # writes pools/<fork-commit>.json (commit from remote.sh)

Run once per fork commit; params.py refuses to write params.json without the matching file. The
split logic (dev = sha1 parity of key, heldout = 2000 evenly spaced, targets = seeded sample of
N_TARGETS) copies harness.py so the numbers describe exactly what the harness will see.
"""

import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from params import HERE, fork_commit, harness_rules

Rec = dict[str, Any]


def read(fn: Path) -> list[Rec]:
    return [json.loads(line) for line in fn.read_text().splitlines() if line.strip()]


def hist(recs: list[Rec]) -> dict[str, Any]:
    c = collections.Counter(int(r["n_lines"]) for r in recs)
    return {"n": len(recs), "by_lines": {str(k): c[k] for k in sorted(c)}}


def parity(r: Rec) -> int:
    return int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2


def main() -> None:
    fork, commit, H = Path(sys.argv[1]), fork_commit(), harness_rules()
    train = read(fork / "data/train.jsonl")
    heldout = read(fork / "data/heldout.jsonl")
    heldout = [heldout[i * len(heldout) // 2000] for i in range(2000)]
    targets_all = read(fork / "data/ladder/rl_targets.jsonl")
    transfer = read(fork / "data/ladder/transfer.jsonl")
    n_targets = min(int(H["N_TARGETS"]), len(targets_all))
    out = {
        "fork_commit": commit,
        "seed": 0,
        "pretrain": {"file": "data/train.jsonl", **hist(train)},
        "pretrain_heldout": {"file": "data/heldout.jsonl (2000 evenly spaced)", **hist(heldout)},
        "rl_targets": {
            "file": "data/ladder/rl_targets.jsonl",
            "pool": hist(targets_all),
            "sampled": hist(random.Random(0).sample(targets_all, n_targets)),
        },
        "transfer_dev": {
            "file": "data/ladder/transfer.jsonl (sha1 parity 0)",
            **hist([r for r in transfer if parity(r) == 0]),
        },
        "transfer_holdout": {
            "file": "data/ladder/transfer.jsonl (sha1 parity 1)",
            **hist([r for r in transfer if parity(r) == 1]),
        },
    }
    fn = HERE / "pools" / f"{commit[:7]}.json"
    fn.write_text(json.dumps(out, indent=1) + "\n")
    print(fn, {k: v["n"] for k, v in out.items() if isinstance(v, dict) and "n" in v})


if __name__ == "__main__":
    main()
