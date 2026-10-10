"""hard712 rows no RL run or sealed split has used, for one length band of the curriculum.

    python -m code.experiments.current.thinking_traces_20261005.curriculum.reuse \
        --lo 10 --hi 12 --index .../curriculum/work/excl_index.pkl \
        --out .../curriculum/L2/reused.jsonl

Source: `hard712/work/gated_lean.jsonl` (all 29,542 gated, Lean-checked rows: the 23,092 the mix
gate selected and the 6,450 spare rows). Dropped:

* rows of the sealed splits `test_1k` / `dev_500` and of the RL samples `pool_s5k_a` / `pool_s5k_b`
  (by premise-order-invariant class, `pkey`);
* rows outside [lo, hi] shortest-proof lines (`min_lines`);
* rows whose class hits any indexed file other than hard712's own pool and work files (gated,
  gated_lean, selected, spare; the index holds hard712's outputs too, so that the fresh
  generation cannot repeat them). This includes the sealed sets' composite pieces under
  `test_13_20/work/` and `test_21_30/work/`, which hard712's own index did not hold.

The kept rows are written in `gated.jsonl` format with the Lean flags removed, so the band's
`run.py lean` checks them again with its fresh rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle

H712 = "/scratch/cowifornia/nd-rl/think/runs/pools/hard712"
USED = [f"{H712}/{p}.jsonl" for p in ("test_1k", "dev_500", "pool_s5k_a", "pool_s5k_b")]
OWN = {
    f"{H712}/pool.jsonl",
    *(f"{H712}/work/{p}.jsonl" for p in ("gated", "gated_lean", "selected", "spare")),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    used: set[str] = set()
    for p in USED:
        with open(p) as f:
            used |= {json.loads(line)["pkey"] for line in f}
    c: collections.Counter[str] = collections.Counter()
    hit_files: collections.Counter[str] = collections.Counter()
    kept = []
    with open(f"{H712}/work/gated_lean.jsonl") as f:
        for line in f:
            r = json.loads(line)
            c["gated"] += 1
            if r["pkey"] in used:
                c["used_or_sealed"] += 1
                continue
            if not a.lo <= r["min_lines"] <= a.hi:
                c["out_of_band"] += 1
                continue
            h = (idx["keys"].get(r["key"], set()) | idx["pkeys"].get(r["pkey"], set())) - OWN
            if h:
                c["excluded"] += 1
                hit_files.update(h)
                continue
            r.pop("lean_ok", None)
            r.pop("lean_ref_ok", None)
            r["reused_from"] = "hard712"
            kept.append(r)
    c["kept"] = len(kept)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    lines = collections.Counter(r["min_lines"] for r in kept)
    rep = dict(counts=dict(c), hit_files=dict(hit_files), min_lines=dict(sorted(lines.items())))
    with open(a.out.replace(".jsonl", ".report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
