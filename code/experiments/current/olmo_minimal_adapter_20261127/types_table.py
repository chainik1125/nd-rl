# ruff: noqa: E501
"""IFBench instruction ids of the 48 reach prompts against the train prompts of each nested prefix n = 1, 4, 16, 64.

    types_table.py [--pool /scratch/.../v2/pools/dpo/own] [--out-stem /scratch/.../v2/types_table]

Per reach prompt and prefix: `all` / `some` / `none` of its instruction ids occur among the instruction ids of the first n train
prompts (the pool manifest's train_pids order). Writes <stem>.csv (one row per reach prompt) and <stem>.json (summary counts).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
POOL = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2/pools/dpo/own")
STEM = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2/types_table")
NS = (1, 4, 16, 64)


def ids_of(row: dict) -> list[str]:
    iid = row["instruction_id_list"]
    return list(json.loads(iid.replace("'", '"')) if isinstance(iid, str) else iid)


def classify(reach_ids: list[str], seen: set[str]) -> str:
    hit = [i in seen for i in reach_ids]
    return "all" if all(hit) else ("some" if any(hit) else "none")


def build(dan: Path, pool: Path, ns=NS) -> tuple[list[dict], dict]:
    probs = {
        r["pid"]: r
        for r in map(json.loads, (dan / "olmo_ifomega/problems.jsonl").read_text().splitlines())
    }
    reach = (dan / "olmo_ifcurve/pids_reach.txt").read_text().split()
    train = json.loads((pool / "MANIFEST.json").read_text())["train_pids"]
    rows = []
    for p in reach:
        ids = ids_of(probs[p])
        row = {"pid": p, "instruction_ids": ";".join(ids)}
        for n in ns:
            seen = {i for q in train[:n] for i in ids_of(probs[q])}
            row[f"n{n}"] = classify(ids, seen)
            row[f"n{n}_shared_ids"] = ";".join(i for i in ids if i in seen)
        rows.append(row)
    summary = {
        "pool": str(pool),
        "n_reach": len(reach),
        "train_ids_by_n": {
            str(n): sorted({i for q in train[:n] for i in ids_of(probs[q])}) for n in ns
        },
        "counts": {
            str(n): {k: sum(r[f"n{n}"] == k for r in rows) for k in ("all", "some", "none")}
            for n in ns
        },
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dan", type=Path, default=DAN)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--out-stem", type=Path, default=STEM)
    a = ap.parse_args()
    rows, summary = build(a.dan, a.pool)
    with open(a.out_stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a.out_stem.with_suffix(".json").write_text(
        json.dumps({**summary, "rows": rows}, indent=1) + "\n"
    )
    for n, c in summary["counts"].items():
        print(
            f"n={n}: all {c['all']}, some {c['some']}, none {c['none']} of {summary['n_reach']} reach prompts"
        )


if __name__ == "__main__":
    main()
