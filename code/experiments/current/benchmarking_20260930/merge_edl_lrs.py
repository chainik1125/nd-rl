"""Merge the EDL learning-rate extension into one run per model (interp audit, 2026-10-01).

`edlx-<model>-64-64` holds lr 0 / 1e-5 / 3e-5 rows run after `edl2-<model>-64-64` (lr 1e-4 /
3e-4 / 1e-3) with identical classes, orders, n, m and per-(class, lr, order) seeds; together they
are the 6-lr sweep. This writes `edl3-<model>-64-64` (the newest generation, which the EDL view
reads by default) and moves the `edlx-*` part file to `out/quarantine/edl_parts/` so it is not
also read as a separate run. Refuses if the two parts disagree on classes/orders/n/m or share a lr.

    code/.venv/bin/python code/experiments/current/benchmarking_20260930/merge_edl_lrs.py base rl
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OUT = Path("/scratch/cowifornia/nd-rl/bench/out")
EDL = OUT / "store" / "edl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge(model: str) -> None:
    old, ext = EDL / f"edl2-{model}-64-64.jsonl", EDL / f"edlx-{model}-64-64.jsonl"
    dest = EDL / f"edl3-{model}-64-64.jsonl"
    a, b = _rows(old), _rows(ext)

    def shape(rows: list[dict]) -> set[tuple]:
        return {(r["class"], r["order"], r["n"], r["m"]) for r in rows}

    if shape(a) != shape(b):
        raise SystemExit(f"{model}: classes/orders/n/m differ between {old.name} and {ext.name}")
    if {r["lr"] for r in a} & {r["lr"] for r in b}:
        raise SystemExit(f"{model}: the two parts share a learning rate")
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    dest.write_text("".join(json.dumps(r) + "\n" for r in a + b))
    parts = OUT / "quarantine" / "edl_parts"
    parts.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ext), parts / ext.name)
    lrs = sorted({r["lr"] for r in a + b})
    print(f"{dest.name}: {len(a) + len(b)} rows, lrs {lrs}; moved {ext.name} to {parts}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["base", "rl"]:
        merge(m)
