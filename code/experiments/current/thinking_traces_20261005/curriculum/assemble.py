"""One band's reservoir for `hard712/run.py select`: reused hard712 rows plus fresh rows.

    python -m code.experiments.current.thinking_traces_20261005.curriculum.assemble \
        --band_dir .../curriculum/L1 --reused .../L1/reused.jsonl \
        --fresh gen=.../L1/gen/work/gated_lean.jsonl regen=.../L1/regen/work/gated_lean.jsonl \
        [--avoid .../L1/work/gated_lean.jsonl ...]

Writes `BAND_DIR/work/gated_lean.jsonl`, the file `run.py select` reads:

* reused rows (`reuse.py`) carry hard712's Lean result: every one of hard712's 29,542 gated rows
  passed Lean on both proofs, so they get `lean_ok = lean_ref_ok = True`; the final rows of every
  band are Lean-checked again by `report.py`;
* fresh rows (`NAME=PATH`, NAME the origin label: `gen` for `run.py generate`, `regen` for
  `regen.py`) are the band's own `run.py lean` output (constructed proof accepted);
* a row whose class (`pkey`) is in an `--avoid` file (an earlier band's reservoir) or already
  kept is dropped, so the bands are disjoint by renaming class even where two constructions of
  one theorem gave different shortest lengths;
* a row that matches a held-out problem (Charles's 2026-10-02 textbook release or textbook72) or
  a sealed test set (test_13_20, test_21_30, hard12_test_1k, hard712 test_1k / dev_500) as
  written or in curried normal form (`heldout.py`) is dropped;
* the class comparisons above (earlier bands, duplicates) are also made in curried normal form.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO


def jl(p: str | Path) -> list[dict[str, Any]]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band_dir", required=True)
    ap.add_argument("--reused", default="")
    ap.add_argument("--fresh", nargs="*", default=[])
    ap.add_argument("--avoid", nargs="*", default=[])
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    a = ap.parse_args()
    hidx = HO.heldout_index({**HO.FILES, **HO.SEALED})
    held_by: collections.Counter[str] = collections.Counter()
    avoid: set[str] = set()
    for p in a.avoid:
        for r in jl(p):
            avoid |= {r["pkey"]} | HO.keys(r["prompt"])
    seen: set[str] = set()
    out, c = [], collections.Counter()
    sources = [("reused", a.reused)] if a.reused else []
    sources += [tuple(s.split("=", 1)) if "=" in s else ("fresh", s) for s in a.fresh]
    for src, p in sources:
        for r in jl(p):
            c[f"{src}_in"] += 1
            if not a.lo <= r["min_lines"] <= a.hi:
                c[f"{src}_out_of_band"] += 1
                continue
            ck = HO.keys(r["prompt"])
            if r["pkey"] in avoid or ck & avoid:
                c[f"{src}_in_earlier_band"] += 1
                continue
            if r["pkey"] in seen or ck & seen:
                c[f"{src}_dup"] += 1
                continue
            h = HO.hits(hidx, r["prompt"])
            if h:
                c[f"{src}_heldout"] += 1
                held_by.update(h)
                continue
            if src == "reused":
                r["lean_ok"] = r["lean_ref_ok"] = True
            elif not r.get("lean_ok"):
                c[f"{src}_lean_rejected"] += 1
                continue
            r["origin"] = src
            seen.add(r["pkey"])
            seen |= ck
            out.append(r)
            c[f"{src}_kept"] += 1
    work = Path(a.band_dir) / "work"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "gated_lean.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    rep = dict(
        counts=dict(c),
        heldout_removed_by_file=dict(held_by),
        n=len(out),
        min_lines=dict(sorted(collections.Counter(r["min_lines"] for r in out).items())),
    )
    with open(work / "assemble_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
