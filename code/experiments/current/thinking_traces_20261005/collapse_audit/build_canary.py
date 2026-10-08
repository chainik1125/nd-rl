"""Collapse-audit fix: build the long-context canary file (`code.thinking.canary`, canary/1).

    M=code.experiments.current.thinking_traces_20261005.collapse_audit
    code/.venv/bin/python -m $M.build_canary \
        --out /scratch/cowifornia/nd-rl/think/runs/pools/canary_long.jsonl

Source: the generator pools of long proofs, `runs/pools/long_cands.jsonl` and `long2_cands.jsonl`
(13-20 reference lines, generated for the RL pool and never used in training). Steps: each proof's
reference steps whose flat state holds >= `--min_hyp` (10) hypothesis lines (`canary.proof_row`).

Exclusion (renaming classes, `code.thinking.data.class_keys`: the atom-renaming key and the
premise-order-invariant key; a record is out when either key matches):
* a fresh index (`test_21_30.seal.build_index`, the sealing code of both test sets) over every
  sealed set: hard12_test_1k (and the hard12_test file it was drawn from), test_13_20 (and the
  matched 1,000-item eval subset), test_21_30, textbook72 (dev and train), plus the two 600-item
  calibration sets. Any candidate whose class is in it is dropped.
* a cross-check against the existing exclusion index of test_13_20
  (`runs/pools/test_13_20/work/excl_index.pkl`, `test_13_20/excl.py`), counting hits from its
  sealed-set files only (the index also holds the candidate pools themselves).
* duplicates within the pools (the same atom-renaming class twice) keep the first.
The report (`<out>.meta.json`) gives both checks per set, which must be zero for the kept rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.test_21_30.seal import build_index
from code.thinking.canary import MIN_HYP, proof_row
from code.thinking.data import class_keys

THINK = "/scratch/cowifornia/nd-rl/think"
R = f"{THINK}/runs"
TB = f"{THINK}/nd-takehome/data/eval_only/textbook72"
SOURCES = {
    "long_cands": f"{R}/pools/long_cands.jsonl",
    "long2_cands": f"{R}/pools/long2_cands.jsonl",
}
SEALED = {
    "hard12_test_1k": [
        f"{R}/restart2_hard12/hard12_test_1k.jsonl",
        f"{R}/pools/hard12_test/hard12_test.jsonl",
    ],
    "test_13_20": [
        f"{R}/pools/test_13_20/test_13_20.jsonl",
        f"{R}/eval/matched/sets/t1320_n1000.jsonl",
    ],
    "test_21_30": [f"{R}/pools/test_21_30/test_21_30.jsonl"],
    "textbook72": [f"{TB}/textbook_dev.jsonl", f"{TB}/textbook_train.jsonl"],
    "calib (not sealed)": [
        f"{R}/pools/test_13_20/calib600_13_20.jsonl",
        f"{R}/pools/test_21_30/calib600_21_30.jsonl",
    ],
}
OLD_INDEX = f"{R}/pools/test_13_20/work/excl_index.pkl"


def hits(index: dict[str, Any], key: str, pkey: str | None) -> set[str]:
    out = set(index["keys"].get(key, ()))
    if pkey is not None:
        out |= set(index["pkeys"].get(pkey, ()))
    return out


def git_rev() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True)
    st = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True)
    return {"commit": rev.stdout.strip(), "dirty": bool(st.stdout.strip())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_hyp", type=int, default=MIN_HYP)
    a = ap.parse_args(argv)
    files = {f: s for s, fs in SEALED.items() for f in fs}
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, f"sealed files missing: {missing}"
    fresh = build_index(list(files))
    with open(OLD_INDEX, "rb") as fh:
        old = pickle.load(fh)
    old_sealed = {m["file"] for m in old["manifest"]} & set(files)
    seen: set[str] = set()
    excl: dict[str, collections.Counter[str]] = {"fresh": collections.Counter()}
    excl["old_index"] = collections.Counter()
    rows: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    hyp_hist: collections.Counter[int] = collections.Counter()
    lines_hist: collections.Counter[int] = collections.Counter()
    for src, path in SOURCES.items():
        with open(path) as fh:
            cands = [json.loads(x) for x in fh if x.strip()]
        for r in cands:
            stats[f"read {src}"] += 1
            key, pkey = class_keys(r["prompt"])
            h_new = hits(fresh, key, pkey)
            h_old = hits(old, key, pkey) & old_sealed
            for f in h_new:
                excl["fresh"][files[f]] += 1
            for f in h_old:
                excl["old_index"][files[f]] += 1
            if h_new or h_old:
                stats["dropped: sealed-set class"] += 1
                continue
            if key in seen:
                stats["dropped: duplicate class in the pools"] += 1
                continue
            seen.add(key)
            row = proof_row(
                r["name"],
                r["prompt"],
                r["proof"],
                a.min_hyp,
                source=src,
                n_lines=r.get("n_lines"),
                key=key,
                pkey=pkey,
            )
            if row is None:
                stats[f"dropped: no state with >= {a.min_hyp} hypothesis lines"] += 1
                continue
            rows.append(row)
            lines_hist[int(r.get("n_lines") or 0)] += 1
            for s in row["steps"]:
                hyp_hist[s["n_hyp"]] += 1
    # the kept rows, re-checked against both indexes
    recheck = sum(
        bool(hits(fresh, r["key"], r["pkey"]) or hits(old, r["key"], r["pkey"]) & old_sealed)
        for r in rows
    )
    assert recheck == 0, f"{recheck} kept rows hit a sealed set"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_steps = sum(len(r["steps"]) for r in rows)
    meta = {
        "written_on": time.strftime("%Y-%m-%d"),
        "builder": str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[5])),
        "git": git_rev(),
        "min_hyp": a.min_hyp,
        "sources": {
            s: {"path": p, "mtime": os.path.getmtime(p), "rows": stats[f"read {s}"]}
            for s, p in SOURCES.items()
        },
        "sealed_files": {s: fs for s, fs in SEALED.items()},
        "fresh_index_manifest": fresh["manifest"],
        "old_index": {"path": OLD_INDEX, "sealed_files_in_it": sorted(old_sealed)},
        "exclusion_hits_by_set": {k: dict(v) for k, v in excl.items()},
        "kept_rows_hitting_a_sealed_set": recheck,
        "counts": dict(stats),
        "proofs": len(rows),
        "steps": n_steps,
        "proofs_by_source": dict(collections.Counter(r["source"] for r in rows)),
        "proofs_by_reference_lines": dict(sorted(lines_hist.items())),
        "steps_by_hypothesis_lines": dict(sorted(hyp_hist.items())),
        "bytes": out.stat().st_size,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("counts", "proofs", "steps", "bytes")}, indent=1))
    print("exclusion hits (candidates dropped, per sealed set):", meta["exclusion_hits_by_set"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
