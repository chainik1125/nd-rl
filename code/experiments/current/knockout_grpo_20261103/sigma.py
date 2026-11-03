"""Dan's sigma labels for a corpus of ND proofs ({id?, prompt, proof, n_lines}).

    python -m code.experiments.current.knockout_grpo_20261103.sigma FILE.jsonl[.gz] --name NAME

Writes OUT/sigma/<NAME>.labels.jsonl (one row per record: id, n_lines, the flags of
`common.label`) and OUT/sigma/<NAME>.json (counts, shares, per-length counts). `--workers` (8).
Positive control: on Dan's K12 it must give his 16,703 sigma and 16,961 sigma-plus records.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp

from .common import OUT, jsonl, label, sha256


def _one(r: dict) -> dict:
    lab = label(r["proof"])
    return {"id": r.get("id"), "n_lines": r.get("n_lines"), **lab}


def run(path: str, name: str, workers: int = 8) -> dict:
    rows = list(jsonl(path))
    with mp.Pool(workers) as pool:
        labs = pool.map(_one, rows, chunksize=500)
    d = OUT / "sigma"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.labels.jsonl", "w") as f:
        for x in labs:
            f.write(json.dumps(x) + "\n")
    c: collections.Counter = collections.Counter()
    by_len: dict = collections.defaultdict(collections.Counter)
    for x in labs:
        c["records"] += 1
        L = x["n_lines"]
        by_len[L]["records"] += 1
        if not x.get("parse"):
            c["parse_fail"] += 1
            continue
        for k in ("sigma", "sigma_any", "sigma_plus", "lemma_route", "dn_any"):
            if x[k]:
                c[k] += 1
                by_len[L][k] += 1
    n = c["records"]
    rep = {
        "file": str(path),
        "sha256": sha256(path),
        "counts": dict(c),
        "share_pct": {k: 100.0 * c[k] / n for k in ("sigma", "sigma_any", "sigma_plus")},
        "by_length": {str(k): dict(v) for k, v in sorted(by_len.items(), key=lambda t: t[0] or 0)},
    }
    (d / f"{name}.json").write_text(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rep = run(a.path, a.name, a.workers)
    print(json.dumps({k: rep[k] for k in ("counts", "share_pct")}, indent=1))


if __name__ == "__main__":
    main()
