"""Atom-count pools: Stage-1 sets whose every training record uses exactly 1, 2, 3 or 4 distinct sentence letters
(Charles, 2026-10-08: "pools with only 1, 2, 3, and 4 symbols respectively"). The tokenizers in use (the pinned fork's
and lean_seq / lean_staten) have exactly four atoms, P Q R S, so atoms4 is the ceiling and atoms1 the floor.

Recipe (the size ladder's, restricted by atom count):
  - candidates = K12's own rows with k atoms (already verified; they are generator rows of the same kind) followed by the
    generator population under data/pretrain_data/gen/ (cap-6 forward mode for 2–6 lines, long mode for 7–12) with k atoms;
  - one record per curry-aware renaming class (`BR.keys`), K12's row first when it is in the class;
  - every generator proof re-verified by the pinned verifier (`make_record`), box depth ≤ 2 like K12;
  - audited against every evaluation set and the sealed 72 (`EvalIndex.audit`, seed-split twins kept and tagged);
  - flat over pruned lengths 2–12 at up to (155,000 − 1,024) / 11 records per length; deficits reported, never filled with
    permutations (a renaming of a k-atom record is the same class);
  - records 0–1,023 are K12's first 1,024 rows, as in every other pool (the pinned trainer's held-out val slice, so val_loss
    is comparable); the training rows after them are pure k-atom records. Dan's trainer does not use that head.

    cd code && nohup uv run python pretrain_data/atom_pools.py > ../data/pretrain_data/atom_pools.log 2>&1 &

Writes data/pretrain_data/pools/atoms/atoms{1,2,3,4}.jsonl.gz, stats.json, and the section "## Atom pools" in
docs/pretrain-data/pools.md (`--doc_only` rewrites it from stats.json).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402

OUT_DIR = os.path.join(PC.DATA, "pools", "atoms")
GEN_DIR = os.path.join(PC.DATA, "gen")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
HEAD = "## Atom pools: every training record uses exactly 1, 2, 3 or 4 sentence letters"
VAL_N = 1024
TOTAL = 155_000
ATOM_RE = re.compile(r"\b[PQRS]\b")


def n_atoms(prompt):
    return len(set(ATOM_RE.findall(prompt)))


def write_doc(stats):
    W = [f"\n{HEAD}\n",
         "Built by `atom_pools.py` (2026-10-08). Candidates are K12's own rows with the given number of distinct atoms, then the generator "
         "population under `data/pretrain_data/gen/` (the size ladder's files: cap-6 forward mode for 2–6 lines, long mode for 7–12) with that "
         "number of atoms; one record per curry-aware renaming class (K12's row wins its class; the per-length draw is then random over both sources), every generator proof re-verified by the pinned verifier, "
         "box depth ≤ 2 like K12, audited against every evaluation set and the sealed 72. Flat over lengths 2–12 at up to "
         f"{(TOTAL - VAL_N) // 11:,} per length; a deficit means the population ran out of distinct classes at that length (no permutations: a "
         "renaming of a k-atom record is the same class). Records 0–1,023 are K12's first 1,024 rows in every pool (the pinned trainer's "
         "held-out val slice), so the training rows after them are the pure k-atom set; `n train` counts those. Pools smaller than K12 are "
         "seen more times per 1,200 s pretrain (`epochs`; Dan's stage1-dynamics found steps, not data size, set the result).\n",
         "| pool | n train | from K12 | from generator | epochs (9.56M) | deficit by length | kept seed-set twins | removed eval twins | removed sealed twins | lines | premises | depth |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in stats:
        W.append(f"| {s['label']} | {s['n_train']:,} | {s['from_k12']:,} | {s['from_gen']:,} | {s['epochs']} | "
                 f"{json.dumps(s['deficit']) if s['deficit'] else '—'} | {s['audit'].get('kept_seed_set_twin', 0)} | "
                 f"{s['audit'].get('removed_eval_twin', 0)} | {s['audit'].get('removed_sealed_twin', 0)} | {s['by_len']} | {s['by_prem']} | {s['by_depth']} |")
    W.append("")
    W.append(f"Generator population by atom count (rows before dedup and verification): {json.dumps(stats[0]['population'])}. "
             "K12 itself: " + json.dumps(stats[0]["k12_by_atoms"]) + " records by atom count.\n")
    text = open(DOC).read() if os.path.exists(DOC) else ""
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as fh:
        fh.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--doc_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="read at most this many generator rows per file (smoke test)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.doc_only:
        write_doc(json.load(open(os.path.join(OUT_DIR, "stats.json")))["pools"])
        print("rewrote the atom-pools section")
        return
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    head = k12[:VAL_N]
    per_len = (TOTAL - VAL_N) // 11
    k12_by_atoms = collections.Counter(n_atoms(r["prompt"]) for r in k12)
    print("k12", len(k12), dict(sorted(k12_by_atoms.items())), "per length target", per_len, flush=True)

    # candidates by atom count: K12's own rows first (verified already), then the generator population
    cand = {k: collections.defaultdict(list) for k in (1, 2, 3, 4)}  # k -> length -> records
    seen = {k: set() for k in (1, 2, 3, 4)}
    src = {k: collections.Counter() for k in (1, 2, 3, 4)}
    for r in k12[VAL_N:]:
        k = n_atoms(r["prompt"])
        if k not in cand:
            continue
        ks = PC.BR.keys(r["prompt"])
        if ks & seen[k]:
            continue
        seen[k] |= ks
        cand[k][r["n_lines"]].append(dict(r, pool=f"atoms{k}", origin="k12"))
        src[k]["k12"] += 1
    population = collections.Counter()
    rep = collections.Counter()
    for path in sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl"))):
        n_read = 0
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    rep["bad_line"] += 1
                    continue
                if "prompt" not in r:
                    continue
                n_read += 1
                if a.limit and n_read > a.limit:
                    break
                k = n_atoms(r["prompt"])
                population[k] += 1
                if k not in cand:
                    continue
                L = r.get("n_lines") or r["proof"].count(";") + 1
                if len(cand[k][L]) >= 3 * per_len:  # plenty for this (k, length); skip the expensive steps
                    rep["skipped_full"] += 1
                    continue
                ks = PC.BR.keys(r["prompt"])
                if ks & seen[k]:
                    rep["dup"] += 1
                    continue
                seen[k] |= ks
                rec, why = PC.make_record(r["prompt"], r["proof"], f"atoms{k}", contra_prem=r.get("contra_prem", False), origin="gen")
                if rec is None:
                    rep["verify_fail"] += 1
                    continue
                depth = max(ln["depth"] for ln in PC.parse(rec["proof"]))
                if depth > a.max_depth:
                    rep["depth_over"] += 1
                    continue
                cand[k][rec["n_lines"]].append(rec)
                src[k]["gen"] += 1
        print(os.path.basename(path), "read", n_read, dict(rep), flush=True)
    print("population by atoms", dict(sorted(population.items())), flush=True)

    idx = PC.EvalIndex()
    stats_out = []
    for k in (1, 2, 3, 4):
        rows, deficit, audit = [], {}, collections.Counter()
        for L in range(2, 13):
            rs = cand[k].get(L, [])
            rng.shuffle(rs)
            kept, arep = idx.audit(rs)
            for kk, v in arep.items():
                if kk != "kept":
                    audit[kk] += v
            take = kept[:per_len]
            rows.extend(take)
            if len(take) < per_len:
                deficit[L] = per_len - len(take)
        rng.shuffle(rows)
        rows = [dict(r, slice=True) for r in rows]  # the screens treat `slice` rows as the pool's new material
        pool = head + rows
        name = f"atoms{k}"
        PC.write(pool, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
        s = PC.stats(rows, name)
        s.update(n_train=len(rows), n_total=len(pool), from_k12=sum(1 for r in rows if r.get("origin") == "k12"),
                 from_gen=sum(1 for r in rows if r.get("origin") == "gen"), deficit=deficit, audit=dict(audit),
                 epochs=round(3_180_545 / max(1, len(pool)), 1),
                 by_len_counts=dict(sorted(collections.Counter(r["n_lines"] for r in rows).items())),
                 population=dict(sorted(population.items())), k12_by_atoms=dict(sorted(k12_by_atoms.items())))
        stats_out.append(s)
        print(name, "train", len(rows), "k12", s["from_k12"], "gen", s["from_gen"], "deficit", deficit, dict(audit), flush=True)
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(dict(pools=stats_out, gen_report=dict(rep)), f, indent=1, ensure_ascii=False)
    write_doc(stats_out)
    print("wrote", OUT_DIR, "and the atom-pools section of pools.md")


if __name__ == "__main__":
    main()
