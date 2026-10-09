"""Compose the candidate Stage-1 pools from K12 and the new builders' outputs, and compare them at the data level.

Pools (all 155,000 records, K12's size, so a fixed wall-clock Stage-1 sees the same number of records):
  k12                 Dan's K12 unchanged (baseline)
  k12_minimal         K12 with every proof replaced by a minimal one (minimal_k12.py; idea B1)
  k12_joins1 / 10     K12 with a 1 % / 10 % random slice replaced by join problems (joins.py; idea A11)
  k12_sub1 / 10       K12 with a 1 % / 10 % slice replaced by textbook sub-problems × atom permutations (A1)
  k12_spice1          K12 with a 1 % slice: ½ joins, ¼ sub-problems, ¼ structural (four_prem + depth3 families) (C4)
  k12_minimal_joins10 minimal K12 with a 10 % join slice (B1 + A11)

Every record in every pool was verified by the pinned verifier when it was built; the audit against the evaluation
sets and the sealed 72 happened in the builders. The comparison table goes to docs/pretrain-data/pools.md.

    cd code && uv run python pretrain_data/mix.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402

OUT_DIR = os.path.join(PC.DATA, "pools")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
CAP6 = os.path.join(PC.ROOT, "code", "ws3", "fork_pin", "data", "train.jsonl.gz")  # the take-home Stage-1 set (reference row)
# sections of pools.md that other scripts or hands write; mix.py keeps them when it regenerates its own part
PRESERVED = ("## Reading", "## Size ladder", "## Scan pools", "## Data screens", "## New pools", "## Pool-to-pool similarity",
             "## Checkpoint screens", "## Atom pools", "## K pools")


def preserved_sections(path):
    """{heading: section text} for the preserved '## ' sections of an existing pools.md (each runs to the next '## ')"""
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    out = {}  # keyed by the section's full heading line, in file order
    starts = [i for i in range(len(text)) if text.startswith("\n## ", i)]
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(text)
        sec = text[i + 1: j]
        if any(sec.startswith(head) for head in PRESERVED):
            out[sec.split("\n", 1)[0]] = sec
    return out


def expand(rows, need, rng, label):
    """at least `need` records from `rows`: the rows, then their atom permutations, then repeats (recorded)."""
    if len(rows) >= need:
        return rng.sample(rows, need), dict(source=len(rows), perms=0, repeats=0)
    out = list(rows)
    perms = []
    for r in rows:
        perms.extend(permutations_without_original(r))
    rng.shuffle(perms)
    take = perms[: max(0, need - len(out))]
    out.extend(take)
    reps = 0
    while len(out) < need:
        out.append(dict(rng.choice(rows), repeat=True))
        reps += 1
    return out[:need], dict(source=len(rows), perms=len(take), repeats=reps)


def permutations_without_original(r):
    return [p for p in PC.permutations_of(r) if p["prompt"] != r["prompt"]]


VAL_N = 1024  # the pinned trainer holds out records [0, VAL_N) as its val slice; every pool keeps K12's head there


def replace_slice(base, new_rows, rng, tag):
    """replace len(new_rows) random base rows (never the val head) by new_rows (same total size)"""
    idx = rng.sample(range(VAL_N, len(base)), len(new_rows))
    out = list(base)
    for i, r in zip(idx, new_rows):
        out[i] = dict(r, slice=tag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_write", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    N = len(k12)
    pools = {"k12": k12}
    notes = {}
    minimal_path = os.path.join(PC.DATA, "k12_minimal.jsonl.gz")
    joins_path = os.path.join(PC.DATA, "joins.jsonl.gz")
    sub_path = os.path.join(PC.DATA, "subproblems.jsonl.gz")
    if os.path.exists(minimal_path):
        mn = [dict(r, pool="k12_minimal") for r in PC.read(minimal_path)]
        pools["k12_minimal"] = k12[:VAL_N] + mn[VAL_N:]  # K12's val head stays, so val_loss is comparable
    joins = PC.read(joins_path) if os.path.exists(joins_path) else []
    subs = PC.read(sub_path) if os.path.exists(sub_path) else []
    print("k12", N, "joins", len(joins), "subproblems", len(subs), flush=True)
    for frac in (0.01, 0.10):
        need = int(round(frac * N))
        tag = f"{int(frac * 100)}"
        if joins:
            rows, note = expand(joins, need, rng, "joins")
            pools[f"k12_joins{tag}"] = replace_slice(k12, rows, rng, "joins")
            notes[f"k12_joins{tag}"] = note
        if subs:
            rows, note = expand(subs, need, rng, "subproblems")
            pools[f"k12_sub{tag}"] = replace_slice(k12, rows, rng, "subproblems")
            notes[f"k12_sub{tag}"] = note
    if joins and subs:
        need = int(round(0.01 * N))
        structural = [r for r in joins if r.get("family") in ("four_prem", "depth3")]
        other = [r for r in joins if r.get("family") not in ("four_prem", "depth3")]
        j, n1 = expand(other, need // 2, rng, "joins")
        s, n2 = expand(subs, need // 4, rng, "subproblems")
        t, n3 = expand(structural, need - need // 2 - need // 4, rng, "structural")
        pools["k12_spice1"] = replace_slice(k12, j + s + t, rng, "spice")
        notes["k12_spice1"] = dict(joins=n1, subproblems=n2, structural=n3)
    if joins and "k12_minimal" in pools:
        need = int(round(0.10 * N))
        rows, note = expand(joins, need, rng, "joins")
        pools["k12_minimal_joins10"] = replace_slice(pools["k12_minimal"], rows, rng, "joins")
        notes["k12_minimal_joins10"] = note
    # stats
    pieces = PC.hard_pieces()
    st = []
    if os.path.exists(CAP6):
        cap6 = PC.read(CAP6)
        for r in cap6:
            r.setdefault("pool", "cap6")
        print("stats cap6", flush=True)
        st.append(PC.stats(cap6, "cap6 (take-home Stage-1 set; Robbie's pretrains)", pieces))
    for name, rows in pools.items():
        print("stats", name, flush=True)
        s = PC.stats(rows, name, pieces)
        s["note"] = notes.get(name, {})
        st.append(s)
        if not a.no_write and name != "k12":
            PC.write(rows, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
    hard_ref = {}
    if os.path.exists(PC.HARD):
        d = json.load(open(PC.HARD))
        bg = {(b["citing"], b["cited"]): b["hard_per_1k"] for b in d["bigrams"]}
        hard_ref = {k: round(sum(bg.get(p, 0) for p in v), 1) for k, v in PC.JOINS.items()}
    # slice-only stats for the new material
    extra = []
    if joins:
        extra.append(PC.stats(joins, "joins (all)", pieces))
    if subs:
        extra.append(PC.stats(subs, "subproblems (all)", pieces))
    table = PC.compare_table(st + extra, hard_ref)
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    kept = preserved_sections(DOC)
    with open(DOC, "w") as f:
        f.write("# Candidate Stage-1 pools, compared at the data level\n\n")
        f.write("Generated by `code/pretrain_data/mix.py`. Every pool has 155,000 records (K12's size), except the take-home's cap-6 Stage-1 set "
                "(154,990), which is the pool every one of Robbie's pretrains saw and is listed first for reference. Records 0–1,023 are K12's first 1,024 rows "
                "in every pool (the pinned trainer holds them out as its val slice, so val_loss is comparable across pools). `hard pieces covered` = how many of the "
                "hard textbook proofs' 2–6-line sub-theorem skeletons (from `hard_problems.md`, the top lists) occur as a whole-proof skeleton in the pool; "
                "`unknown pieces covered` = the same for the pieces K12 lacks. Shares are of records; join rates are per 1,000 citations.\n\n")
        f.write(table + "\n\n")
        for head in [h for h in kept if h.startswith("## Reading")]:
            f.write(kept.pop(head).rstrip("\n") + "\n\n")
        f.write("## Composition notes\n\n")
        for k, v in notes.items():
            f.write(f"- `{k}`: {json.dumps(v)}\n")
        f.write("\n## Length, premise and depth distributions\n\n")
        for s in st + extra:
            f.write(f"- `{s['label']}`: lines {s['by_len']}; premises {s['by_prem']}; depth {s['by_depth']}\n")
        for head, sec in kept.items():
            f.write("\n" + sec.rstrip("\n") + "\n")
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(dict(pools=st, extra=extra, hard_ref=hard_ref, notes=notes), f, indent=1, ensure_ascii=False)
    print(table)
    print("wrote", DOC)


if __name__ == "__main__":
    main()
