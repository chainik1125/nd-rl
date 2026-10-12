"""Rebuild the distracted copy of the sealed 72 textbook problems from the sealed files plus distractors.jsonl.

The sealed 72 are never committed (see the repo's CLAUDE.md), so this release ships only the two added premises and
their positions per problem. Restore the sealed files first, then apply:

    bash code/ws3/textbook72/fetch.sh
    python data/pretrain_data/release_2026-10-12_textbook72_distract2/apply_distractors.py

Writes docs/textbook-problem/textbook72/textbook72_distract2.jsonl (a gitignored folder; refuses to overwrite an
existing file unless --force) and checks the sha256 of the canonical rows against MANIFEST.json. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CANON_KEYS = ("name", "prompt", "base_prompt", "n_prem_orig", "n_prem", "n_distract", "role", "slice")


def split_prompt(prompt):
    """'THM A , B SEQ C PRF' -> (['A', 'B'], 'C'); premises are fully parenthesised or single tokens, so a top-level
    comma split is exact."""
    toks = prompt.split()
    assert toks[0] == "THM" and toks[-1] == "PRF", prompt
    i = toks.index("SEQ")
    body, goal = toks[1:i], " ".join(toks[i + 1:-1])
    prem, cur, depth = [], [], 0
    for t in body:
        if t == "," and depth == 0:
            prem.append(" ".join(cur))
            cur = []
            continue
        depth += (t == "(") - (t == ")")
        cur.append(t)
    if cur:
        prem.append(" ".join(cur))
    return prem, goal


def join_prompt(prem, goal):
    p = " , ".join(prem)
    return f"THM {p + ' ' if p else ''}SEQ {goal} PRF"


def apply(prompt, distractors, positions):
    prem, goal = split_prompt(prompt)
    n = len(prem) + len(distractors)
    out, it = [None] * n, iter(prem)
    for d, p in zip(distractors, positions):
        out[p] = d
    for i in range(n):
        if out[i] is None:
            out[i] = next(it)
    return join_prompt(out, goal)


def canon_line(r):
    return json.dumps({k: r[k] for k in CANON_KEYS}, ensure_ascii=False, sort_keys=True)


def canon_sha256(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update((canon_line(r) + "\n").encode())
    return h.hexdigest()


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t72", default=os.path.join(ROOT, "code", "ws3", "textbook72"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "textbook-problem", "textbook72", "textbook72_distract2.jsonl"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(HERE, "MANIFEST.json")))
    dist = {r["name"]: r for r in read_jsonl(os.path.join(HERE, "distractors.jsonl"))}
    rows = []
    for fn, slc in (("textbook_dev.jsonl", "dev58"), ("textbook_train.jsonl", "train14")):
        p = os.path.join(a.t72, fn)
        if not os.path.exists(p):
            sys.exit(f"{p} missing: run bash code/ws3/textbook72/fetch.sh first")
        for r in read_jsonl(p):
            d = dist.get(r["name"])
            if d is None:
                continue  # the one problem that took no distractors
            prompt2 = apply(r["prompt"], d["distractors"], d["positions"])
            rows.append(dict(name=r["name"], prompt=prompt2, base_prompt=r["prompt"], n_prem_orig=len(split_prompt(r["prompt"])[0]),
                             n_prem=len(split_prompt(prompt2)[0]), n_distract=len(d["distractors"]), role=r.get("role"), slice=slc))
    assert len(rows) == len(dist), (len(rows), len(dist))
    got, want = canon_sha256(rows), manifest["distracted_set"]["canonical_sha256"]
    if got != want:
        sys.exit(f"canonical sha256 mismatch: got {got}, manifest {want}")
    print(f"ok: {len(rows)} distracted problems rebuilt, canonical sha256 {got}")
    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} exists; not overwritten (use --force)")
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
