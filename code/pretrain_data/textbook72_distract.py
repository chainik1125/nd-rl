"""Build the distracted copy of the sealed 72 (two unused premises per problem) for experiments/pretrain-data/
textbook72_distract2. Output goes ONLY to the gitignored docs/textbook-problem/textbook72/ folder; nothing per
problem is printed or written elsewhere. See the proposal for the construction.

    cd code && uv run python pretrain_data/textbook72_distract.py
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem"), os.path.join(HERE, "..", "textbook_problem", "fork_tools")]
import pool_common as PC  # noqa: E402  (pins the verifier)
import scan_pools as SP  # noqa: E402  (rand_formula, prompt_text, LG)
from patterns import fstr  # noqa: E402
from minlen import minlen  # noqa: E402
from data import load_textbook72  # noqa: E402

OUT_DIR = os.path.join(PC.ROOT, "docs", "textbook-problem", "textbook72")
OUT = os.path.join(OUT_DIR, "textbook72_distract2.jsonl")


def subformulas(f, acc):
    acc.add(fstr(f))
    for x in f[1:]:
        if isinstance(x, tuple):
            subformulas(x, acc)
    return acc


def main():
    assert os.path.isdir(OUT_DIR), OUT_DIR
    rng = random.Random(0)
    rows = load_textbook72()
    out, status = [], collections.Counter()
    for r in rows:
        prem, goal = SP.LG.parse_prompt(r["prompt"])
        forbidden = set()
        for f in prem + [goal]:
            subformulas(f, forbidden)
        ref = int(r.get("reference_lines") or 0)
        # the reference is an upper bound, not the minimum: find the minimal length first so a distractor is only
        # called a shortcut when the distracted problem has a proof shorter than minimal + 2 premise lines
        n_min, min_status = ref, "ref"
        if ref:
            try:
                u0 = minlen(r["prompt"], bound=ref, time_limit=10.0)
                if u0.get("min_lines_ub") is not None:
                    n_min, min_status = int(u0["min_lines_ub"]), "found"
                elif u0.get("timeout"):
                    min_status = "timeout"
            except Exception as e:  # noqa: BLE001
                min_status = f"error:{type(e).__name__}"
        consistent = SP.LG.satisfiable(prem)  # inconsistent premises: no satisfiability test can be asked of a distractor
        placed = None
        for draw in range(6):
            ds, tries = [], 0
            while len(ds) < 2 and tries < 200:
                tries += 1
                d = SP.rand_formula(rng, 2)
                s = fstr(d)
                if s in forbidden or any(fstr(x) == s for x in ds) or len(s.split()) > 9:
                    continue
                if consistent and not SP.LG.satisfiable(prem + ds + [d]):
                    continue
                ds.append(d)
            if len(ds) < 2:
                continue
            order = list(prem)
            for d in ds:
                order.insert(rng.randint(0, len(order)), d)
            prompt2 = SP.prompt_text(order, goal)
            st = "unknown"
            if n_min:
                try:
                    u = minlen(prompt2, bound=n_min + 1, time_limit=10.0)
                    if u.get("timeout"):
                        st = "unknown"
                    elif u.get("min_lines_ub") is not None:
                        st = "shortcut"
                    else:
                        st = "no_shortcut"
                except Exception as e:  # noqa: BLE001
                    st = f"error:{type(e).__name__}"
            if st == "shortcut":
                continue
            placed = dict(name=r["name"], prompt=prompt2, base_prompt=r["prompt"], n_prem_orig=len(prem), n_prem=len(order),
                          reference_lines=ref, min_lines=n_min, min_status=min_status, n_distract=2, shortcut_check=st,
                          role=r.get("role"), slice=r.get("slice"))
            break
        if placed is None:
            status["failed"] += 1
            continue
        status[placed["shortcut_check"]] += 1
        out.append(placed)
    with open(OUT, "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("wrote", len(out), "of", len(rows), "to the gitignored textbook72 folder; shortcut status", dict(status),
          "; minimal-length status", dict(collections.Counter(p["min_status"] for p in out)))
    print("premises after distraction:", dict(sorted(collections.Counter(p["n_prem"] for p in out).items())))


if __name__ == "__main__":
    main()
