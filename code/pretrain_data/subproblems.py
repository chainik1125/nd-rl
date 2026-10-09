"""Prototype for the pretrain-data workstream: how many easier problems do the textbook proofs contain?

Every line of a verified Fitch proof is itself a theorem: the premises plus the assumptions of the boxes
open at that line entail the line's formula, and the lines it transitively cites are its proof (with the
enclosing assumptions promoted to premises). This script extracts those sub-problems from the project's
verified reference proofs, verifies each with the pinned verifier, deduplicates by curry-aware renaming
class, and reports the yield, its length and premise distributions, and its overlap with every
evaluation set (the sealed 72 counted only).

    cd code && uv run python pretrain_data/subproblems.py [--out ../data/pretrain_data/subproblems_proto.jsonl]

Analysis only: nothing here is a training set. The sealed 72 are never used as seeds.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../textbook_problem"), os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import eval_analysis as EA  # noqa: E402
import prove  # noqa: E402
import transcribe as T  # noqa: E402
from data import load_textbook72  # noqa: E402
from patterns import lines_to_text, parse  # noqa: E402
from release_1002 import BR  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

L = prove.L


def open_assumptions(lines, i):
    """AS lines of the boxes enclosing line i (one per depth level 1..depth_i), in order."""
    d_i = lines[i]["depth"]
    out = {}
    for j in range(i, -1, -1):
        ln = lines[j]
        if ln["rule"] == "AS" and 1 <= ln["depth"] <= d_i and ln["depth"] not in out:
            # the most recent AS at that depth that has not been closed before i
            closed = any(lines[k]["depth"] < ln["depth"] for k in range(j + 1, i + 1))
            if not closed:
                out[ln["depth"]] = j
        if len(out) == d_i:
            break
    return [out[d] for d in sorted(out)]


def subproblem(lines, i, prune_prem=False):
    """-> (prompt, proof_text) for the theorem proved at line i, or None if trivial.

    prune_prem: keep only the premises and enclosing assumptions the sub-proof actually cites
    (the smallest theorem the line proves), instead of carrying every premise of the seed."""
    goal_ln = lines[i]
    if goal_ln["rule"] in ("PR", "AS"):
        return None
    d_i = goal_ln["depth"]
    enclosing_all = open_assumptions(lines, i)   # every box containing line i vanishes in the sub-problem
    enclosing = enclosing_all
    prem_lines = [ln for ln in lines if ln["rule"] == "PR"]
    goal = goal_ln["formula"]
    # dependency closure from line i
    byidx = {ln["idx"]: ln for ln in lines}
    closure = set()
    stack = [goal_ln["idx"]]
    while stack:
        x = stack.pop()
        if x in closure or x not in byidx:
            continue
        closure.add(x)
        stack.extend(byidx[x]["refs"])
    if prune_prem:
        prem_lines = [ln for ln in prem_lines if ln["idx"] in closure]
        enclosing = [j for j in enclosing if lines[j]["idx"] in closure]
    prem = [ln["formula"] for ln in prem_lines] + [lines[j]["formula"] for j in enclosing]
    if goal in prem:
        return None
    keep = closure | set(ln["idx"] for ln in prem_lines) | set(lines[j]["idx"] for j in enclosing)
    kept = [ln for ln in lines if ln["idx"] in keep]
    pos = {ln["idx"]: p for p, ln in enumerate(lines)}

    def inside(j, k):
        """is line position k inside the box whose assumption is at position j?"""
        d = lines[j]["depth"]
        if k < j or lines[k]["depth"] < d:
            return False
        return all(lines[m]["depth"] >= d for m in range(j + 1, k + 1))

    new = []
    enc_idx = {lines[j]["idx"] for j in enclosing}
    for ln in kept:
        ln2 = dict(ln)
        if ln["rule"] == "AS" and ln["idx"] in enc_idx:
            ln2["rule"], ln2["refs"], ln2["depth"] = "PR", [], 0
        else:
            k = pos[ln["idx"]]
            shift = sum(1 for j in enclosing_all if inside(j, k))
            ln2["depth"] = ln["depth"] - shift
        new.append(ln2)
    # premises first (verifier requires PR lines at the top), then the rest in order
    prs = [ln for ln in new if ln["rule"] == "PR"]
    rest = [ln for ln in new if ln["rule"] != "PR"]
    if not rest:
        return None
    text = lines_to_text(prs + rest)
    prompt = T.make_prompt(prem, goal)
    return prompt, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_lines", type=int, default=2)
    ap.add_argument("--max_lines", type=int, default=0, help="drop sub-problems longer than this (0 = keep all)")
    ap.add_argument("--prune_prem", action="store_true", help="keep only the premises each sub-proof cites")
    ap.add_argument("--seed_sets", default="v0,lean_v2",
                    help="comma-separated evaluation sets whose problems may seed the pool (questions.md 1 default)")
    ap.add_argument("--pool", action="store_true", help="write K12-format records, audited against every evaluation set")
    a = ap.parse_args()
    refs = EA.load_refs()
    # evaluation index: every scraped/derived problem's curry-aware keys, plus the sealed 72 (counted only)
    eval_prompts = {name: r["prompt"] for name, r in refs.items()}
    eval_index = {}
    for name, p in eval_prompts.items():
        for k in BR.keys(p):
            eval_index.setdefault(k, set()).add(name)
    t72 = load_textbook72()
    idx72 = BR.index([r["prompt"] for r in t72])
    sets = {}
    csv_path = os.path.join(HERE, "../../experiments/textbook-problem/eval_dan_best_state/analysis/per_problem.csv")
    if os.path.exists(csv_path):
        import csv
        for row in csv.DictReader(open(csv_path)):
            sets[row["name"]] = row["set"]
    allowed = set(a.seed_sets.split(",")) if a.seed_sets else None

    seeds = {n: r for n, r in refs.items() if r.get("source", {}).get("id") != "derived" and r.get("reference_proof")
             and (allowed is None or sets.get(n) in allowed)}
    rows, seen = [], {}
    stats = collections.Counter()
    per_seed = collections.Counter()
    for name, r in seeds.items():
        lines = parse(r["reference_proof"])
        if lines is None:
            stats["seed_unparsed"] += 1
            continue
        for i in range(len(lines) - 1):  # the last line is the seed problem itself
            sp = subproblem(lines, i, prune_prem=a.prune_prem)
            if sp is None:
                continue
            prompt, text = sp
            ok, reason, n = safe_verify(prompt, text)
            if not ok:
                stats["verify_fail"] += 1
                stats[f"fail:{reason[:30]}"] += 1
                continue
            if n < a.min_lines or (a.max_lines and n > a.max_lines):
                stats["too_short_or_long"] += 1
                continue
            keys = BR.keys(prompt)
            if any(k in seen for k in keys):
                stats["dup_class"] += 1
                continue
            hit_eval = set()
            for k in keys:
                hit_eval |= eval_index.get(k, set())
            hit_eval.discard(name)
            hit72 = any(k in idx72 for k in keys)
            for k in keys:
                seen[k] = name
            rows.append({"seed": name, "seed_set": sets.get(name, r.get("source", {}).get("id")), "line": i + 1, "prompt": prompt, "proof": text,
                         "n_lines": n, "n_prem": len(L.parse_prompt(prompt)[0]), "eval_twin": sorted(hit_eval), "sealed_twin": hit72})
            per_seed[name] += 1
            stats["kept"] += 1
    by_len = collections.Counter(r["n_lines"] for r in rows)
    by_prem = collections.Counter(r["n_prem"] for r in rows)
    by_set = collections.Counter(r["seed_set"] for r in rows)
    n_eval_twin = sum(1 for r in rows if r["eval_twin"])
    n_sealed = sum(1 for r in rows if r["sealed_twin"])
    seeds_with = sum(1 for n in seeds if per_seed[n])
    summary = {
        "seed_sets": a.seed_sets, "prune_prem": a.prune_prem,
        "seeds": len(seeds), "seeds_yielding": seeds_with, "subproblems_distinct_classes": len(rows),
        "mean_per_seed": round(len(rows) / max(1, len(seeds)), 2),
        "by_lines": dict(sorted(by_len.items())), "by_premises": dict(sorted(by_prem.items())), "by_seed_set": dict(by_set),
        "twins_of_other_eval_problems": n_eval_twin, "twins_of_sealed_72": n_sealed, "stats": dict(stats),
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        if a.pool:
            import pool_common as PC
            recs = []
            for r in rows:
                rec, why = PC.make_record(r["prompt"], r["proof"], "subproblems", seed=r["seed"], seed_set=r["seed_set"], seed_line=r["line"])
                if rec:
                    recs.append(rec)
            recs, audit = PC.EvalIndex().audit(recs, allowed_sets=allowed or ())
            summary["audit"] = audit
            PC.write(recs, a.out)
            with open(a.out.replace(".jsonl.gz", ".stats.json"), "w") as f:
                json.dump(summary, f, indent=1)
            print("audit:", audit)
            print("wrote", a.out, len(recs))
        else:
            with open(a.out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("wrote", a.out)


if __name__ == "__main__":
    main()
