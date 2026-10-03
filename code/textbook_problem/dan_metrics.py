"""Post-process Dan's `state_eval.py` rows (eval_dan_best_state) into the plan's metrics, re-judging
every Lean-accepted proof with the pinned nd_verify through strict_verify.

    cd code && uv run python textbook_problem/dan_metrics.py --run ../experiments/textbook-problem/eval_dan_best_state/run_best12_r8

Reads run/s<seed>/<set>.jsonl (and the textbook72 rows from the gitignored docs folder), writes
run/metrics.jsonl (one line per seed × set) and prints the tables for results.md. Avg solve rate is
the mean over problems of n_ok / n_tried as Lean counted it; `nd_disagree` counts distinct accepted
proofs the pinned verifier rejects (and the problems they belong to), which is the caveat on that
number. The textbook 72 are reported in aggregate only.
"""

import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import prove  # noqa: E402
from derive import structural_ok  # noqa: E402

T72 = os.path.join(HERE, "../../docs/textbook-problem/textbook72/dan_eval")
SETS = ["v0", "v1", "hand14", "batch3", "lean_v2", "derived_plain", "textbook72"]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def summarize(rows, meta, k):
    rates, disagree_proofs, disagree_rows, struct_fail = [], 0, 0, 0
    by = collections.defaultdict(list)
    for r in rows:
        m = meta.get(r["name"], {})
        bad = [p for p in r["proofs"] if not prove.safe_verify(r["prompt"], p)[0]]
        disagree_proofs += len(bad)
        disagree_rows += bool(bad)
        rate = r["n_ok"] / max(1, r["n_tried"])
        if m.get("check") and r["proofs"]:
            okp = [p for p in r["proofs"] if structural_ok(p, m["check"])]
            if not okp:
                struct_fail += 1
                rate = 0.0
        rates.append(rate)
        if m.get("nuance"):
            by[m["nuance"]].append(rate)
    n = len(rates)
    mean = sum(rates) / n if n else 0.0
    ci = 1.96 * statistics.pstdev(rates) / math.sqrt(n) if n > 1 else 0.0
    solved = sum(1 for x in rates if x > 0)
    out = dict(n=n, k=k, avg_solve_rate=round(mean, 4), ci95=round(ci, 4), any_of_k=solved,
               any_of_k_rate=round(solved / n, 4) if n else 0.0, never_solved=n - solved,
               nd_disagree_proofs=disagree_proofs, nd_disagree_rows=disagree_rows, struct_fail_rows=struct_fail)
    if by:
        out["by_nuance"] = {nu: dict(n=len(v), avg_solve_rate=round(sum(v) / len(v), 4), any_of_k=sum(1 for x in v if x > 0))
                            for nu, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    inputs = os.path.join(a.run, "..", "inputs")
    meta = {}
    for s in SETS:
        p = os.path.join(inputs, f"{s}.jsonl")
        for r in jl(p) or []:
            meta[r["name"]] = r
    out = []
    for seed in range(3):
        for s in SETS:
            p = os.path.join(T72 if s == "textbook72" else a.run, f"s{seed}", f"{s}.jsonl")
            rows = jl(p)
            if rows is None:
                continue
            summ = summarize(rows, meta, a.k)
            summ.update(seed=seed, set=s)
            out.append(summ)
    with open(os.path.join(a.run, "metrics.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("| set | seed | n | avg solve rate | ±95% | any-of-32 | never | nd≠Lean proofs (rows) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['set']} | s{r['seed']} | {r['n']} | {r['avg_solve_rate']:.3f} | {r['ci95']:.3f} | {r['any_of_k']} ({r['any_of_k_rate']:.0%}) | {r['never_solved']} | {r['nd_disagree_proofs']} ({r['nd_disagree_rows']}) |")
    # seed summary per set
    print("\n| set | seeds | avg solve rate mean (min–max) | any-of-32 mean (min–max) |\n|---|---|---|---|")
    for s in SETS:
        rs = [r for r in out if r["set"] == s]
        if not rs:
            continue
        a_ = [r["avg_solve_rate"] for r in rs]
        b_ = [r["any_of_k"] for r in rs]
        print(f"| {s} | {len(rs)} | {sum(a_)/len(a_):.3f} ({min(a_):.3f}–{max(a_):.3f}) | {sum(b_)/len(b_):.1f} ({min(b_)}–{max(b_)}) of {rs[0]['n']} |")
    nus = [r for r in out if r["set"] == "derived_plain"]
    if nus:
        names = sorted(nus[0]["by_nuance"])
        print("\n| nuance | n | " + " | ".join(f"s{r['seed']} avg / any" for r in nus) + " |\n|---|---|" + "---|" * len(nus))
        for nu in names:
            print(f"| {nu} | {nus[0]['by_nuance'][nu]['n']} | " + " | ".join(f"{r['by_nuance'][nu]['avg_solve_rate']:.3f} / {r['by_nuance'][nu]['any_of_k']}" for r in nus) + " |")


if __name__ == "__main__":
    main()
