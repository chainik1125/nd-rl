"""How much would by-length change if each theorem's length were a global min, not per run?

by_length.py bins a solved theorem by min(this run's shortest pruned proof, reference n_lines).
The global version fixes each theorem's length once, as min(reference, the shortest pruned proof
any seed of any run in runs.csv ever found), and bins every run by that. Prints both for a few
runs, 3-seed means, and the % change per length. Numbers for the README footnote; no chart.

Run from the summary dir:  python3 charts/global_min.py
"""
import csv
import json
import os
import statistics

SEED_DIRS = [os.path.expanduser(f"~/repos/{c}/artifacts/autoresearch")
             for c in ("nd-rl", "nd-rl-3", "nd-rl-4", "nd-rl-5")]
LENGTHS = range(7, 13)
SHOW = ["012_baseline-3seeds", "081-claude_polar-express", "100-claude_lean-baseline-2400",
        "153-claude_lean-width384"]


def seed_proofs(run, seed):
    path = next(f"{d}/{run}_s{seed}/eval_dev.jsonl" for d in SEED_DIRS
                if os.path.exists(f"{d}/{run}_s{seed}/eval_dev.jsonl"))
    return [json.loads(line) for line in open(path)]


runs = {r["run"]: r for r in csv.DictReader(open("charts/runs.csv"))}
evals = {run: [seed_proofs(run, s) for s in "012"] for run in runs}

best = {}  # theorem -> global min length
for seeds in evals.values():
    for e in seeds:
        for t in e:
            cand = min(t["pruned_lens"]) if t["solved"] else t["n_lines"]
            best[t["name"]] = min(best.get(t["name"], t["n_lines"]), cand, t["n_lines"])


def counts(run, length_of):
    return {n: statistics.mean(sum(length_of(t) == n for t in e if t["solved"])
                               for e in evals[run]) for n in LENGTHS}


print("theorems whose global min beats the reference:",
      sum(best[k] < t["n_lines"] for k, t in ((t["name"], t) for t in evals[SHOW[-1]][0])))
print(f"{'run':32} {'length':>6} {'per-run':>8} {'global':>8} {'change':>8}")
for run in SHOW:
    per_run = counts(run, lambda t: min(*t["pruned_lens"], t["n_lines"]))
    glob = counts(run, lambda t: best[t["name"]])
    for n in LENGTHS:
        a, b = per_run[n], glob[n]
        change = f"{100 * (b - a) / a:+.0f}%" if a else "n/a"
        print(f"{run:32} {n:>6} {a:>8.1f} {b:>8.1f} {change:>8}")
