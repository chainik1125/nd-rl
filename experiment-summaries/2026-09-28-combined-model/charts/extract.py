"""Pull the factorial runs' numbers out of their summary.json files into committed CSVs.

  runs.csv   one row per run: cell, seed, dev metric, pipeline seconds, RL samples, GPU
  passk.csv  one row per run x pass@k pool x problem: n_lines, n_ok of n_samples

Summaries live in the gitignored artifacts/autoresearch/fact-<tok>-<pre>-<rl>_s<seed>/ of the
checkout that ran them (nd-rl-7), so the charts read only these CSVs.
Run from the summary dir:  python3 charts/extract.py
"""

import csv
import glob
import json
import os

ART = os.path.expanduser("~/repos/nd-rl-7/artifacts/autoresearch")

runs, passk = [], []
for fn in sorted(glob.glob(f"{ART}/fact-*_s*/summary.json")):
    run = fn.split("/")[-2]
    cell, seed = run.rsplit("_s", 1)
    _, tok, pre, rl = cell.split("-")
    s = json.load(open(fn))
    if s["status"] != "ok":
        print(f"skip {run}: {s['status']} {s.get('reason', '')[:80]}")
        continue
    rounds = s["rl"]["rounds"]
    runs.append(
        {
            "run": run,
            "tok": tok,
            "pre": pre,
            "rl": rl,
            "seed": seed,
            "metric": s["metric"],
            "pipeline_secs": round(s["eval"]["pipeline_secs"]),
            "rl_samples": sum(r["sampling"]["samples"] for r in rounds),
            "rl_solved": s["metric_rl"],
            "gpu": s["gpu"],
        }
    )
    for pool, d in s["eval"]["passk"].items():
        for name, p in d["problems"].items():
            passk.append(
                {
                    "run": run,
                    "pool": pool,
                    "problem": name,
                    "split": p["split"] or "",
                    "n_lines": p["n_lines"] if p["n_lines"] is not None else "",
                    "n_ok": p["n_ok"],
                    "n": d["n_samples"],
                }
            )

for name, rows in (("runs", runs), ("passk", passk)):
    with open(f"charts/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote charts/{name}.csv ({len(rows)} rows)")
