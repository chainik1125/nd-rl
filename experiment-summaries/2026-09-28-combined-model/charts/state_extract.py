"""Pull the proof-state experiment's numbers out of its summary.json files into state_runs.csv.

One row per finished run: recipe, arm, seed, dev metric, RL targets solved, wall-clock of the
whole run and of its RL stage. Crashed and lost runs are listed with status and no numbers.
Summaries live in the gitignored artifacts/autoresearch/st-<recipe>-<arm>_s<seed>/ of the
checkout that ran them (nd-rl-8). Run from the summary dir:  python3 charts/state_extract.py
"""

import csv
import json
import os

ART = os.path.expanduser("~/repos/nd-rl-8/artifacts/autoresearch")

rows = []
for recipe in ("198", "207", "201", "204"):
    for arm in ("C0", "C0k192", "SN", "SH"):
        for seed in range(3):
            fn = f"{ART}/st-{recipe}-{arm}_s{seed}/summary.json"
            s = json.load(open(fn)) if os.path.exists(fn) else {"status": "missing"}
            ok = s["status"] == "ok"
            rows.append(
                {
                    "recipe": recipe,
                    "arm": arm,
                    "seed": seed,
                    "status": s["status"],
                    "metric": s["metric"] if ok else "",
                    "rl_solved": s["metric_rl"] if ok else "",
                    "total_secs": round(s["total_secs"]) if ok else "",
                    "rl_secs": round(s["rl"]["secs"]) if ok else "",
                    "gpu": s.get("gpu", ""),
                }
            )
with open("charts/state_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote charts/state_runs.csv: {sum(r['status'] == 'ok' for r in rows)} ok of {len(rows)}")
