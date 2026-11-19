# ruff: noqa: E501
"""Seconds per iteration of each guard-off line with vs without another job on its GPU (serrano, read-only).

Per-iteration `secs` from metrics.jsonl (runs_g0/<line>/seg*/); the co-resident job count from the monitor's polls
(`neighbors`: [time, n tag_grpo processes not ours, last iteration seen]); an iteration takes the count of the first
poll that had seen it. Iteration 1 of each segment (warm-up) is left out.
"""

import json
import statistics
from pathlib import Path

G0 = Path("/scratch/cowifornia/nd-rl/author_tags/runs_g0")
st = json.loads((G0 / "monitor_state.json").read_text())
for ln, s in st["tag"].items():
    nb = [x for x in s.get("neighbors", []) if x[2] is not None]
    rows = []
    for seg in sorted((G0 / ln).glob("seg[0-9]*")):
        rs = [json.loads(x) for x in (seg / "metrics.jsonl").read_text().splitlines()]
        rows += [(seg.name, r["iter"], r["secs"]) for r in rs[1:]]
    by = {"alone": [], "shared": []}
    for _segn, it, secs in rows:
        n = next((x[1] for x in nb if x[2] >= it), None)
        if n is not None:
            by["shared" if n > 0 else "alone"].append(secs)
    out = {k: (len(v), round(statistics.median(v), 1) if v else None) for k, v in by.items()}
    print(f"{ln} ({s['host']}): n iterations, median s/it -> {out}")
