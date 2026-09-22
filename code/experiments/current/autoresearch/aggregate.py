#!/usr/bin/env python3
"""Per-config mean/sd and pooled within-config sd for a sweep.  python3 aggregate.py s20"""

import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "autoresearch")
sweep = sys.argv[1]
by = collections.defaultdict(list)
for d in sorted(glob.glob(f"{ART}/{sweep}_*_s[0-9]")):
    cfg = os.path.basename(d)[len(sweep) + 1 : -3]
    try:
        with open(f"{d}/summary.json") as f:
            s = json.load(f)
    except FileNotFoundError:
        continue
    by[cfg].append((s["status"], s["metric"], s["metric_rl"], round(s["total_secs"])))

print("| config | n | dev>=7 per seed | mean | sd | RL solved per seed | mean | sd | total s |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
ss_dev, ss_rl, df = 0.0, 0.0, 0
for cfg, rows in by.items():
    ok = [r for r in rows if r[0] == "ok"]
    m = [r[1] for r in ok]
    rl = [r[2] for r in ok]

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else float("nan")

    if len(ok) > 1:
        ss_dev += sum((x - statistics.mean(m)) ** 2 for x in m)
        ss_rl += sum((x - statistics.mean(rl)) ** 2 for x in rl)
        df += len(ok) - 1
    fails = [r[0] for r in rows if r[0] != "ok"]
    print(
        f"| {cfg} | {len(ok)}{' +' + ','.join(fails) if fails else ''} | {m} "
        f"| {statistics.mean(m):.0f} | {sd(m):.0f} | {rl} | {statistics.mean(rl):.0f} "
        f"| {sd(rl):.0f} | {[r[3] for r in ok]} |"
    )
if df:
    print(
        f"\npooled within-config sd: dev>=7 {(ss_dev / df) ** 0.5:.1f}, "
        f"RL solved {(ss_rl / df) ** 0.5:.1f}  (df={df})"
    )
