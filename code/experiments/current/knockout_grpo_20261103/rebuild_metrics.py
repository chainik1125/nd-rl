# ruff: noqa: SIM115, E501  -- a one-off repair script, kept as it ran (log.md, 2026-11-04)
"""Rebuild metrics.jsonl rows that a stale rsync overwrote (2026-11-04, serrano): for each lost
iteration, sampled tokens and phase seconds from the run's registry (exact), the skip share,
pass@1 and ban mass from the stage log's `it N:` line, and trained tokens estimated as
ratio x sampled x (1 - skip) with the ratio fitted on the run's intact rows. Rows are marked
`reconstructed`. The file is rewritten atomically right after the run's next append.
    python3 rebuild_metrics.py TAG [--wait]"""

import collections
import glob
import json
import os
import re
import shutil
import statistics as st
import sys
import time

K = "/scratch/cowifornia/nd-rl/knockout_grpo"
tag, wait = sys.argv[1], "--wait" in sys.argv
mp = f"{K}/runs/{tag}_L3/metrics.jsonl"
log = f"{K}/logs/{tag}_L3.log"
IT = re.compile(
    r"^it (\d+): pass@1 ([\d.]+)% .* skip (\d+)% .* ban ([\d.e+-]+)/([\d.e+-]+) .* (\d+)s \("
)


def rows():
    return [json.loads(x) for x in open(mp).read().splitlines() if x.strip()]


reg = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{K}/runs/{tag}_L3/registry/*.jsonl")):
    for x in open(f):
        r = json.loads(x)
        lb = r.get("labels") or {}
        if "round" in lb:
            reg[lb["round"]][(r["metric"], lb.get("phase"))] = r["value"]
info = {}
for x in open(log):
    m = IT.match(x)
    if m:
        info[int(m.group(1))] = dict(
            pass1=float(m.group(2)) / 100,
            skip=int(m.group(3)) / 100,
            ban=float(m.group(4)),
            banmax=float(m.group(5)),
            secs=int(m.group(6)),
        )
if wait:  # rewrite right after the run's next append, so no row of its is lost
    n0 = len(open(mp).read().splitlines())
    for _ in range(900):
        time.sleep(1)
        if len(open(mp).read().splitlines()) > n0:
            break
cur = rows()
have = {r["iter"] for r in cur}
known = [r for r in cur if r["iter"] in reg and r["iter"] in info and not r.get("reconstructed")]
ratio = st.mean(
    r["trained_tokens"] / (reg[r["iter"]][("gen_tokens", "sample")] * (1 - info[r["iter"]]["skip"]))
    for r in known
)
lost = sorted(i for i in reg if i not in have and i in info and i <= max(have))
new = []
for i in lost:
    s = reg[i][("gen_tokens", "sample")]
    att = reg[i].get(("attempts", "sample"), 4096)
    new.append(
        {
            "iter": i,
            "tokens_per_attempt": s / att,
            "trained_tokens": ratio * s * (1 - info[i]["skip"]),
            "secs": info[i]["secs"],
            "reward_pass1": info[i]["pass1"],
            "ban_think": info[i]["ban"],
            "ban_think_max": info[i]["banmax"],
            "reconstructed": True,
            "trained_ratio": ratio,
        }
    )
out = sorted(cur + new, key=lambda r: r["iter"])
bak = mp + f".before_rebuild_{int(time.time())}"
shutil.copy2(mp, bak)
tmp = mp + ".tmp"
with open(tmp, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
os.replace(tmp, mp)
print(
    f"{tag}: {len(new)} rows rebuilt ({lost[0] if lost else None}-{lost[-1] if lost else None}), ratio {ratio:.3f}, {len(out)} rows now; backup {bak}"
)
