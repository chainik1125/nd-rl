"""Print N seeded-random raw samples from a JSONL dump (for results.md). Reproducible:

    uv run python ../experiments/ws3/exp-00-local-check/pick_samples.py <dump.jsonl> [n] [seed]
"""
import json
import random
import sys

path, n, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20, int(sys.argv[3]) if len(sys.argv) > 3 else 0
import gzip
opener = gzip.open if path.endswith('.gz') else open
rows = [json.loads(l) for l in opener(path, 'rt') if l.strip()]
rng = random.Random(seed)
idx = sorted(rng.sample(range(len(rows)), n))
print(f"{n} of {len(rows)} rows from {path}, seed {seed}, row indices {idx}\n")
for i in idx:
    r = rows[i]
    ok = r.get("ok", r.get("reward"))
    head = f"row {i}"
    if "step" in r:
        head += f", step {r['step']}"
    if "length" in r:
        head += f", length {r['length']}"
    if r.get("slice"):
        head += f", slice {r['slice']}"
    print(f"### {head}: {'ACCEPTED' if ok in (True, 1.0) else 'rejected'} ({r.get('reason')})")
    print(f"prompt: `{r['prompt']}`  ")
    print(f"completion: `{r['completion']}`\n")
