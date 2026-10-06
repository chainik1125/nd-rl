#!/usr/bin/env bash
# Block C targets (p1): merge the strict-derived-ORE 9-14-line shards, minlen (bound 8) a 3,000-theorem sample, keep None.
cd /workspace/nd-takehome
set -e
python3 make_coverage_sets.py merge --glob "data/p2/raw_long_c8.w*.jsonl" --out data/p2/pool_long_c8.jsonl --prefix c8long
python3 - <<'PY'
import json, random
rs = [json.loads(l) for l in open('data/p2/pool_long_c8.jsonl')]
random.Random(0).shuffle(rs)
with open('data/p2/pool_long_c8_sample.jsonl', 'w') as f:
    for r in rs[:3000]: f.write(json.dumps(r) + '\n')
print('sample 3000 of', len(rs))
PY
nice -n 10 python3 minlen.py --in data/p2/pool_long_c8_sample.jsonl --out data/p2/pool_long_c8_minlen.jsonl --bound 8 --time 10 --procs 4
echo BLOCKC_TARGETS_MINLEN_DONE
