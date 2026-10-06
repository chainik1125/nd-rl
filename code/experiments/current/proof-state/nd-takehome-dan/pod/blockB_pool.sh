#!/usr/bin/env bash
# Block B pool: merge the reductio_nodn long shards, intuit-label, minlen-label the classical-only ones.
cd /workspace/nd-takehome
set -e
python3 make_coverage_sets.py merge --glob "data/p2/raw_long_rnd.w*.jsonl" --out data/p2/pool_rnd.jsonl --prefix rnd
python3 intuit.py --in data/p2/pool_rnd.jsonl --out data/p2/pool_rnd_intuit.jsonl
python3 - <<'PY'
import json
it = {json.loads(l)['name']: json.loads(l) for l in open('data/p2/pool_rnd_intuit.jsonl')}
n = 0; co = 0
with open('data/p2/pool_rnd_co.jsonl', 'w') as f:
    for l in open('data/p2/pool_rnd.jsonl'):
        r = json.loads(l); n += 1
        if it[r['name']]['classical_only']:
            co += 1; f.write(l)
print('pool', n, 'classical-only', co)
PY
python3 minlen.py --in data/p2/pool_rnd_co.jsonl --out data/p2/pool_rnd_co_minlen.jsonl --bound 8 --time 20 --procs 5
echo BLOCKB_POOL_DONE
