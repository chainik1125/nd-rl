#!/usr/bin/env bash
# Run 5 (p1), second top-up: waits for c8_pool.sh, then a larger generation batch (same settings, new seed).
cd /workspace/nd-takehome
export OMP_NUM_THREADS=1
until grep -q NEC_C8C_DONE artifacts/r5_c8_pool.log; do sleep 30; done
nice -n 5 python3 make_coverage_sets.py gen --long --min 12 --max 16 --out data/p2/raw_long_c8d --workers 6 --tries 9000000 --cap_np 500000 --cap_pat 500000 --only derived_ore_strict --seed 6000 > artifacts/r5/gen_c8d.log 2>&1
python3 make_coverage_sets.py merge --glob "data/p2/raw_long_c8d.w*.jsonl" --out data/p2/pool_long_c8d.jsonl --prefix c8longd >> artifacts/r5/gen_c8d.log 2>&1
echo GEN_C8D_DONE
nice -n 5 python3 minlen.py --in data/p2/pool_long_c8d.jsonl --out data/p2/pool_long_c8d_minlen.jsonl --bound 8 --time 10 --procs 6 > artifacts/r5/minlen_c8d.log 2>&1
python3 - <<'PY'
import json
names = {json.loads(l)['name'] for l in open('data/p2/pool_long_c8d_minlen.jsonl') if json.loads(l)['min_lines_ub'] is None}
n = 0
with open('data/p2/run5_c8d_cands.jsonl', 'w') as fo:
    for l in open('data/p2/pool_long_c8d.jsonl'):
        r = json.loads(l)
        if r['name'] in names:
            fo.write(json.dumps({k: r[k] for k in ('name', 'thm', 'key', 'prompt', 'n_lines', 'n_prem', 'pat', 'proof') if k in r}) + '\n'); n += 1
print('c8d None-at-8 candidates', n)
PY
echo MINLEN_C8D_DONE
nice -n 5 python3 necessity.py --in data/p2/run5_c8d_cands.jsonl --out data/p2/run5_c8d_nec.jsonl --pattern derived_ore_strict --bound 10 --time 60 --procs 6 > artifacts/r5/nec_c8d.log 2>&1
echo NEC_C8D_DONE
