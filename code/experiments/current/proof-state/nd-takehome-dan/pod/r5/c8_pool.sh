#!/usr/bin/env bash
# Run 5 (p1): necessity oracle on the existing cap-8 candidates, then a generated top-up (unchanged generator, 12-16 lines,
# strict-derived-ORE output filter), minlen bound 8 -> keep None, oracle on those too.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=1
mkdir -p artifacts/r5
nice -n 5 python3 necessity.py --in data/p2/run5_c8_cands.jsonl --out data/p2/run5_c8_nec.jsonl --pattern derived_ore_strict --bound 10 --time 60 --procs 6 > artifacts/r5/nec_c8.log 2>&1
echo NEC_C8_DONE
nice -n 5 python3 make_coverage_sets.py gen --long --min 12 --max 16 --out data/p2/raw_long_c8c --workers 6 --tries 2500000 --cap_np 200000 --cap_pat 200000 --only derived_ore_strict --seed 5000 > artifacts/r5/gen_c8c.log 2>&1
python3 make_coverage_sets.py merge --glob "data/p2/raw_long_c8c.w*.jsonl" --out data/p2/pool_long_c8c.jsonl --prefix c8longc >> artifacts/r5/gen_c8c.log 2>&1
echo GEN_C8C_DONE
nice -n 5 python3 minlen.py --in data/p2/pool_long_c8c.jsonl --out data/p2/pool_long_c8c_minlen.jsonl --bound 8 --time 10 --procs 6 > artifacts/r5/minlen_c8c.log 2>&1
python3 - <<'PY'
import json
names = {json.loads(l)['name'] for l in open('data/p2/pool_long_c8c_minlen.jsonl') if json.loads(l)['min_lines_ub'] is None}
n = 0
with open('data/p2/run5_c8c_cands.jsonl', 'w') as fo:
    for l in open('data/p2/pool_long_c8c.jsonl'):
        r = json.loads(l)
        if r['name'] in names:
            fo.write(json.dumps({k: r[k] for k in ('name', 'thm', 'key', 'prompt', 'n_lines', 'n_prem', 'pat', 'proof') if k in r}) + '\n'); n += 1
print('c8c None-at-8 candidates', n)
PY
echo MINLEN_C8C_DONE
nice -n 5 python3 necessity.py --in data/p2/run5_c8c_cands.jsonl --out data/p2/run5_c8c_nec.jsonl --pattern derived_ore_strict --bound 10 --time 60 --procs 6 > artifacts/r5/nec_c8c.log 2>&1
echo NEC_C8C_DONE
