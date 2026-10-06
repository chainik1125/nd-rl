#!/usr/bin/env bash
# Run 2 (p3), third batch: oracle on 30k impi_ore candidates (3 % keep the pattern at >= 7 lines) and the nested-ORE schema pool.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=1
until grep -q R2_TARGETS2_DONE artifacts/r2_targets2.log; do sleep 30; done
python3 - <<'PY'
import json, random
rs = [json.loads(l) for l in open('data/r2/pool_impi_ore.jsonl')]
random.Random(1).shuffle(rs)
done = {json.loads(l)['name'] for l in open('data/r2/nec_impi_ore.jsonl')}
n = 0
with open('data/r2/cands_impi_ore.jsonl', 'a') as fo:
    for r in rs:
        if r['name'] in done: continue
        fo.write(json.dumps({k: r[k] for k in ('name', 'thm', 'key', 'prompt', 'n_lines', 'n_prem', 'pat', 'pat2', 'proof') if k in r}) + '\n'); n += 1
        if n >= 24000: break
print('impi_ore extra cands', n)
PY
nice -n 5 python3 necessity.py --in data/r2/cands_impi_ore.jsonl --out data/r2/nec_impi_ore.jsonl --pattern impi_ore --bound 10 --time 60 --procs 6 > artifacts/r2/nec_impi_ore2.log 2>&1
echo "NEC_DONE impi_ore2"
python3 ore_pool.py --n 800 --out data/r2/cands_nested_ore.jsonl --exclude data/p2/heldout.jsonl --seed 0 > artifacts/r2/ore_pool.log 2>&1
rm -f data/r2/nec_nested_ore.jsonl
nice -n 5 python3 necessity.py --in data/r2/cands_nested_ore.jsonl --out data/r2/nec_nested_ore.jsonl --pattern nested_ore --bound 13 --time 120 --procs 6 > artifacts/r2/nec_nested_ore.log 2>&1
echo "NEC_DONE nested_ore"
echo R2_TARGETS3_DONE
