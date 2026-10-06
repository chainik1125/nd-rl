#!/usr/bin/env bash
# Run 2 (p3), second candidate batch: depth4 with the generator's box-depth cap raised to 5 (target pools only), a larger
# negi_ande_hyp batch, and IMPE-chain schema instances; then the oracle on each. Waits for targets.sh.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=1
until grep -q R2_TARGETS_DONE artifacts/r2_targets.log; do sleep 30; done
nice -n 5 python3 make_coverage_sets.py gen --long --min 8 --max 14 --out data/r2/raw_depth4b --workers 6 --tries 1500000 --cap_np 500000 --cap_pat 500000 --only p2:depth4 --seed 7100 --gen_max_depth 5 > artifacts/r2/gen_depth4b.log 2>&1
python3 make_coverage_sets.py merge --glob "data/r2/raw_depth4b.w*.jsonl" --out data/r2/pool_depth4.jsonl --prefix r2depth4 >> artifacts/r2/gen_depth4b.log 2>&1
echo "GEN_DONE depth4b $(wc -l < data/r2/pool_depth4.jsonl)"
nice -n 5 python3 make_coverage_sets.py gen --long --min 7 --max 14 --out data/r2/raw_negi_ande_hypb --workers 6 --tries 7000000 --cap_np 500000 --cap_pat 500000 --only p2:negi_ande_hyp --seed 7200 > artifacts/r2/gen_negi_ande_hypb.log 2>&1
python3 make_coverage_sets.py merge --glob "data/r2/raw_negi_ande_hyp*.w*.jsonl" --out data/r2/pool_negi_ande_hyp.jsonl --prefix r2negi >> artifacts/r2/gen_negi_ande_hypb.log 2>&1
echo "GEN_DONE negi_ande_hypb $(wc -l < data/r2/pool_negi_ande_hyp.jsonl)"
python3 chain_pool.py --n 1600 --out data/r2/cands_impe_chain4.jsonl --exclude data/p2/heldout.jsonl --seed 0 > artifacts/r2/chain_pool.log 2>&1
echo "CHAIN_CANDS $(wc -l < data/r2/cands_impe_chain4.jsonl)"
for spec in depth4:10 negi_ande_hyp:10; do
  IFS=: read pat bound <<< "$spec"
  python3 - $pat <<'PY'
import json, sys, random
pat = sys.argv[1]
rs = [json.loads(l) for l in open(f'data/r2/pool_{pat}.jsonl')]
random.Random(0).shuffle(rs)
with open(f'data/r2/cands_{pat}.jsonl', 'w') as fo:
    for r in rs[:6000]:
        fo.write(json.dumps({k: r[k] for k in ('name', 'thm', 'key', 'prompt', 'n_lines', 'n_prem', 'pat', 'pat2', 'proof') if k in r}) + '\n')
print(pat, 'cands', min(6000, len(rs)), 'of', len(rs))
PY
  rm -f data/r2/nec_$pat.jsonl
  nice -n 5 python3 necessity.py --in data/r2/cands_$pat.jsonl --out data/r2/nec_$pat.jsonl --pattern $pat --bound $bound --time 60 --procs 6 > artifacts/r2/nec_$pat.log 2>&1
  echo "NEC_DONE $pat"
done
rm -f data/r2/nec_impe_chain4.jsonl
nice -n 5 python3 necessity.py --in data/r2/cands_impe_chain4.jsonl --out data/r2/nec_impe_chain4.jsonl --pattern impe_chain4 --bound 10 --time 60 --procs 6 > artifacts/r2/nec_impe_chain4.log 2>&1
echo "NEC_DONE impe_chain4"
echo R2_TARGETS2_DONE
