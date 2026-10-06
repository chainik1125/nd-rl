#!/usr/bin/env bash
# Run 2 (p3): generate candidate target pools per pattern with the UNCHANGED generator (long mode, 7-14 lines; 12-16 for
# nested_ore), output filter --only p2:<pattern>; merge by class; then the necessity oracle per pattern.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=1
mkdir -p artifacts/r2 data/r2
for spec in depth4:7:14:1500000 impe_chain4:7:14:1500000 impi_ore:7:14:1500000 negi_ande_hyp:7:14:1500000 ori_ore:7:14:1500000 nested_ore:12:16:1500000; do
  IFS=: read pat mn mx tries <<< "$spec"
  nice -n 5 python3 make_coverage_sets.py gen --long --min $mn --max $mx --out data/r2/raw_$pat --workers 6 --tries $tries --cap_np 500000 --cap_pat 500000 --only p2:$pat --seed 7000 > artifacts/r2/gen_$pat.log 2>&1
  python3 make_coverage_sets.py merge --glob "data/r2/raw_$pat.w*.jsonl" --out data/r2/pool_$pat.jsonl --prefix r2$pat >> artifacts/r2/gen_$pat.log 2>&1
  echo "GEN_DONE $pat $(wc -l < data/r2/pool_$pat.jsonl)"
done
for spec in depth4:10 impe_chain4:10 impi_ore:10 negi_ande_hyp:10 ori_ore:10 nested_ore:13; do
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
  nice -n 5 python3 necessity.py --in data/r2/cands_$pat.jsonl --out data/r2/nec_$pat.jsonl --pattern $pat --bound $bound --time 60 --procs 6 > artifacts/r2/nec_$pat.log 2>&1
  echo "NEC_DONE $pat"
done
echo R2_TARGETS_DONE
