#!/usr/bin/env bash
# Validation-36 for every round checkpoint of an arm: greedy (prove.py, the submission interface) and pass@32 (eval_set.py).
# usage: bash eval_val36_rounds.sh ei_abs_s0 8
ARM=$1; R=$2
python3 - <<'PY'
import json
rs=[json.loads(l) for l in open('targets/validation_36_reference_proofs.jsonl')]
with open('artifacts/val36_targets.jsonl','w') as f:
    for r in rs:
        f.write(json.dumps({'name': r['name'], 'thm': r['thm'], 'prompt': r['prompt'], 'n_lines': r['reference_lines'], 'bin': r['bin']})+'\n')
PY
for r in $(seq 1 $R); do
  CK=ckpts/${ARM}_r${r}.pt
  [ -f $CK ] || continue
  python3 prove.py --ckpt $CK --in targets/validation_36.jsonl --out artifacts/${ARM}_r${r}_val36_greedy.jsonl --greedy
  python3 eval_targets.py --proofs artifacts/${ARM}_r${r}_val36_greedy.jsonl > artifacts/${ARM}_r${r}_val36_greedy.txt
  python3 eval_set.py --ckpt $CK --in artifacts/val36_targets.jsonl --out artifacts/${ARM}_r${r}_val36_k32.jsonl --k 32 --temperature 0.8 --seed 0 --summary artifacts/${ARM}_r${r}_val36_k32.json > /dev/null
  echo "$ARM r$r greedy: $(grep SOLVED artifacts/${ARM}_r${r}_val36_greedy.txt)  pass@32: $(python3 -c "import json;d=json.load(open('artifacts/${ARM}_r${r}_val36_k32.json'));print(d['solved'],'/',d['n'])")"
done
