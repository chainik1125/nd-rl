#!/usr/bin/env bash
# Smoke pipeline for every variant, inside ONE GPU lease (run through tests/with_gpu.py):
#   make_frozen (smoke pretrain) -> harness --smoke --no-score with rl_repr.py -> evaluate --limit 6 at L6/12/18
# plus the harness equivalence check: rl_C.py vs rl_repr.py smoke runs on the released 039 s0 pretrain.
#   $PY tests/with_gpu.py --name repr_smoke -- bash tests/run_smoke.sh [variants...]
set -euo pipefail
R=/scratch/cowifornia/nd-rl/ab3/repr; PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
O=$R/tests/out; cd $O
VARS=${@:-abs rel posrand nope posline}
for v in $VARS; do
  $PY $R/make_frozen.py --label smoke_$v --dest $O/frozen_$v --pretrain $O/pt_smoke_$v/pretrain.pt
  rm -rf $O/smoke_rl_$v
  $PY $O/frozen_$v/harness.py --rl $R/rl_repr.py --out $O/smoke_rl_$v --seed 101 --gpu env --no-score --smoke > $O/smoke_rl_$v.log 2>&1 || true
  $PY -c "import json;s=json.load(open('$O/smoke_rl_$v/summary.json'));print('$v smoke RL', s['status'], 'heldout', s['heldout_greedy'], 'rounds', s['rl_stats'].get('rounds'), 'compute', round(s['compute']['frac'],3), s.get('reason','')[-300:])"
  $PY $R/evaluate.py --ckpt $O/smoke_rl_$v/final.pt --label ${v}_s0 --limit 6 --lengths 6,12,18 --gpu env --out-root $O/eval_smoke --force > $O/eval_smoke_$v.log 2>&1
  grep -h "pass@1" $O/eval_smoke_$v.log | cut -c1-160
done
if [[ " $VARS " == *" abs "* ]]; then
  $PY $R/make_frozen.py --label ref039 --dest $O/frozen_ref039 --pretrain /scratch/cowifornia/nd-rl/checkpoints/runs/039_muon-adamw-finish-5pct_s0/pretrain.pt
  for f in C repr; do
    rl=$R/rl_repr.py; [ $f = C ] && rl=/scratch/cowifornia/nd-rl/ab3/lengthexp/rl_C.py
    rm -rf $O/eq_$f
    $PY $O/frozen_ref039/harness.py --rl $rl --out $O/eq_$f --seed 101 --gpu env --no-score --smoke > $O/eq_$f.log 2>&1 || true
  done
  $PY $R/tests/compare_smoke.py $O/eq_C $O/eq_repr
fi
