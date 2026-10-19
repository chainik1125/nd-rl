#!/usr/bin/env bash
# After the deadline: run the top recipes with two fresh seeds each, sequentially on ONE card, so the ranking
# does not depend on card placement. Usage: same_card_compare.sh <gpu> <node-id> [<node-id> ...]
# Each run: 1,200 s budget, seeds 101 and 102, harness with explicit --gpu (no lock contention after the deadline).
set -uo pipefail
GPU=$1; shift
AB=/scratch/cowifornia/nd-rl/ab2; PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
export AB_WORKER=monitor
mkdir -p $AB/monitor/samecard && cd $AB/monitor/samecard
for node in "$@"; do
  cp $AB/nodes/$node/rl.py rl_$node.py
  for seed in 101 102; do
    out=runs/${node}_s${seed}
    $PY $AB/frozen/harness.py --rl rl_$node.py --out $out --seed $seed --gpu $GPU > $out.log 2>&1
    python3 -c "import json; s=json.load(open('$out/summary.json')); print('$node seed $seed', {k: s.get(k) for k in ('status','metric','by_len','frontier','frontier_by_len','heldout_greedy','rl_secs')}, 'rounds', (s.get('rl_stats') or {}).get('rounds'))"
  done
done
echo "same-card comparison done on GPU $GPU"
