#!/usr/bin/env bash
# Wait for an add-on run to finish, then score it on the sealed sets (GPU given). Usage: score_addon.sh <arm> <gpu>
set -uo pipefail
AB=/scratch/cowifornia/nd-rl/ab; PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
ARM=$1; GPU=$2; RUN=$AB/_addon/$ARM/runs/addon_L7-10_s7; OUT=$AB/logs/final/addon_$ARM; mkdir -p $OUT
until [ -f $RUN/summary.json ] && grep -q '"status"' $RUN/summary.json; do sleep 15; done
python3 -c "import json; s=json.load(open('$RUN/summary.json')); print('addon $ARM status', s.get('status'), 'heldout', s.get('heldout_greedy'), 'rl_secs', round(s.get('rl_secs') or 0))"
cp $RUN/final.pt $OUT/final.pt
for spec in "dev 0" "test10 0" "test11 0" "test11 1" "textbook 0" "textbook 1"; do set -- $spec
  CUDA_VISIBLE_DEVICES=$GPU $PY $AB/common/score_host.py --ckpt $OUT/final.pt --set $1 --k 32 --seed $2 --out $OUT/${1}_k32_s$2.json --per-theorem $OUT/${1}_k32_s$2_per_theorem.jsonl 2>&1 | grep -E "pass@|rror" || true
done
echo "addon $ARM scored"
