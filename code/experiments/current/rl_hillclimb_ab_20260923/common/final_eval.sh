#!/usr/bin/env bash
# Final sealed evaluation, run by the monitor on GPU 4 after the deadline.
# Usage: final_eval.sh <label> <checkpoint-host-path> [gpu]
# Scores: dev (k=32, seed 1), test10, test11 (L11-14), textbook, each at k=32 seeds 0 and 1; per-theorem files for the monitor's analysis.
set -euo pipefail
LABEL=$1; CKPT=$2; GPU=${3:-4}
AB=/scratch/cowifornia/nd-rl/ab; OUT=$AB/logs/final/$LABEL; mkdir -p $OUT
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
cp "$CKPT" $OUT/final.pt
for set in dev test10 test11 textbook; do
  for seed in 0 1; do
    [ "$set" = dev ] && [ "$seed" = 0 ] && [ "$LABEL" != baseline ] && continue   # dev seed 0 already exists from the run
    CUDA_VISIBLE_DEVICES=$GPU $PY $AB/common/score_host.py --ckpt $OUT/final.pt --set $set --k 32 --seed $seed \
      --out $OUT/${set}_k32_s${seed}.json --per-theorem $OUT/${set}_k32_s${seed}_per_theorem.jsonl 2>&1 | grep -E "pass@|Error|error" || true
  done
done
echo "final eval done: $LABEL"
