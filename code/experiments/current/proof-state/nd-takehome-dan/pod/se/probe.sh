#!/usr/bin/env bash
# Batch probe: one pass over the RL-target pool at each batch size, to pick the largest that fits (policy 2026-09-28).
source pod/se/env.sh
CK=$1
for B in 1024 2048 4096 8192; do
  echo "=== batch $B"
  python3 state_eval.py --ckpt $CK --in data/ladder/rl_targets.jsonl --k 1 --temperature 0.8 --seed 99 \
    --batch $B --summary artifacts/se/probe_b$B.json 2>&1 | grep -E "^env |SOLVED|CUDA out of memory|Error" | tail -3
done
