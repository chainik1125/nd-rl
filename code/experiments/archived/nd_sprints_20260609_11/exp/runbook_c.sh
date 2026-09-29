#!/bin/bash
# pod-c runbook: offset-index ablation chain.
# Prereqs: code + data (train_L8.jsonl, eval_*.jsonl, rl_targets.jsonl,
# targets_textbook.jsonl) present in /root/nd.
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1

python3 -m exp.reindex_data --in $D/train_L8.jsonl --out $D/train_L8off.jsonl --seed 11
echo "reindex done"

python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off --iters 6000 --bs 128 > $R/L8off.log 2>&1
echo "L8off train done"

# round-0 cliff snapshot for the offset format
for ev in eval_iid eval_ood eval_local; do
  python3 -m train.evalgen --ckpt $R/L8off/ckpt.pt --targets $D/$ev.jsonl \
    --k 1 --temp 0.0 --bs 1536 --out $RES/cliff_L8off_${ev}_greedy.jsonl \
    > $RES/log_L8off_${ev}_greedy.txt 2>&1
  python3 -m train.evalgen --ckpt $R/L8off/ckpt.pt --targets $D/$ev.jsonl \
    --k 8 --temp 0.8 --bs 1536 --out $RES/cliff_L8off_${ev}_pass8.jsonl \
    > $RES/log_L8off_${ev}_pass8.txt 2>&1
  echo "L8off eval $ev done"
done

# RL chain on the offset format
python3 -m exp.expert_iter --mode rl --rounds 5 --data $D --out $R/ei_rl_off \
  --round0-ckpt $R/L8off/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 > $R/ei_rl_off.log 2>&1
echo "OFFSET RL CHAIN DONE"
