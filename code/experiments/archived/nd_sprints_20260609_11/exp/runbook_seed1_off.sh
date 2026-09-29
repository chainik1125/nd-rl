#!/bin/bash
# Seed-1 replication of the offset-format chain (pod-D).
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1
python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off_s1 --iters 6000 --bs 128 --seed 1 > $R/L8off_s1.log 2>&1
echo "L8off seed1 done"
python3 -m exp.expert_iter --mode rl --rounds 5 --data $D --out $R/ei_rl_off_s1 \
  --round0-ckpt $R/L8off_s1/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 --seed 1 > $R/ei_rl_off_s1.log 2>&1
echo "SEED1 OFFSET CHAIN DONE"
