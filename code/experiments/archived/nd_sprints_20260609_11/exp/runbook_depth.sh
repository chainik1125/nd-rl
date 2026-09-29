#!/bin/bash
# Depth-scaling arm: train cap-8 offset SFT at depth D, then 3 HER rounds.
# Usage: bash runbook_depth.sh <D>
D=$1
cd /root/nd/code
DATA=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1

python3 -m train.train --train $DATA/train_L8off.jsonl --out $R/L8off_d$D \
  --iters 6000 --bs 128 --layers $D --seed 0 > $R/L8off_d$D.log 2>&1
echo "depth $D sft done"

python3 -m exp.expert_iter --mode her --rounds 3 --data $DATA --out $R/ei_d$D \
  --round0-ckpt $R/L8off_d$D/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 \
  --her-cap 4000 --her-rep 4 --layers $D --seed 0 > $R/ei_d$D.log 2>&1
echo "DEPTH $D DONE"
