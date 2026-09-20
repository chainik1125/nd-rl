#!/bin/bash
# Fresh offset-chain instance with per-round iid (forgetting) tracking.
# Regenerates data (statistically equivalent fresh instance), builds splits,
# trains L8off, runs the chain with iid_pass1 recorded every round.
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1

python3 -m nd.gen_data --n 450000 --min-lines 2 --max-lines 16 --steps-hi 24 --seed 0 --out $D/pool_short.jsonl > $D/gen_short.log 2>&1
python3 -m nd.gen_data --n 60000 --min-lines 17 --max-lines 30 --steps-hi 44 --seed 1 --out $D/pool_long.jsonl > $D/gen_long.log 2>&1
echo "pools done"
python3 exp/make_splits.py $D $D/targets_textbook.jsonl > $D/splits.log 2>&1
echo "splits done"
python3 -m exp.reindex_data --in $D/train_L8.jsonl --out $D/train_L8off.jsonl --seed 11
echo "reindex done"
python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off_f --iters 6000 --bs 128 --seed 2 > $R/L8off_f.log 2>&1
echo "train done"
python3 -m exp.expert_iter --mode rl --rounds 5 --data $D --out $R/ei_off_forget \
  --round0-ckpt $R/L8off_f/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 --seed 2 > $R/ei_off_forget.log 2>&1
echo "FORGET CHAIN DONE"
