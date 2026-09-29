#!/bin/bash
# Protocol-transition experiment, data-master pod (also runs the HER arm).
# Generates the shared instance, trains the shared starting checkpoint,
# signals READY (other arms copy data+ckpt), then runs hindsight-EI.
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1

python3 -m nd.gen_data --n 450000 --min-lines 2 --max-lines 16 --steps-hi 24 --seed 0 --out $D/pool_short.jsonl > $D/gen_short.log 2>&1
python3 -m nd.gen_data --n 60000 --min-lines 17 --max-lines 30 --steps-hi 44 --seed 1 --out $D/pool_long.jsonl > $D/gen_long.log 2>&1
echo "pools done"
python3 exp/make_splits.py $D $D/targets_textbook.jsonl > $D/splits.log 2>&1
python3 -m exp.reindex_data --in $D/train_L8.jsonl --out $D/train_L8off.jsonl --seed 11
python3 -m exp.reindex_data --in $D/oracle_long.jsonl --out $D/oracle_longoff.jsonl --seed 12
echo "splits done"
python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off_x --iters 6000 --bs 128 --seed 0 > $R/L8off_x.log 2>&1
echo "ckpt done"
touch /root/nd/READY

python3 -m exp.expert_iter --mode her --rounds 5 --data $D --out $R/ei_her \
  --round0-ckpt $R/L8off_x/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 \
  --her-cap 4000 --her-rep 4 --seed 0 > $R/ei_her.log 2>&1
echo "HER ARM DONE"
