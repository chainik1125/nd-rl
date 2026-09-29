#!/bin/bash
# Capacity falsifier (theory prediction P3): the 12-line ceiling of 4-layer
# models is a NESTING ceiling, not a length ceiling. Train a depth-4 model on
# LOCAL-ONLY (nesting-0) proofs and HER-EI on local-only band targets; if its
# frontier runs PAST 12 lines on flat proofs, length alone is not the ceiling.
cd /root/nd/code
D=/root/nd/data_nestM
R=/root/nd/runs
export PYTHONUNBUFFERED=1
mkdir -p $D

# local-only pools (no IMPI/NEGI/ORE -> nesting 0), long band included
python3 -m nd.gen_data --n 450000 --min-lines 2 --max-lines 8 --steps-hi 20 --seed 0 --local-only --out $D/pool_short.jsonl > $D/gs.log 2>&1
python3 -m nd.gen_data --n 120000 --min-lines 9 --max-lines 30 --steps-hi 60 --seed 1 --local-only --out $D/pool_long.jsonl > $D/gl.log 2>&1
echo "nest pools done"
python3 exp/make_splits.py $D $D/../data/targets_textbook.jsonl > $D/splits.log 2>&1
python3 -m exp.reindex_data --in $D/train_L8.jsonl --out $D/train_L8off.jsonl --seed 11
ln -sf $D/../data/targets_textbook.jsonl $D/targets_textbook.jsonl
echo "nest splits done"
python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off_nestM --iters 6000 --bs 128 --layers 4 --seed 0 > $R/L8off_nestM.log 2>&1
echo "nest train done"
python3 -m exp.expert_iter --mode her --rounds 4 --data $D --out $R/ei_nestM \
  --round0-ckpt $R/L8off_nestM/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 --her-cap 4000 \
  --her-rep 4 --layers 4 --seed 0 > $R/ei_nestM.log 2>&1
echo "NEST DONE"
