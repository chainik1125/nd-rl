#!/bin/bash
# W5 second seed: regenerate the identical pool (seed 0 => same train_L8off and
# rl_targets), but train the base model and run HER-EI with a NEW seed. The
# high/low novelty partition files are pushed in (same partitions as seed 0),
# so only model init + rollout sampling differ. Usage: runbook_w5seed.sh ARM SEED
ARM=$1; SEED=$2
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1

if [ ! -f $D/train_L8off.jsonl ]; then
  python3 -m nd.gen_data --n 450000 --min-lines 2 --max-lines 16 --steps-hi 24 --seed 0 --out $D/pool_short.jsonl > $D/gs.log 2>&1
  python3 -m nd.gen_data --n 60000 --min-lines 17 --max-lines 30 --steps-hi 44 --seed 1 --out $D/pool_long.jsonl > $D/gl.log 2>&1
  python3 exp/make_splits.py $D $D/targets_textbook.jsonl > $D/splits.log 2>&1
  python3 -m exp.reindex_data --in $D/train_L8off.jsonl --out $D/train_L8off.jsonl --seed 11 2>/dev/null || python3 -m exp.reindex_data --in $D/train_L8.jsonl --out $D/train_L8off.jsonl --seed 11
fi
echo "data ready"
# arm data dir with pushed partition as rl_targets
CD=$D/../data_${ARM}_s$SEED
mkdir -p $CD
for f in train_L8off.jsonl eval_transfer.jsonl eval_iid.jsonl targets_textbook.jsonl; do ln -sf $D/$f $CD/$f; done
ln -sf $D/rl_targets_${ARM}nov.jsonl $CD/rl_targets.jsonl

python3 -m train.train --train $D/train_L8off.jsonl --out $R/L8off_s$SEED --iters 6000 --bs 128 --layers 4 --seed $SEED > $R/L8off_s$SEED.log 2>&1
echo "base seed$SEED trained"
python3 -m exp.expert_iter --mode her --rounds 4 --data $CD --out $R/w5_${ARM}_s$SEED \
  --round0-ckpt $R/L8off_s$SEED/ckpt.pt --primitives train_L8off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 983 --rep 8 --her-cap 4000 --her-rep 4 \
  --seed $SEED > $R/w5_${ARM}_s$SEED.log 2>&1
echo "W5 ${ARM} seed$SEED DONE"
