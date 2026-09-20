#!/bin/bash
# Overhang sweep chain for one cap. Usage: bash runbook_cap.sh <CAP>
# Prereqs: /root/nd/data has pool_short.jsonl, standard splits, textbook.
CAP=$1
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
export PYTHONUNBUFFERED=1

python3 -m exp.make_caps --data $D --cap $CAP
python3 -m exp.reindex_data --in $D/train_L${CAP}.jsonl --out $D/train_L${CAP}off.jsonl --seed 11
echo "cap data done"

# chain-specific data dir: symlinks + cap-specific targets as rl_targets
CD=/root/nd/data_cap$CAP
mkdir -p $CD
ln -sf $D/train_L${CAP}off.jsonl $CD/train_L${CAP}off.jsonl
ln -sf $D/targets_textbook.jsonl $CD/targets_textbook.jsonl
ln -sf $D/targets_L${CAP}.jsonl $CD/rl_targets.jsonl
head -400 $D/eval_iid.jsonl > $CD/eval_iid.jsonl
head -300 $D/eval_transfer.jsonl > $CD/eval_transfer.jsonl

python3 -m train.train --train $D/train_L${CAP}off.jsonl --out $R/L${CAP}off --iters 6000 --bs 128 --seed 0 > $R/L${CAP}off.log 2>&1
echo "cap train done"

python3 -m exp.expert_iter --mode rl --rounds 3 --data $CD --out $R/ei_cap$CAP \
  --round0-ckpt $R/L${CAP}off/ckpt.pt --primitives train_L${CAP}off.jsonl \
  --iters 6000 --n-rollouts 32 --n-targets 1200 --rep 8 --seed 0 > $R/ei_cap$CAP.log 2>&1
echo "CAP $CAP CHAIN DONE"
