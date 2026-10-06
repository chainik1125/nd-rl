#!/usr/bin/env bash
# Arm SN (state, canonical names) for one seed: Stage 1 then the same measurements as arm S.
source pod/se/env.sh
S=$1; B=${2:-2048}
python3 state_train.py --data data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl \
  --mode lean_staten --steps 6000 --recs 128 --out ckpts/se/stage1_SN_s${S}.pt --cap 6 --seed $S
bash pod/se/seed.sh SN $S $B
