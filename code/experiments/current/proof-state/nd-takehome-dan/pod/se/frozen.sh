#!/usr/bin/env bash
# The frozen control only, for an arm+seed whose Stage-1 checkpoint is in the bucket (run in parallel with its T1).
# Usage: bash pod/se/frozen.sh <arm> <seed> [batch]
source pod/se/env.sh
A=$1; S=$2; B=${3:-2048}
BK=hf://buckets/dan-pandori/nd-rl/state-env
[ -f ckpts/se/stage1_${A}_s${S}.pt ] || hf buckets cp $BK/ckpts/se/stage1_${A}_s${S}.pt ckpts/se/stage1_${A}_s${S}.pt
md5sum ckpts/se/stage1_${A}_s${S}.pt
echo "=== ladder frozen ${A} s${S} $(date -u +%FT%TZ)"
python3 state_ladder_ei.py --init ckpts/se/stage1_${A}_s${S}.pt --seed $S --batch $B --train data/p2/train_depth3_f0_a1.jsonl \
  --heldout data/p2/heldout.jsonl --name la_frozen_${A}_s${S} --no_train
hf buckets sync artifacts/se $BK/artifacts/se >/dev/null 2>&1 || echo "UPLOAD FAILED"
echo "=== done $(date -u +%FT%TZ)"
