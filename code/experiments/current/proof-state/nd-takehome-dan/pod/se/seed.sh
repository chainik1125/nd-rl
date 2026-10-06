#!/usr/bin/env bash
# One arm+seed of run state-env: held-out greedy, then ladder T1, then the frozen control at equal attempts.
# Usage: bash pod/se/seed.sh <arm S|SH> <seed> [batch]
source pod/se/env.sh
A=$1; S=$2; B=${3:-2048}
CK=ckpts/se/stage1_${A}_s${S}.pt
BK=hf://buckets/dan-pandori/nd-rl/state-env
up() { hf buckets sync "$1" "$BK/$1" >/dev/null 2>&1 || echo "UPLOAD FAILED $1"; }

echo "=== heldout ${A} s${S} $(date -u +%FT%TZ)"
python3 state_eval.py --ckpt $CK --in data/p2/heldout.jsonl --k 1 --temperature 0 --batch $B \
  --out artifacts/se/heldout_${A}_s${S}.jsonl --summary artifacts/se/heldout_${A}_s${S}.json
up ckpts/se; up artifacts/se

echo "=== ladder T1 ${A} s${S} $(date -u +%FT%TZ)"
python3 state_ladder_ei.py --init $CK --name la_T1_${A}_s${S} --seed $S --batch $B \
  --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl
up artifacts/se; up ckpts/se

echo "=== ladder frozen ${A} s${S} $(date -u +%FT%TZ)"
python3 state_ladder_ei.py --init $CK --name la_frozen_${A}_s${S} --seed $S --batch $B --no_train \
  --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl
up artifacts/se; up ckpts/se
echo "=== seed done $(date -u +%FT%TZ)"
