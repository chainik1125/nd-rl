#!/usr/bin/env bash
# T1 then frozen for one arm+seed whose Stage-1 checkpoint and held-out already exist.  Usage: bash pod/se/ladders.sh <arm> <seed> [batch]
source pod/se/env.sh
A=$1; S=$2; B=${3:-2048}
BK=hf://buckets/dan-pandori/nd-rl/state-env
up() { hf buckets sync "$1" "$BK/$1" >/dev/null 2>&1 || echo "UPLOAD FAILED $1"; }
LA="--init ckpts/se/stage1_${A}_s${S}.pt --seed $S --batch $B --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl"
up artifacts/se; up ckpts/se
echo "=== ladder T1 ${A} s${S} $(date -u +%FT%TZ)"
python3 state_ladder_ei.py $LA --name la_T1_${A}_s${S}
up artifacts/se; up ckpts/se
echo "=== ladder frozen ${A} s${S} $(date -u +%FT%TZ)"
python3 state_ladder_ei.py $LA --name la_frozen_${A}_s${S} --no_train
up artifacts/se; up ckpts/se
echo "=== done $(date -u +%FT%TZ)"
