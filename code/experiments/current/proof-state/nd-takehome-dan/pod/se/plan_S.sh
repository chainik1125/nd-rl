#!/usr/bin/env bash
# Arm S for one seed after the 05:19 environment crash: resume T1 from its last completed round (or start it), then
# frozen, then arm SH.  Usage: bash pod/se/plan_S.sh <seed> <next T1 round, 1 = fresh> [batch]
source pod/se/env.sh
S=$1; R=$2; B=${3:-2048}
BK=hf://buckets/dan-pandori/nd-rl/state-env
up() { hf buckets sync "$1" "$BK/$1" >/dev/null 2>&1 || echo "UPLOAD FAILED $1"; }
LA="--seed $S --batch $B --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl"
D=artifacts/se/la_T1_S_s$S
if [ "$R" -gt 1 ]; then
  cp -n $D/args.json $D/args_r1.json
  echo "=== ladder T1 S s$S RESUME from round $R $(date -u +%FT%TZ)"
  python3 state_ladder_ei.py --init ckpts/se/stage1_S_s$S.pt --name la_T1_S_s$S $LA --resume --start_round $R --rounds $((9 - R))
else
  echo "=== ladder T1 S s$S $(date -u +%FT%TZ)"
  python3 state_ladder_ei.py --init ckpts/se/stage1_S_s$S.pt --name la_T1_S_s$S $LA
fi
up artifacts/se; up ckpts/se
echo "=== ladder frozen S s$S $(date -u +%FT%TZ)"
rm -rf artifacts/se/la_frozen_S_s$S
python3 state_ladder_ei.py --init ckpts/se/stage1_S_s$S.pt --name la_frozen_S_s$S $LA --no_train
up artifacts/se; up ckpts/se
echo "=== arm SH s$S $(date -u +%FT%TZ)"
bash pod/se/arm_sh.sh $S $B
echo "=== plan done $(date -u +%FT%TZ)"
