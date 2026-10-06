#!/usr/bin/env bash
# Resume a T1 ladder from its next round and upload.  Usage: bash pod/se/resume_t1.sh <arm> <seed> <next round> [batch]
source pod/se/env.sh
A=$1; S=$2; R=$3; B=${4:-2048}
D=artifacts/se/la_T1_${A}_s${S}
cp -n $D/args.json $D/args_r1.json
echo "=== ladder T1 ${A} s${S} RESUME from round $R $(date -u +%FT%TZ)"
python3 state_ladder_ei.py --init ckpts/se/stage1_${A}_s${S}.pt --name la_T1_${A}_s${S} --seed $S --batch $B \
  --train data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl --resume --start_round $R --rounds $((9 - R))
hf buckets sync artifacts/se hf://buckets/dan-pandori/nd-rl/state-env/artifacts/se >/dev/null 2>&1 || echo "UPLOAD FAILED"
hf buckets sync ckpts/se hf://buckets/dan-pandori/nd-rl/state-env/ckpts/se >/dev/null 2>&1 || echo "UPLOAD FAILED"
echo "=== done $(date -u +%FT%TZ)"
