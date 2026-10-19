#!/usr/bin/env bash
# Final sealed evaluation for run 2, run by the monitor after the deadline through the GPU lock.
# Usage: final_eval_ab2.sh <label> <checkpoint-host-path>
# Scores: sealed_L12-14 at k=64 and k=256, sealed_L15-16 at k=256, textbook at k=256, dev (seed 1) at k=32,
# frontier (full set, k=64). Per-theorem rows go to monitor/final/<label>/ (monitor-only).
set -uo pipefail
LABEL=$1; CKPT=$2
AB=/scratch/cowifornia/nd-rl/ab2; OUT=$AB/monitor/final/$LABEL; mkdir -p $OUT
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
export AB_WORKER=monitor
cp "$CKPT" $OUT/final.pt
g=$($PY $AB/frozen/gpulock.py acquire --wait 7200 --run final_$LABEL 2>/dev/null | tail -1)
[ -n "$g" ] || { echo "no GPU for $LABEL"; exit 1; }
for spec in "sealed_L12-14 64" "sealed_L12-14 256" "sealed_L15-16 256" "textbook 256" "dev 32" "frontier 64"; do set -- $spec
  extra=""; [ "$1" = frontier ] && extra="--full"; seed=0; [ "$1" = dev ] && seed=1
  CUDA_VISIBLE_DEVICES=$g $PY $AB/common/score_host.py --ckpt $OUT/final.pt --set $1 --k $2 --seed $seed $extra \
    --out $OUT/${1}_k$2_s$seed.json --per-theorem $OUT/${1}_k$2_s${seed}_per_theorem.jsonl 2>&1 | grep -E "pass@|rror"
done
$PY $AB/frozen/gpulock.py release $g >/dev/null 2>&1 || true
echo "final eval done: $LABEL"
