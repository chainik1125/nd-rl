#!/usr/bin/env bash
# Work-queue worker for one GPU: pull2.sh GPU WAIT_MIN [MODEL=T0] [K=32]
# Repeatedly moves the next shard <work>/todo/NAME.jsonl to <work>/doing/ (atomic mv, so several GPUs share one
# queue) and runs run_passrate.sh on it (seed = 100 + the trailing digits of NAME). Before each shard it waits (poll
# 30 s, at most WAIT_MIN minutes) until the GPU is free: < 500 MiB in use AND no compute process on it (a pass-rate
# run drops to ~400 MiB while Lean judges a chunk, so the memory test alone is not enough). Stops when todo/ is empty.
# (pull.sh, the first version, had only the memory test and once double-booked a GPU; superseded by this script.)
set -u
G=$1 WAIT=$2 MODEL=${3:-T0} K=${4:-32}
H=$(dirname "$(readlink -f "$0")")
W=/scratch/cowifornia/nd-rl/think/runs/pools/hard12_work
mkdir -p $W/todo $W/doing
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G")
free() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G")
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
  [ "$used" -lt 500 ] && [ "$apps" -eq 0 ]
}
while true; do
  for i in $(seq 0 $((WAIT * 2))); do
    if free; then sleep 5; free && break; fi
    [ "$i" -eq $((WAIT * 2)) ] && { echo "$(date -u +%FT%TZ) GPU $G busy after $WAIT min: stop"; exit 1; }
    sleep 30
  done
  next=""
  for f in $(ls $W/todo/*.jsonl 2>/dev/null | sort); do
    if mv "$f" $W/doing/ 2>/dev/null; then next=$W/doing/$(basename "$f"); break; fi
  done
  [ -z "$next" ] && { echo "$(date -u +%FT%TZ) GPU $G: queue empty"; exit 0; }
  name=$(basename "$next" .jsonl)
  idx=$(echo "$name" | grep -o '[0-9]*$'); seed=$((100 + 10#${idx:-0}))
  echo "$(date -u +%FT%TZ) GPU $G takes $name (seed $seed)"
  "$H/run_passrate.sh" "$G" "$name" "$next" "$MODEL" "$K" "$seed"
done
