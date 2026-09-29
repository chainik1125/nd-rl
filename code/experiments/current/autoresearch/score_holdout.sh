#!/usr/bin/env bash
# On the pod (after `pod.py up` warmed the fork and Lean): score every line of <jobs> with
# score_holdout.py, <procs> at a time on the one GPU. A line is "<hf-run-dir> <pool>".
# Usage: bash score_holdout.sh <jobs> <procs> <out-dir>
set -uo pipefail
JOBS=$1; PROCS=$2; OUT=$3
export ND_FORK=/workspace/nd-takehome
export LEAN_GATE_WORKERS=$(( $(nproc) / PROCS ))  # share the CPUs between concurrent Lean gates
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
mkdir -p "$OUT"
cd /workspace/autoresearch
export PY OUT
xargs -P "$PROCS" -L 1 bash -c \
  '$PY score_holdout.py "$0" --pool "$1" --out "$OUT/$0.$1.json" > "$OUT/$0.$1.log" 2>&1 \
   || echo "FAILED $0 $1"' < "$JOBS"
# Concurrent jobs can run the GPU out of memory: retry each failure alone.
while read -r run pool; do
  [ -f "$OUT/$run.$pool.json" ] || $PY score_holdout.py "$run" --pool "$pool" \
    --out "$OUT/$run.$pool.json" > "$OUT/$run.$pool.log" 2>&1 < /dev/null || echo "FAILED $run $pool"
done < "$JOBS"
grep -h " solved " "$OUT"/*.log
