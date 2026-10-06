#!/usr/bin/env bash
# Run the commands in a jobs file, N at a time. Usage: bash pod/runq.sh <jobsfile> <N>
# Each line: <name> <command...>  ; stdout/stderr -> artifacts/p2/<name>.log ; a line is skipped if artifacts/p2/<name>.done exists.
cd /workspace/nd-takehome
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p artifacts/p2 ckpts/p2 ckpts/p3 artifacts/p3 artifacts/r5 ckpts/r5 artifacts/r2 ckpts/r2 artifacts/r3 ckpts/r3 artifacts/r4 ckpts/r4
J=$1; N=${2:-3}
run_one() {
  name=$1; shift
  [ -f artifacts/p2/$name.done ] && { echo "skip $name"; return; }
  echo "$(date -u +%H:%M:%S) start $name"
  bash -c "$*" > artifacts/p2/$name.log 2>&1 && touch artifacts/p2/$name.done && echo "$(date -u +%H:%M:%S) done $name" || echo "$(date -u +%H:%M:%S) FAILED $name"
}
export -f run_one
while IFS= read -r line; do
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 15; done
  run_one $line &
  sleep 5
done < <(grep -v '^#' "$J" | grep -v '^\s*$')
wait
echo "$(date -u +%H:%M:%S) QUEUE DONE $J"
