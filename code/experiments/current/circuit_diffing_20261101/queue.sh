#!/usr/bin/env bash
# Run crosscoder jobs one after another on one GPU, each under its own timeout.
#   queue.sh GPU JOBFILE LOG
# JOBFILE: one line per job, the crosscoder.py arguments after --gpu (e.g. "--b L3:0 --layer 3").
# Lines starting with '#' are skipped. A job whose output exists is skipped by crosscoder.py.
set -u
GPU=$1
JOBS=$2
LOG=$3
ROOT=/scratch/cowifornia/nd-rl/worktrees/circuit-diffing
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
cd "$ROOT" || exit 1
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 LEAN_GATE_WORKERS=8
export CUDA_VISIBLE_DEVICES=$GPU
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  echo "=== $(TZ=America/Los_Angeles date '+%I:%M %p') job: $line" >>"$LOG"
  # shellcheck disable=SC2086
  timeout 3600 nice "$PY" -m code.experiments.current.circuit_diffing_20261101.crosscoder \
    --gpu "$GPU" $line >>"$LOG" 2>&1
  echo "=== $(TZ=America/Los_Angeles date '+%I:%M %p') exit $? : $line" >>"$LOG"
done <"$JOBS"
echo "=== queue done" >>"$LOG"
