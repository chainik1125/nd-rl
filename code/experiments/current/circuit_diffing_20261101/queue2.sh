#!/usr/bin/env bash
# Run jobs one after another on one GPU, each under its own timeout, after an optional PID exits.
#   queue2.sh GPU JOBFILE LOG [WAIT_PID]
# JOBFILE: one job per line, "<module> <args>", the module relative to the experiment package
# (e.g. "crosscoder --b L3:0 --layer 5"); "--gpu GPU" is appended. '#' lines are skipped.
# The file is read once at start, so it can be edited for a later queue while this one runs.
set -u
GPU=$1
JOBS=$2
LOG=$3
WAIT=${4:-}
ROOT=/scratch/cowifornia/nd-rl/worktrees/circuit-diffing
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
PKG=code.experiments.current.circuit_diffing_20261101
cd "$ROOT" || exit 1
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 LEAN_GATE_WORKERS=8
export CUDA_VISIBLE_DEVICES=$GPU
mapfile -t LINES <"$JOBS"
if [[ -n "$WAIT" ]]; then
  while kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
fi
for line in "${LINES[@]}"; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -r mod args <<<"$line"
  echo "=== $(TZ=America/Los_Angeles date '+%I:%M %p') job: $line" >>"$LOG"
  # shellcheck disable=SC2086
  timeout 3600 nice "$PY" -m "$PKG.$mod" $args --gpu "$GPU" >>"$LOG" 2>&1
  echo "=== $(TZ=America/Los_Angeles date '+%I:%M %p') exit $? : $line" >>"$LOG"
done
echo "=== queue done" >>"$LOG"
