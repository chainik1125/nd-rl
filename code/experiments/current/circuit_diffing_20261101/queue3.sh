#!/usr/bin/env bash
# Stage 3 job queue on ynez: jobs one after another on the one reserved GPU, each under its own
# timeout. common.assert_gpu_free refuses a job if the GPU has any compute process.
#   queue3.sh GPU JOBFILE LOG
# JOBFILE: one job per line, "<timeout s> <module> <args>" ("--gpu GPU" is appended); '#' lines
# are skipped. A file $ART/logs/STOP halts the queue before its next job.
set -u
GPU=$1
JOBS=$2
LOG=$3
ROOT=/scratch/cowifornia/nd-rl/worktrees/circuit-diffing
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
PKG=code.experiments.current.circuit_diffing_20261101
ART=/scratch/cowifornia/nd-rl/circuits
cd "$ROOT" || exit 1
# at most 12 cores: 8 Lean workers + 3 torch threads + the main thread
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export LEAN_GATE_WORKERS=8 CIRCUITS_RUNS=$ART/runs CIRCUITS_ALLOWED_GPUS=$GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mapfile -t LINES <"$JOBS"
for line in "${LINES[@]}"; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -r tmo mod args <<<"$line"
  if [[ -e $ART/logs/STOP ]]; then
    echo "=== $(date -u +%H:%MZ) STOP file: halted before: $line" >>"$LOG"
    break
  fi
  echo "=== $(date -u +%H:%MZ) job: $line" >>"$LOG"
  # shellcheck disable=SC2086
  timeout "$tmo" nice "$PY" -m "$PKG.$mod" $args --gpu "$GPU" >>"$LOG" 2>&1
  echo "=== $(date -u +%H:%MZ) exit $? : $line" >>"$LOG"
done
echo "=== $(date -u +%H:%MZ) queue done" >>"$LOG"
