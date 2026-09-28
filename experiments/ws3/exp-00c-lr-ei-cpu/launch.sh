#!/usr/bin/env bash
# Exact launch command for exp-00c. Run from the repo root; laptop CPU, sequential (the arms
# would contend for the same cores). make_split.py was run before the proposal commit.
set -euo pipefail
cd "$(dirname "$0")/../../../code"
E=../experiments/ws3/exp-00c-lr-ei-cpu
{
  echo "=== make_pool start $(date -u +%FT%TZ)"
  uv run python $E/make_pool.py
  echo "=== make_pool done $(date -u +%FT%TZ)"
  # first GRPO arm also computes the frozen-checkpoint baseline, shared by every arm
  uv run python ws3/run.py --config $E/configs/lr3e-5.yaml
  for arm in lr1e-4 lr3e-4 ei; do
    mkdir -p $E/run_$arm
    cp -R $E/run_lr3e-5/baseline $E/run_$arm/
    uv run python ws3/run.py --config $E/configs/$arm.yaml --stages train,evaluate,report
  done
  uv run python $E/analyze.py
  echo "=== ALL DONE $(date -u +%FT%TZ)"
} > $E/stdout.log 2>&1
