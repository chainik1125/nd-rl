#!/usr/bin/env bash
# Exact launch command for exp-00. Run from the repo root. All stages, laptop CPU, no GPU.
set -euo pipefail
cd "$(dirname "$0")/../../../code"
mkdir -p ../experiments/ws3/exp-00-local-check/run
exec uv run python ws3/run.py --config ../experiments/ws3/exp-00-local-check/config.yaml \
  > ../experiments/ws3/exp-00-local-check/run/stdout.log 2>&1
