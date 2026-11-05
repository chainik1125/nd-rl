#!/usr/bin/env bash
# One-time setup of a Runpod pod for a knockout_danbase line (bounded; log in /workspace/setup.log).
set -euo pipefail
exec >> /workspace/setup.log 2>&1
echo "setup start $(date -u +%FT%TZ)"
mkdir -p /workspace/kd /workspace/exp /workspace/env /scratch/cowifornia/nd-rl/think/nd-rl/code \
  /scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current
ln -sfn /workspace/kd /scratch/cowifornia/nd-rl/knockout_danbase
ln -sfn /workspace/exp /scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
# the Python env from the lockfile (torch cu130), not the image's torch
cd /workspace/env
export UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/workspace/uvcache UV_LINK_MODE=copy
timeout 2400 uv sync --frozen --no-install-project --no-dev --python 3.11
ln -sfn /workspace/venv /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv
echo "venv done $(date -u +%FT%TZ)"
# Lean 4 v4.34.0, core only (no Mathlib, no lake build)
curl -sSfL --max-time 120 https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /workspace/elan-init.sh
ELAN_HOME=/workspace/elan timeout 1200 sh /workspace/elan-init.sh -y --default-toolchain leanprover/lean4:v4.34.0 --no-modify-path
ln -sfn /workspace/elan /root/.elan
/root/.elan/bin/lean --version
echo "setup done $(date -u +%FT%TZ)"
