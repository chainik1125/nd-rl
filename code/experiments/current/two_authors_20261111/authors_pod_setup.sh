#!/usr/bin/env bash
# authors_pod_setup.sh: one-time setup of a Runpod pod for the two-authors experiment. It runs ON the pod (copied
# there by `authors_pod.sh setup`), bounded, logging to /workspace/setup.log; the last line is "setup done <UTC>".
# Adapted from knockout_danbase_20261104/pod_setup.sh (2026-11-05): the same Python env (uv sync of the lab lockfile,
# torch cu130) and Lean 4 v4.34.0 (elan, core only), and the lab paths linked to /workspace so every script runs
# unchanged:
#   /scratch/cowifornia/nd-rl/knockout_danbase                         -> /workspace/kd   (Dan's code, GRPO snapshot, pools)
#   /scratch/cowifornia/nd-rl/two_authors                              -> /workspace/ta   (TA: sets, corpus, bases, runs)
#   /scratch/cowifornia/nd-rl/knockout_grpo                            -> /workspace/kg   (dan_axis.json only)
#   /scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments/current -> /workspace/exp (the three code folders)
#   /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv                   -> /workspace/venv
#   /root/.elan                                                        -> /workspace/elan
# Takes 8 to 15 minutes (knockout pods, A40). Never touches any API key.
set -euo pipefail
exec >> /workspace/setup.log 2>&1
echo "setup start $(date -u +%FT%TZ)"
mkdir -p /workspace/kd /workspace/ta /workspace/kg /workspace/exp /workspace/env \
  /scratch/cowifornia/nd-rl/think/nd-rl/code /scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments
ln -sfn /workspace/kd /scratch/cowifornia/nd-rl/knockout_danbase
ln -sfn /workspace/ta /scratch/cowifornia/nd-rl/two_authors
ln -sfn /workspace/kg /scratch/cowifornia/nd-rl/knockout_grpo
ln -sfn /workspace/exp /scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments/current
# the Python env from the lab lockfile (torch cu130), not the image's torch
command -v uv >/dev/null || timeout 600 python3 -m pip install --quiet uv
cd /workspace/env
export UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/workspace/uvcache UV_LINK_MODE=copy
timeout 2400 uv sync --frozen --no-install-project --no-dev --python 3.11
ln -sfn /workspace/venv /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv
echo "venv done $(date -u +%FT%TZ)"
# Lean 4 v4.34.0, core only (no Mathlib, no lake build)
if [[ ! -x /workspace/elan/bin/lean ]]; then
  curl -sSfL --max-time 120 https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /workspace/elan-init.sh
  ELAN_HOME=/workspace/elan timeout 1200 sh /workspace/elan-init.sh -y --default-toolchain leanprover/lean4:v4.34.0 --no-modify-path
fi
ln -sfn /workspace/elan /root/.elan
/root/.elan/bin/lean --version
echo "setup done $(date -u +%FT%TZ)"
