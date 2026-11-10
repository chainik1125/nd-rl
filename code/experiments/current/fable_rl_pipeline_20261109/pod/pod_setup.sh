#!/usr/bin/env bash
# One-time setup of a Runpod pod for fable_rl (bounded; log in /workspace/setup.log).
# Installs the lab's Python env from its lockfile (torch cu130) and Lean 4 v4.34.0 core.
set -euo pipefail
exec >> /workspace/setup.log 2>&1
echo "setup start $(date -u +%FT%TZ)"
mkdir -p /workspace/env /workspace/fr /workspace/data /workspace/runs
cd /workspace/env
export UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/workspace/uvcache UV_LINK_MODE=copy
timeout 2400 uv sync --frozen --no-install-project --no-dev --python 3.11
echo "venv done $(date -u +%FT%TZ)"
curl -sSfL --max-time 120 https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /workspace/elan-init.sh
ELAN_HOME=/workspace/elan timeout 1200 sh /workspace/elan-init.sh -y --default-toolchain leanprover/lean4:v4.34.0 --no-modify-path
ln -sfn /workspace/elan /root/.elan
/root/.elan/bin/lean --version
rm -rf /workspace/uvcache
echo "setup done $(date -u +%FT%TZ)"
