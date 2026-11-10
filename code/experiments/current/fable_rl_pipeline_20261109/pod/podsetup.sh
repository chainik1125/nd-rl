#!/usr/bin/env bash
# podsetup.sh <alias>: copy the setup script and the lab env's lockfile to a pod, then run it.
set -euo pipefail
h=${1:?usage: podsetup.sh <alias>}
F=/scratch/cowifornia/nd-rl/fable_rl
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=/scratch/cowifornia/nd-rl/think/nd-rl/code
SSH=(ssh -F $F/ssh_config)
"${SSH[@]}" "$h" 'mkdir -p /workspace/env'
rsync -rlpt -e "ssh -F $F/ssh_config" "$HERE/pod_setup.sh" "$h:/workspace/pod_setup.sh"
rsync -rlpt -e "ssh -F $F/ssh_config" "$ENV/pyproject.toml" "$ENV/uv.lock" "$h:/workspace/env/"
timeout 3900 "${SSH[@]}" "$h" 'bash /workspace/pod_setup.sh; tail -3 /workspace/setup.log'
"${SSH[@]}" "$h" 'grep -q "^setup done" /workspace/setup.log' || { echo "SETUP FAILED on $h"; exit 1; }
echo "SETUP_OK $h $(date -u +%FT%TZ)"
