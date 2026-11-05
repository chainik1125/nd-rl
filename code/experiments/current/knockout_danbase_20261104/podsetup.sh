#!/usr/bin/env bash
# podsetup.sh <alias> [<line>]: install a new Runpod pod's environment, then check it.
# Written by agent:claude, 2026-11-05, for the coordinator to run (copies to pods are the
# coordinator's, on Leon's approval; the supervisor never copies to a pod).
#
# Needs: the pod's ssh alias in ~/.ssh/config (`python3 register_pod.py ssh <alias> <ip> <port>`).
# Copies three files to the pod's /workspace (nothing else): pod_setup.sh, and the lab env's
# pyproject.toml and uv.lock (no data). pod_setup.sh then installs the cu130 Python env from the
# lockfile (uv sync, 40 min bound) and Lean 4 v4.34.0 (elan, 20 min bound), logs to
# /workspace/setup.log, and links the lab paths to /workspace (KD -> /workspace/kd, the working
# folder -> /workspace/exp, the lab venv path -> /workspace/venv). Takes 8 to 15 min.
# With <line>: when the line's files are already on the pod (runpod_copy_line.sh ran), runs
# podcheck.sh with its base; otherwise runpod_copy_line.sh runs it when the copy ends.
set -euo pipefail
h=${1:?usage: podsetup.sh <alias> [<line>]}
line=${2:-}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name>"; exit 2; }
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
ENV=/scratch/cowifornia/nd-rl/think/nd-rl/code
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
"${SSH[@]}" "$h" 'mkdir -p /workspace/env && nvidia-smi --query-gpu=name --format=csv,noheader'
rsync -rlpt -e "ssh -o BatchMode=yes" "$W/pod_setup.sh" "$h:/workspace/pod_setup.sh"
rsync -rlpt -e "ssh -o BatchMode=yes" "$ENV/pyproject.toml" "$ENV/uv.lock" "$h:/workspace/env/"
"${SSH[@]}" "$h" 'cd /workspace/env && md5sum pyproject.toml uv.lock' | sed 's/^/pod: /'
(cd "$ENV" && md5sum pyproject.toml uv.lock) | sed 's/^/lab: /'
echo "setup start $(date -u +%FT%TZ) (8 to 15 min; log /workspace/setup.log on the pod)"
timeout 3900 "${SSH[@]}" -o ServerAliveInterval=60 "$h" 'bash /workspace/pod_setup.sh; tail -2 /workspace/setup.log'
"${SSH[@]}" "$h" 'grep -q "^setup done" /workspace/setup.log' || { echo "SETUP FAILED: see /workspace/setup.log on $h (stop the pod if it cannot be fixed)"; exit 1; }
echo "setup done $(date -u +%FT%TZ)"
if [[ -n $line ]]; then
  base=$(cd "$W" && python3 -c "import common as C, sys; print(C.LINES[sys.argv[1]]['base'])" "$line")
  if "${SSH[@]}" "$h" "test -s /workspace/kd/bases/$base && test -s /workspace/exp/danread.py"; then
    bash "$W/podcheck.sh" "$h" "$base"
  else
    echo "the line's files are not on the pod yet: runpod_copy_line.sh $line $h runs podcheck.sh when it ends"
  fi
fi
