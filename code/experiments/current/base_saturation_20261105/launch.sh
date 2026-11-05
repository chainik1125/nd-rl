#!/usr/bin/env bash
# launch.sh <alias> ROLE [ARGS]: start pod/go.sh ROLE on a base_saturation pod, detached (setsid,
# nohup), and print its first events. Roles: see pod/go.sh. Nothing is copied.
set -euo pipefail
h=${1:?usage: launch.sh <alias> ROLE [ARGS]}
shift
[[ $h =~ ^rp-sat-[a-z0-9-]+$ ]] || { echo "alias must look like rp-sat-<name>"; exit 2; }
q=$(printf ' %q' "$@")
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
  "cd /workspace/sat && mkdir -p logs status && cp -n alias status/alias 2>/dev/null; \
   setsid nohup bash exp/go.sh$q > logs/go_\$(date -u +%Y%m%dT%H%M%SZ)_$1.log 2>&1 < /dev/null & \
   sleep 3; tail -3 events.log"
