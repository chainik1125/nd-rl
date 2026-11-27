#!/usr/bin/env bash
# stage_inputs.sh: copy the rank1 code next to its inputs so that `m2_pod.sh push` (which rsyncs $M2_DATA/inputs/) puts it at
# /workspace/m2/inputs/rank1/code/ on the pod. Serrano, CPU, no pod contact. Run after any edit of rank1/*.py, then push.
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd)
M2_DATA=${M2_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit/metrics2}
mkdir -p "$M2_DATA/inputs/rank1/code"
cp -p "$X"/*.py "$M2_DATA/inputs/rank1/code/"
rm -f "$M2_DATA/inputs/rank1/code/__pycache__" 2>/dev/null || true
ls "$M2_DATA/inputs/rank1/code"
