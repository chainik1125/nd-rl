#!/usr/bin/env bash
# run.sh <alias> <name> <timeout seconds> <python module and args...>: start a bounded job on a
# pod under nohup, log to /workspace/runs/<name>.log. Refuses if a job of that name is running.
set -euo pipefail
h=$1; name=$2; to=$3; shift 3
F=/scratch/cowifornia/nd-rl/fable_rl
ssh -F $F/ssh_config "$h" "mkdir -p /workspace/runs; if [ -f /workspace/runs/$name.pid ] && kill -0 \$(cat /workspace/runs/$name.pid) 2>/dev/null; then echo ALREADY_RUNNING; exit 3; fi; cd /workspace/fr && PYTHONUNBUFFERED=1 LEAN_GATE_WORKERS=7 setsid nohup timeout $to /workspace/venv/bin/python $* > /workspace/runs/$name.log 2>&1 < /dev/null & echo \$! > /workspace/runs/$name.pid; echo STARTED $name pid \$(cat /workspace/runs/$name.pid)"
