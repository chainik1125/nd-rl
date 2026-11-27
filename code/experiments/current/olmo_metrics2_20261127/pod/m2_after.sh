#!/usr/bin/env bash
# m2_after.sh <alias> <job> -- <serrano command...>   (orchestrator helper, 2026-11-27; serrano side, log only)
# Polls the pod every 60 s until runs/<job>/state/DONE exists, then runs the command once (e.g. `bash pod/m2_pod.sh chain ...`).
# If the job ends FAILED, it logs and exits without running the command, unless --any is given (then it runs it too).
# <job> may be a chain's state path, e.g. _chains/seg1 (its state/ holds DONE or FAILED). Bounded by `timeout` by the caller.
set -uo pipefail
h=${1:?alias}; job=${2:?job}; shift 2; ANY=0; [[ ${1:-} == --any ]] && { ANY=1; shift; }; [[ ${1:-} == -- ]] && shift
LOG=/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/logs/after_${h}.log; mkdir -p "$(dirname "$LOG")"
ev() { echo "$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date +'%-I:%M %p PT')) $*" | tee -a "$LOG"; }
ev "WAIT $job then: $*"
while true; do
  s=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" "cd /workspace/m2/runs/$job/state 2>/dev/null && ls" 2>/dev/null | tr '\n' ' ')
  if [[ " $s " == *" DONE "* ]]; then ev "DONE $job -> starting"; "$@" >> "$LOG" 2>&1; ev "STARTED rc=$?"; exit 0; fi
  if [[ " $s " == *" FAILED "* && $ANY == 1 ]]; then ev "FAILED $job -> starting anyway (--any)"; "$@" >> "$LOG" 2>&1; ev "STARTED rc=$?"; exit 0; fi
  if [[ " $s " == *" FAILED "* ]]; then ev "FAILED $job -> not starting"; exit 1; fi
  sleep 60
done
