#!/usr/bin/env bash
# m2_stop.sh <job> (POD side): SIGTERM to the recorded process groups of THAT job only (runner group from state/pid and
# command group from state/jobpg; each is checked to be a live group whose leader's pgid matches before signalling).
# Never pkill. env: M2_ROOT.
set -uo pipefail
M2=${M2_ROOT:-/workspace/m2}
if [ "${1:-}" = --chain ]; then   # stop a chain: no further job starts, the current job is stopped as below
  C=$M2/runs/_chains/${2:?chain name}/state; touch "$C/STOP"
  cur=$(cat "$C/current" 2>/dev/null); echo "STOP flag set for chain $2 (current job: ${cur:--})"
  [ -n "$cur" ] && [ -f "$C/RUNNING" ] && exec bash "$0" "$cur"
  exit 0
fi
JOB=${1:?usage: m2_stop.sh <job> | --chain <name>}
ST=$M2/runs/$JOB/state
[ -f "$ST/pid" ] && [ ! -f "$ST/DONE" ] && [ ! -f "$ST/FAILED" ] || { echo "NOT_RUNNING $JOB"; exit 1; }
read -r rpid rpg _ < "$ST/pid"
live() { kill -0 "$1" 2>/dev/null && [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" != Z ]; }   # a zombie is not alive
live "$rpid" || { echo "NOT_RUNNING $JOB (runner $rpid gone)"; exit 1; }
sig() { # sig <label> <pgid>
  [ -n "$2" ] && [ "$(ps -o pgid= -p "$2" 2>/dev/null | tr -d ' ')" = "$2" ] || { echo "skip $1: no live group leader $2"; return; }
  echo "SIGTERM $1 pgid=$2"; kill -TERM -- "-$2" 2>/dev/null
}
[ -f "$ST/jobpg" ] && sig command "$(cat "$ST/jobpg")"
touch "$ST/stop_requested"   # the runner is NOT signalled (its own ps/sleep children would die); it notices the command ended
for _ in $(seq 1 "${STOP_GRACE:-20}"); do live "$rpid" || break; sleep 1; done
if live "$rpid"; then   # grace over: SIGKILL the command's group (catches a TERM-ignoring grandchild), then the runner's
  [ -f "$ST/jobpg" ] && { echo "SIGKILL command pgid=$(cat "$ST/jobpg")"; kill -KILL -- "-$(cat "$ST/jobpg")" 2>/dev/null; }
  for _ in $(seq 1 15); do live "$rpid" || break; sleep 1; done
fi
if live "$rpid"; then echo "SIGKILL runner pgid=$rpg"; kill -KILL -- "-$rpg" 2>/dev/null; sleep 1; fi
if live "$rpid"; then echo "STILL_ALIVE $JOB (runner $rpid)"; exit 1; fi
echo "STOPPED $JOB"
