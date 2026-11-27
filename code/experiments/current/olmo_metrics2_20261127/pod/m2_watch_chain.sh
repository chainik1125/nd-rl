#!/usr/bin/env bash
# m2_watch_chain.sh <alias> <chain-name> <chain-file>   (orchestrator helper, 2026-11-27; SERRANO side, log only, kills nothing)
# For each job of the chain file in order: waits until the job has a state on the pod (or this chain has ended), then runs
# (stdin of the loop is fd 3 and ssh / m2_watch.sh get /dev/null: the first version lost the rest of the list to ssh)
# m2_watch.sh for it with expected minutes = timeout / 90 (OVERRUN = 1.6 x that). Bound it with `timeout` from the caller.
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd); h=${1:?alias}; cn=${2:?chain name}; cf=${3:?chain file}
while read -r job to _rest <&3; do
  while true; do
    s=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" "ls /workspace/m2/runs/$job/state 2>/dev/null; ls /workspace/m2/runs/_chains/$cn/state 2>/dev/null | sed 's/^/chain_/'" 2>/dev/null | tr '\n' ' ')
    [[ " $s " == *" RUNNING "* || " $s " == *" DONE "* || " $s " == *" FAILED "* ]] && break
    [[ " $s " == *" chain_DONE "* || " $s " == *" chain_FAILED "* ]] && exit 0
    sleep 60
  done
  GROW_STALL=${GROW_STALL:-2400} bash "$P/m2_watch.sh" "$h" "$job" $(( to / 90 )) < /dev/null
done 3< <(grep -vE '^[[:space:]]*(#|$)' "$cf")
