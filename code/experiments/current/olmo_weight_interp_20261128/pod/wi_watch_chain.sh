#!/usr/bin/env bash
# wi_watch_chain.sh <alias> <chain-name> <chain-file>   (SERRANO side; log only: it never stops or kills anything)
# For each job of the chain file in order: waits until the job has a state on the pod (or the chain has ended), then runs
# metrics 2's pod/m2_watch.sh for it (unedited; pointed at this run's driver and data root). That watcher appends to
# $WI_DATA/logs/watch_<alias>.log, in UTC and PT:
#   STALL    heartbeat older than 10 min, or the newest output file has not grown for GROW_STALL s (default 2400: a cell
#            that waits for its write, or whose remaining draws are all long, writes nothing for a while)
#   OVERRUN  the job has run 1.6 x its expected minutes (expected: download 10, preflight 2, write 20, cell 30, delete 2)
#   NOSTATUS ssh or status failed;   EXIT  the job ended (DONE / FAILED / STALE / ABSENT)
# Bound it from the caller, e.g.  timeout 4h bash pod/wi_watch_chain.sh rp-wi cells $WI_DATA/chains/chain_cells.txt
# env: WI_DATA, POLL (120 s), GROW_STALL, WI_WATCH_WAIT (60 s between looks for a job that has not started).
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd); h=${1:?alias}; cn=${2:?chain name}; cf=${3:?chain file}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name>"; exit 2; }
[[ -r $cf ]] || { echo "cannot read $cf"; exit 2; }
export WI_DATA=${WI_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit/interp}
M2WATCH=$(dirname "$(dirname "$P")")/olmo_metrics2_20261127/pod/m2_watch.sh
[[ -r $M2WATCH ]] || { echo "missing $M2WATCH"; exit 2; }
R=/workspace/wi
expected() { case $1 in download_*) echo 10 ;; preflight) echo 2 ;; write_*) echo 20 ;; cell_*) echo 30 ;; rm_*) echo 2 ;; *) echo 30 ;; esac; }
while read -r job _rest <&3; do   # the list is on fd 3: ssh and the watcher must not eat it
  while true; do
    s=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" "ls $R/runs/$job/state 2>/dev/null; ls $R/runs/_chains/$cn/state 2>/dev/null | sed 's/^/chain_/'" 2>/dev/null | tr '\n' ' ')
    [[ " $s " == *" RUNNING "* || " $s " == *" DONE "* || " $s " == *" FAILED "* ]] && break
    [[ " $s " == *" chain_DONE "* || " $s " == *" chain_FAILED "* ]] && exit 0
    sleep "${WI_WATCH_WAIT:-60}"
  done
  M2_POD_CMD=$P/wi_pod.sh M2_DATA=$WI_DATA GROW_STALL=${GROW_STALL:-2400} bash "$M2WATCH" "$h" "$job" "$(expected "$job")" < /dev/null
done 3< <(grep -vE '^[[:space:]]*(#|$)' "$cf")
