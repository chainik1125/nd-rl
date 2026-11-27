#!/usr/bin/env bash
# m2_chain.sh <chain-name> (POD side; started by `m2_pod.sh chain` under setsid + nohup). Reads runs/_chains/<name>/chain.txt:
#   <job> <timeout-seconds> <continue-codes, e.g. 0 or 0,11> [--nogpu] -- <shell command>
# (blank lines and # comments ignored). Each job goes through m2_job.sh (own state, timeout, heartbeat, GPU lock; --nogpu
# jobs take download.lock instead). The chain stops at the first job whose exit code is not in its continue list (FAILED
# holds "<job> rc=<n>"); a job whose state/DONE exists is skipped (rc recorded as 0 skipped), so a re-run resumes.
# state/: pid, current, rcs ("<job> <rc>" per line), RUNNING, DONE | FAILED, STOP (set by m2_stop.sh --chain: no more jobs).
# log/chain.log. env: M2_ROOT.
set -uo pipefail
CN=${1:?usage: m2_chain.sh <chain-name>}
M2=${M2_ROOT:-/workspace/m2}; D=$M2/runs/_chains/$CN; ST=$D/state; P=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ST" "$D/log"
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" | tee -a "$D/log/chain.log"; }
[ -s "$D/chain.txt" ] || { log "NO_CHAIN_FILE"; exit 2; }
if [ -f "$ST/pid" ] && [ -f "$ST/RUNNING" ] && kill -0 "$(cut -d' ' -f1 "$ST/pid")" 2>/dev/null; then log "ALREADY_RUNNING"; exit 3; fi
rm -f "$ST/DONE" "$ST/FAILED" "$ST/STOP" "$ST/rcs" "$ST/current"
echo "$$ $(ps -o pgid= $$ | tr -d ' ') $(utc) $CN" > "$ST/pid"; echo "$(utc)" > "$ST/RUNNING"
log "CHAIN_START $CN"
final=0
while IFS= read -r ln || [ -n "$ln" ]; do
  [[ -z ${ln//[[:space:]]/} || $ln =~ ^[[:space:]]*# ]] && continue
  head=${ln%% -- *}; cmd=${ln#* -- }
  read -r job to codes flag <<< "$head"
  nogpu=0; [ "$flag" = --nogpu ] && nogpu=1
  if [ -f "$ST/STOP" ]; then log "STOPPED before $job"; echo "$(utc) stopped before $job" > "$ST/FAILED"; final=1; break; fi
  if [ -f "$M2/runs/$job/state/DONE" ]; then log "SKIP_DONE $job"; echo "$job 0" >> "$ST/rcs"; continue; fi
  echo "$job" > "$ST/current"; log "JOB_START $job timeout=$to continue=$codes nogpu=$nogpu"
  NOGPU=$nogpu setsid --wait bash "$P/m2_job.sh" "$job" "$to" -- bash -c "$cmd"
  rc=$?
  echo "$job $rc" >> "$ST/rcs"; log "JOB_END $job rc=$rc"
  if [[ ,$codes, != *,$rc,* ]]; then
    log "CHAIN_STOP at $job: rc=$rc not in continue list $codes"; echo "$(utc) $job rc=$rc" > "$ST/FAILED"; final=1; break
  fi
done < "$D/chain.txt"
rm -f "$ST/RUNNING" "$ST/current"
[ $final = 0 ] && { echo "$(utc)" > "$ST/DONE"; log "CHAIN_DONE $CN"; }
exit $final
