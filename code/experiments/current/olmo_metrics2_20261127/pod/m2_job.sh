#!/usr/bin/env bash
# m2_job.sh <job> <timeout-seconds> -- <command...>   (POD side; started by m2_pod.sh job under setsid + nohup)
# Runs ONE job under `timeout --kill-after`, holding flock on $M2/gpu.lock (or $M2/download.lock with NOGPU=1), so two
# GPU jobs never run at once. Files under $M2/runs/<job>/:
#   state/pid        "<runner pid> <process group id> <start utc> <job>"
#   state/jobpg      process-group id of the command (its `timeout` leads it)
#   state/RUNNING    start utc;  state/DONE "<utc> rc=0";  state/FAILED "<utc> rc=<n> [stopped]"
#   state/heartbeat  every HB_INTERVAL s (default 60): ts=<epoch> job=<job> newest=<path> newest_mtime=<epoch>
#                    newest_size=<bytes> gpu_mem=<MiB> gpu_util=<%>   (newest = newest-modified file of the run
#                    directory outside state/)
#   logs/job.log     the command's stdout+stderr;  logs/runner.log  this script's own events
# Refusals (nothing started): DONE exists and FORCE!=1 (exit 4); lock held (exit 3); no GPU_OK in meta/gpu_status
# (exit 3, GPU jobs only); GPU memory still in use after GPU_WAIT s (exit 5, state/FAILED rc=5 GPU_BUSY). env: M2_ROOT (default /workspace/m2), FORCE=1, NOGPU=1, HB_INTERVAL.
set -uo pipefail
JOB=${1:?usage: m2_job.sh <job> <timeout-seconds> -- <command...>}
TO=${2:?timeout seconds}
shift 2
[[ ${1:-} == -- ]] && shift
[[ $# -gt 0 ]] || { echo "no command"; exit 2; }
M2=${M2_ROOT:-/workspace/m2}
R=$M2/runs/$JOB; ST=$R/state; LG=$R/logs
HB_INTERVAL=${HB_INTERVAL:-60}
utc() { date -u +%FT%TZ; }
mkdir -p "$ST" "$LG"
rlog() { echo "$(utc) $*" >> "$LG/runner.log"; }
if [[ -f $ST/DONE && ${FORCE:-0} != 1 ]]; then rlog "REFUSED_DONE $JOB (state/DONE exists; --force to rerun)"; exit 4; fi
LOCK=$M2/gpu.lock; [[ ${NOGPU:-0} == 1 ]] && LOCK=$M2/download.lock
exec 8> "$LOCK"
flock -n 8 || { rlog "LOCKED $JOB: another job holds $LOCK"; exit 3; }
if [[ ${NOGPU:-0} != 1 ]] && ! grep -q "^GPU_OK" "$M2/meta/gpu_status" 2>/dev/null; then rlog "GPU_BAD_OR_NO_SETUP $JOB"; exit 3; fi
# GPU-free gate (GPU jobs, nvidia-smi present): wait up to GPU_WAIT s (default 120) for every GPU's used memory to fall
# below GPU_FREE_MIB (default 2000); a vLLM engine-core child can outlive its parent. Otherwise refuse: exit 5, state/FAILED.
gpu_busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -v lim="${GPU_FREE_MIB:-2000}" '$1 + 0 >= lim { b = 1 } END { exit !b }'; }
if [[ ${NOGPU:-0} != 1 ]] && command -v nvidia-smi > /dev/null 2>&1; then
  t_end=$(( $(date +%s) + ${GPU_WAIT:-120} ))
  while gpu_busy && [[ $(date +%s) -lt $t_end ]]; do sleep "${GPU_POLL:-5}"; done
  if gpu_busy; then
    rlog "GPU_BUSY $JOB: used memory $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB >= ${GPU_FREE_MIB:-2000} after ${GPU_WAIT:-120}s; not started"
    rm -f "$ST/DONE" "$ST/RUNNING"; echo "$(utc) rc=5 GPU_BUSY" > "$ST/FAILED"; exit 5
  fi
fi
rm -f "$ST/DONE" "$ST/FAILED" "$ST/heartbeat" "$ST/stop_requested"
echo "$$ $(ps -o pgid= $$ | tr -d ' ') $(utc) $JOB" > "$ST/pid"
echo "$(utc)" > "$ST/RUNNING"
rlog "START $JOB timeout=${TO}s cmd: $*"
heartbeat() {
  local line f mt sz mem
  f=$(find "$R" -type f ! -path "$ST/*" -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -1)
  mt=$(echo "$f" | awk '{printf "%d", $1}'); sz=$(echo "$f" | awk '{print $2}'); f=$(echo "$f" | cut -d' ' -f3-)
  mem=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  mem=${mem:-na,na}
  echo "ts=$(date +%s) job=$JOB newest=${f:--} newest_mtime=${mt:-0} newest_size=${sz:-0} gpu_mem=${mem%%,*} gpu_util=${mem##*,}" \
    > "$ST/heartbeat.tmp" && mv "$ST/heartbeat.tmp" "$ST/heartbeat"
}
heartbeat
( exec 8>&-   # the heartbeat must not hold the lock
  while sleep "$HB_INTERVAL"; do heartbeat; done ) > /dev/null 2>&1 &
HB=$!
STOPPED=0
trap 'STOPPED=1; rlog "SIGTERM received (stop)"' TERM
trap 'kill $HB 2>/dev/null' EXIT
cd "$M2" || exit 2
# the command runs under its own `timeout` (which makes itself a process-group leader): its pid is the group id, recorded
# in state/jobpg so that m2_stop.sh can signal exactly this job's group
timeout --signal=TERM --kill-after=60 "$TO" "$@" > "$LG/job.log" 2>&1 &
TP=$!
echo "$TP" > "$ST/jobpg"
wait $TP; rc=$?
while kill -0 $TP 2>/dev/null; do wait $TP; rc=$?; done   # a trapped SIGTERM interrupts `wait`; keep waiting
# sweep the command's process group: a grandchild (vLLM engine core) that outlived the command gets SIGTERM, then after
# KILL_GRACE s (default 10) SIGKILL
grp_alive() { ps -eo pgid=,stat= 2>/dev/null | awk -v g="$TP" '$1 == g && $2 !~ /^Z/ { f = 1 } END { exit !f }'; }
if grp_alive; then
  rlog "command group $TP still has members after exit: SIGTERM, SIGKILL in ${KILL_GRACE:-10}s"
  kill -TERM -- "-$TP" 2>/dev/null
  for _ in $(seq 1 "${KILL_GRACE:-10}"); do grp_alive || break; sleep 1; done
  grp_alive && { rlog "SIGKILL to command group $TP"; kill -KILL -- "-$TP" 2>/dev/null; }
fi
kill $HB 2>/dev/null; wait $HB 2>/dev/null
heartbeat
rm -f "$ST/RUNNING"
if [[ $rc == 0 ]]; then echo "$(utc) rc=0" > "$ST/DONE"; rlog "DONE $JOB"
else echo "$(utc) rc=$rc$([[ $STOPPED == 1 || -f $ST/stop_requested ]] && echo ' stopped')$([[ $rc == 124 ]] && echo ' timeout')" > "$ST/FAILED"; rlog "FAILED $JOB rc=$rc"; fi
exit $rc
