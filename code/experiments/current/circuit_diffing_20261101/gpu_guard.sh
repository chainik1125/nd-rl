#!/usr/bin/env bash
# Every 2 min: any compute process on GPU $1 that is not a descendant of PID $2 is reported as
# ALERT (owner and pid only; never its environment). Exits when PID $2 exits.
#   gpu_guard.sh GPU QUEUE_PID LOG
GPU=$1
QPID=$2
LOG=$3
desc() { # is $1 a descendant of $QPID?
  local p=$1
  while [[ -n "$p" && "$p" -gt 1 ]]; do
    [[ "$p" -eq "$QPID" ]] && return 0
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  return 1
}
while kill -0 "$QPID" 2>/dev/null; do
  for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader); do
    if ! desc "$pid"; then
      echo "$(date -u +%H:%MZ) ALERT foreign process on GPU $GPU: pid $pid user $(ps -o user= -p "$pid")" >>"$LOG"
    fi
  done
  sleep 120
done
echo "$(date -u +%H:%MZ) guard exit (queue $QPID gone)" >>"$LOG"
