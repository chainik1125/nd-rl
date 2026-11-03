#!/usr/bin/env bash
# Emits new kq / watchdog / line event lines from serrano, ynez and temescal (poll every 60 s).
K=/scratch/cowifornia/nd-rl/knockout_grpo
SSHC="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
declare -A seen
first=1
while true; do
  for h in serrano ynez temescal; do
    if [ $h = serrano ]; then out=$(cat $K/logs/kq_serrano.log $K/logs/watchdog.log 2>/dev/null)
    else out=$(SSH_AUTH_SOCK= timeout 40 $SSHC $h "cat $K/logs/kq_$h.log $K/logs/watchdog.log 2>/dev/null" 2>/dev/null || echo "SSH-FAIL $h"); fi
    if [ "${out:0:8}" = "SSH-FAIL" ]; then echo "[$h] SSH-FAIL"; continue; fi
    n=$(echo "$out" | wc -l); prev=${seen[$h]:-0}
    if [ "$n" -gt "$prev" ] && [ $first = 0 ]; then
      echo "$out" | tail -n $((n - prev)) | grep -E --line-buffered "LAUNCH|EXIT|STALL|OVERRUN|STOP|FAIL|failed|Traceback|wait \(GPU .* another user" | sed "s/^/[$h] /"
    fi
    seen[$h]=$n
  done
  first=0
  sleep 60
done
