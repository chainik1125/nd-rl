#!/usr/bin/env bash
# Emits new kq / watchdog / line event lines from serrano, ynez and temescal (poll every 60 s).
K=/scratch/cowifornia/nd-rl/knockout_grpo
SSHC="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
declare -A seen
while true; do
  for h in serrano ynez temescal; do
    if [ $h = serrano ]; then out=$(cat $K/logs/kq_serrano.log $K/logs/watchdog.log 2>/dev/null; grep -h -E "STOP|line done|Traceback|Error|rc [1-9]" $K/logs/*_L[23].log $K/logs/*line*.log 2>/dev/null | tail -5)
    else out=$(SSH_AUTH_SOCK= timeout 40 $SSHC $h "cat $K/logs/kq_$h.log $K/logs/watchdog.log 2>/dev/null" 2>/dev/null || echo "SSH-FAIL $h"); fi
    n=$(echo "$out" | wc -l); prev=${seen[$h]:-0}
    if [ "$n" -gt "$prev" ]; then
      echo "$out" | tail -n $((n - prev)) | grep -E --line-buffered "LAUNCH|EXIT|STALL|OVERRUN|STOP|FAIL|failed|Traceback|wait \(GPU .* another user" | sed "s/^/[$h] /"
      seen[$h]=$n
    fi
  done
  sleep 60
done
