#!/usr/bin/env bash
# (re)start the queue waiter. Invoked as: podrun "bash /workspace/nd-takehome/pod/rq.sh"  (its own cmdline does not contain 'queue2')
pkill -f 'queue2.sh'
sleep 1
cd /workspace/nd-takehome
setsid nohup bash /workspace/nd-takehome/pod/queue2.sh > /workspace/nd-takehome/artifacts/queue2.log 2>&1 < /dev/null &
sleep 1
ps aux | grep '[q]ueue2.sh' | awk '{print $2, $12, $13}'
