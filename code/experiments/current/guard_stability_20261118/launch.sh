#!/usr/bin/env bash
# launch.sh <alias> <test> <BASE_SEED> <LEAN_GATE_WORKERS> [extra grpo flags...]: start pod_run.sh detached on a pod
# (its own session, so ssh returns at once), then print the first lines of its log. Run ON SERRANO.
set -euo pipefail
h=$1 test=$2 bs=$3 w=$4
shift 4
flags=$(printf '%q ' "$@")
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" \
  "cd /workspace/gs && BASE_SEED=$bs SEED=$bs LEAN_GATE_WORKERS=$w setsid -f bash pod_run.sh $test $flags > /workspace/gs/$test.nohup 2>&1 < /dev/null"
sleep 15
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" \
  "head -3 /workspace/at/stability/$test/seg1/train.log; cat /workspace/gs/$test.nohup; ps -o pid,sid,etime,args -e | grep -E 'tag_grpo' | grep -v grep | cut -c1-120"
