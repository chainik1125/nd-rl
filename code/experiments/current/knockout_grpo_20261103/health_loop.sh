#!/usr/bin/env bash
# Every 10 min: collect from ynez / temescal, rewrite STATUS.md, emit ALERT lines and new STOP / done events.
K=/scratch/cowifornia/nd-rl/knockout_grpo
declare -A seen
while true; do
  bash $K/tools/collect.sh > /dev/null 2>&1
  python3 $K/tools/health.py 2>&1 | grep --line-buffered -E "ALERT|Traceback|Error"
  for d in $K/evals/*/; do
    t=$(basename $d)
    for f in STOP_LINE line_done.json; do
      if [ -f $d/$f ] && [ -z "${seen[$t$f]}" ]; then seen[$t$f]=1; echo "EVENT $t $f: $(head -c 300 $d/$f | tr '\n' ' ')"; fi
    done
  done
  sleep 600
done
