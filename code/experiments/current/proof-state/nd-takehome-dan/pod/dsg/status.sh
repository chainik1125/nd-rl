#!/usr/bin/env bash
# VPS side: one-line status per job on each pod. Usage: bash pod/dsg/status.sh [pods...]
for n in ${@:-dsg-1 dsg-2 dsg-3}; do
  [ -f ~/.config/nd-rl/pods/$n ] || continue
  echo "== $n"; podrun $n 'for f in artifacts/dsg/logs/*.log; do j=$(basename $f .log); [ $j = setup ] && continue; s=run; [ -f artifacts/dsg/$j.done ] && s=DONE; [ -f artifacts/dsg/$j.failed ] && s=FAILED; echo "$j [$s] $(grep -v "^\[lean_gate\]" $f | tail -n 1 | cut -c1-140)"; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' < /dev/null 2>&1 | grep -v "^Warning"
done
