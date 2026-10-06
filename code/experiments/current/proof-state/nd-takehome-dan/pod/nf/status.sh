#!/usr/bin/env bash
# VPS side: one-line status per job on each pod. Usage: bash pod/nf/status.sh [pods...]
for n in ${@:-nf-1 nf-2}; do
  [ -f ~/.config/nd-rl/pods/$n ] || continue
  echo "== $n"; podrun $n 'for f in artifacts/nf/logs/*.log; do j=$(basename $f .log); [ $j = setup ] && continue; s=run; [ -f artifacts/nf/$j.done ] && s=DONE; [ -f artifacts/nf/$j.failed ] && s=FAILED; echo "$j [$s] $(grep -v "^\[lean_gate\]" $f | tail -n 1 | cut -c1-120)"; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime' < /dev/null 2>&1 | grep -v "^Warning"
done
