#!/usr/bin/env bash
# VPS side (resume phase): one line per job on each pod plus the current round of the running ladder job.
for n in ${@:-dsg-1 dsg-2}; do
  [ -f ~/.config/nd-rl/pods/$n ] || continue
  echo "== $n"
  podrun $n 'for f in artifacts/dsg/logs/*.log; do j=$(basename $f .log); [ $j = setup ] && continue; s=run; [ -f artifacts/dsg/$j.done ] && s=DONE; [ -f artifacts/dsg/$j.failed ] && s=FAILED; [ $s = run ] || continue; echo "$j [$s] $(grep -v "^\[lean_gate\]" $f | tail -n 1 | cut -c1-120)"; done; echo "done: $(ls artifacts/dsg/*.done 2>/dev/null | wc -l)/16  failed: $(ls artifacts/dsg/*.failed 2>/dev/null | wc -l)"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' < /dev/null 2>&1 | grep -v "^Warning"
done
