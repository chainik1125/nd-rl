#!/usr/bin/env bash
# VPS side: launch addendum 2's extra-seed chains on one pod.  bash pod/nf/launch_seeds.sh <pod> <pool> [<pool> ...]
cd /home/dan/work/noise-floor
P=$1; shift
bash pod/nf/sync.sh $P push pod/nf/job.sh > /dev/null < /dev/null
mapfile -t LINES < <(python3 pod/nf/jobs_seeds.py "$@")
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r name deps cmd <<< "$line"
  pre=""; [ "$deps" != "-" ] && pre="until [ -f artifacts/nf/$deps.done ]; do sleep 20; done; "
  podrun $P "[ -f artifacts/nf/logs/$name.log ] && { echo already $name; exit 0; }; mkdir -p pod/nf/cmds artifacts/nf/logs; cat > pod/nf/cmds/$name.cmd <<'XEOF'
$pre$cmd
XEOF
setsid nohup bash pod/nf/job.sh $name \"\$(cat pod/nf/cmds/$name.cmd)\" > /dev/null 2>&1 < /dev/null & disown; sleep 1; echo launched $name" < /dev/null 2>&1 | tail -1
done
