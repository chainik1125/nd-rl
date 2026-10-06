#!/usr/bin/env bash
# VPS side: launch every job of a pod, one podrun call per job under setsid/nohup; each job waits for its
# dependencies' done markers on the pod. Jobs whose log already exists are skipped.
#   bash pod/nf/launch.sh <pod-name> <1|2> [only-job-name]
cd /home/dan/work/noise-floor
P=$1; N=$2; ONLY=$3
bash pod/nf/sync.sh $P push pod/nf/job.sh > /dev/null < /dev/null
mapfile -t LINES < <(python3 pod/nf/jobs.py $N)
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r name deps cmd <<< "$line"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  pre=""; IFS=',' read -ra DS <<< "$deps"
  for d in "${DS[@]}"; do [ -n "$d" ] && [ "$d" != "-" ] && pre="${pre}until [ -f artifacts/nf/$d.done ]; do sleep 30; done; "; done
  podrun $P "[ -f artifacts/nf/logs/$name.log ] && { echo already $name; exit 0; }; mkdir -p pod/nf/cmds artifacts/nf/logs; cat > pod/nf/cmds/$name.cmd <<'XEOF'
$pre$cmd
XEOF
setsid nohup bash pod/nf/job.sh $name \"\$(cat pod/nf/cmds/$name.cmd)\" > /dev/null 2>&1 < /dev/null & disown; sleep 1; echo launched $name" < /dev/null 2>&1 | tail -1
done
