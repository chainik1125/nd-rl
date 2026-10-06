#!/usr/bin/env bash
# VPS side (resume phase 2026-09-23): launch every job of an arm on a pod, one podrun call per job under setsid/nohup; each job waits for its
# dependencies' done markers on the pod. Jobs whose log already exists are skipped. Usage: bash pod/dsg/launch2.sh <pod> <seed 0|1> [only-job-name]
cd /home/dan/work/ds-generator
P=$1; ARM=$2; ONLY=$3   # ARM is the seed in the resume phase
bash pod/dsg/sync.sh $P push pod/dsg/job.sh > /dev/null < /dev/null
mapfile -t LINES < <(python3 pod/dsg/jobs2.py $ARM)
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r name deps cmd <<< "$line"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  pre=""; IFS=',' read -ra DS <<< "$deps"
  for d in "${DS[@]}"; do [ -n "$d" ] && [ "$d" != "-" ] && pre="${pre}until [ -f artifacts/dsg/$d.done ]; do sleep 30; done; "; done
  podrun $P "[ -f artifacts/dsg/logs/$name.log ] && { echo already $name; exit 0; }; mkdir -p pod/dsg/cmds artifacts/dsg/logs; cat > pod/dsg/cmds/$name.cmd <<'XEOF'
$pre$cmd
XEOF
CUDA_MEM_FRACTION=${CMF:-0.90} LEAN_GATE_WORKERS=${LGW:-7} ND_SAMPLE_COMPACT=${NDC:-0} setsid nohup bash pod/dsg/job.sh $name \"\$(cat pod/dsg/cmds/$name.cmd)\" > /dev/null 2>&1 < /dev/null & disown; sleep 1; echo launched $name" < /dev/null 2>&1 | tail -1
done
