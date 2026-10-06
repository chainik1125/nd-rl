#!/usr/bin/env bash
# VPS side: launch ONE named job from jobs.py on a pod, optionally after done-markers. Usage: bash pod/lf/launch1.sh <pod> <d3|ladder> <rand|seq> <jobname> [after_job ...]
cd /home/dan/work/lean-format
P=$1; ROLE=$2; SCH=$3; NAME=$4; shift 4
cmd=$(python3 pod/lf/jobs.py $ROLE $SCH | awk -F'\t' -v n="$NAME" '$1==n{print $2}')
[ -n "$cmd" ] || { echo "no job $NAME"; exit 1; }
pre=""; for a in "$@"; do pre="${pre}until [ -f artifacts/lf/$a.done ]; do sleep 30; done; "; done
podrun $P "rm -f artifacts/lf/$NAME.failed; mkdir -p pod/lf/cmds; cat > pod/lf/cmds/$NAME.cmd <<'XEOF'
$pre$cmd
XEOF
setsid nohup bash pod/lf/job.sh $NAME \"\$(cat pod/lf/cmds/$NAME.cmd)\" > /dev/null 2>&1 < /dev/null & disown; sleep 1; echo launched $NAME" < /dev/null 2>&1 | tail -1
