#!/usr/bin/env bash
# VPS side: launch an arbitrary command as a job. Usage: bash pod/lf/launch_raw.sh <pod> <jobname> '<cmd>' [after_job ...]
cd /home/dan/work/lean-format
P=$1; NAME=$2; cmd=$3; shift 3
pre=""; for a in "$@"; do pre="${pre}until [ -f artifacts/lf/$a.done ]; do sleep 30; done; "; done
podrun $P "rm -f artifacts/lf/$NAME.failed; mkdir -p pod/lf/cmds; cat > pod/lf/cmds/$NAME.cmd <<'XEOF'
$pre$cmd
XEOF
setsid nohup bash pod/lf/job.sh $NAME \"\$(cat pod/lf/cmds/$NAME.cmd)\" > /dev/null 2>&1 < /dev/null & disown; sleep 1; echo launched $NAME" < /dev/null 2>&1 | tail -1
