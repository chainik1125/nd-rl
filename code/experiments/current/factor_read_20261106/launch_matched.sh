#!/usr/bin/env bash
# Amendment 3: the matched-compute control on ynez GPU 0, detached, under `timeout` and the watchdog (STALL / EXIT / OVERRUN).
# fr_run.py is unchanged: it gates on gpu.py before each stage and holds the mkdir lock. Its own ledger and cap: 3,600 s.
# Usage (on ynez): launch_matched.sh EXPECTED_HOURS TAGS
set -u
source /scratch/cowifornia/nd-rl/factor_read/code/env.sh
cd "$FR/code" || exit 1
EXPECTED=${1:-1}
TAGS=$2
M="$FR/matched"
mkdir -p "$M/reads" "$FR/logs"
USED=$("$PY" -c "
import json,os
p='$M/gpu_ledger.jsonl'
print(int(sum(json.loads(x)['secs'] for x in open(p) if x.strip())) if os.path.exists(p) else 0)")
LEFT=$((3600 - USED))
[ "$LEFT" -gt 0 ] || { echo "no GPU budget left ($USED s used)"; exit 4; }
echo "$(date) launching with $LEFT s of the budget left ($USED s used): $TAGS" >> "$FR/logs/matched.log"
nohup setsid timeout "$LEFT" "$PY" fr_run.py --ckpts ckpts_matched.json --refs ../refs/refs.jsonl --out ../matched/reads --ckpt_dir ../matched/ckpts \
  --tags "$TAGS" --stages A,pick,reach,negi,dn,C --gpu_budget 3600 >> "$FR/logs/matched.log" 2>&1 &
PID=$!
echo "$PID" > "$FR/logs/matched.pid"
nohup setsid "$PY" /scratch/cowifornia/nd-rl/think/runs/watchdog.py "factor_read_matched:$PID:0:$FR/logs/matched.log:$EXPECTED" >> "$FR/logs/watchdog_matched.log" 2>&1 &
echo "$!" > "$FR/logs/watchdog_matched.pid"
echo "fr_run under timeout: PID $PID; watchdog: PID $!"
