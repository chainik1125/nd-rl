#!/usr/bin/env bash
# The full read on ynez GPU 0, detached, under `timeout` and the watchdog (STALL / EXIT / OVERRUN).
# fr_run.py gates on gpu.py before each stage, holds the mkdir lock, and stops at the GPU budget by its ledger.
# Usage (on ynez): launch_full.sh EXPECTED_HOURS [TAGS]
set -u
source /scratch/cowifornia/nd-rl/factor_read/code/env.sh
cd "$FR/code" || exit 1
EXPECTED=${1:-2}
TAGS=${2:-ko_s0,ko_s1,ko_s2,dose01_s0,dose01_s1,dose01_s2,dose1_s0,dose1_s1,dose1_s2,dko_s0,dko_s1,dko_s2,dd01_s0,dd01_s1,dd01_s2,dd1_s0,dd1_s1,dd1_s2,dd01g0_s2}
USED=$("$PY" -c "
import json,os
p='$FR/gpu_ledger.jsonl'
print(int(sum(json.loads(x)['secs'] for x in open(p) if x.strip())) if os.path.exists(p) else 0)")
LEFT=$((10800 - USED))
[ "$LEFT" -gt 0 ] || { echo "no GPU budget left ($USED s used)"; exit 4; }
echo "$(date) launching with $LEFT s of the budget left ($USED s used)" >> "$FR/logs/full.log"
nohup setsid timeout "$LEFT" "$PY" fr_run.py --ckpts ckpts.json --refs ../refs/refs.jsonl --out ../reads --ckpt_dir ../ckpts \
  --tags "$TAGS" --stages A,pick,reach,negi,dn,C --gpu_budget 10800 >> "$FR/logs/full.log" 2>&1 &
PID=$!
echo "$PID" > "$FR/logs/full.pid"
nohup setsid "$PY" /scratch/cowifornia/nd-rl/think/runs/watchdog.py "factor_read:$PID:0:$FR/logs/full.log:$EXPECTED" >> "$FR/logs/watchdog.log" 2>&1 &
echo "$!" > "$FR/logs/watchdog.pid"
echo "fr_run under timeout: PID $PID; watchdog: PID $!"
