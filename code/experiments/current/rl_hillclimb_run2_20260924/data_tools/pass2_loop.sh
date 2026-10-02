#!/bin/bash
# generator pass 2 (bound 16, 120 s): rolling snapshots of the pass-1 output, total wall clock capped at $1 seconds
cd /scratch/cowifornia/nd-rl/ab2/data
T0=$(date +%s); CAP=$1
while true; do
  EL=$(( $(date +%s) - T0 )); REM=$(( CAP - EL )); [ $REM -le 60 ] && break
  if pgrep -f "out labels/gen_b10.jsonl" >/dev/null; then P=20; P1=1; else P=40; P1=0; fi
  python3 tools/pass2_input.py
  N=$(wc -l < raw/gen_u.jsonl)
  if [ "$N" -eq 0 ]; then [ $P1 -eq 0 ] && break; sleep 60; continue; fi
  # while pass 1 still runs, label at most 2000 per round so the next round picks up newly exhausted items and 40 procs
  LIM=""; [ $P1 -eq 1 ] && LIM="--limit 2000"
  python3 tools/label.py --in raw/gen_u.jsonl --out labels/gen_b16.jsonl --bound 16 --time 120 --procs $P --deadline $REM --resume $LIM
done
echo "PASS2 LOOP END $(( $(date +%s) - T0 ))s"
