#!/usr/bin/env bash
# One queue per GPU for GRPO recipe v2: run line.sh for each ARM:SEED in order on this GPU.
#   bash queue.sh GPU "ARM:SEED ARM:SEED" [WAIT_LOG WAIT_LINE [PRE_CMD]]
# WAIT_LOG/WAIT_LINE: wait until the GPU's current job writes its done line first; PRE_CMD runs before the arms.
source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh
g=$1; jobs=$2; wl=$3; wline=$4; pre=$5
if [ -n "$wl" ]; then say "queue GPU $g: waiting for '$wline' in $wl"; waitline $wl "$wline"; fi
if [ -n "$pre" ]; then say "queue GPU $g: pre: $pre"; bash -c "$pre"; fi
for j in $jobs; do
  IFS=: read -r arm s <<< "$j"
  say "queue GPU $g: start $arm seed $s"
  bash $R/grpo_v2/line.sh $g $s $arm > $L/v2_${arm}_s$s.log 2>&1
  say "queue GPU $g: $arm seed $s exit $?: $(tail -1 $L/v2_${arm}_s$s.log | cut -c1-160)"
  python3 $R/backup.py >> $L/backup.log 2>&1 || true  # ynez: mirror the arm's results to NFS
done
say "queue GPU $g done"
