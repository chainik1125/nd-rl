#!/bin/bash
# Post-chain-2 analyses (run on pod-c after ei_rl_off completes).
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1

while ! grep -q '"round": 5' $R/ei_rl_off/history.json 2>/dev/null; do sleep 30; done
echo "chain-2 complete"

python3 -m train.evalgen --ckpt $R/L8off/ckpt.pt --targets $D/eval_transfer.jsonl \
  --limit 500 --k 64 --temp 1.0 --bs 2048 --out $RES/L8off_transfer_pass64.jsonl \
  > $RES/log_L8off_pass64.txt 2>&1
echo "round0-off pass@64 done"

python3 -m train.evalgen --ckpt $R/ei_rl_off/round5/ckpt.pt --targets $D/eval_transfer.jsonl \
  --limit 500 --k 64 --temp 1.0 --bs 2048 --out $RES/ei_off_round5_transfer_pass64.jsonl \
  > $RES/log_ei_off_r5_pass64.txt 2>&1
echo "round5-off pass@64 done"

python3 -m exp.analyze_ei --run $R/ei_rl_off --baseline $D/train_L8off.jsonl \
  --out $RES/ei_rl_off_analysis.json > $RES/log_analyze_ei_off.txt 2>&1
echo "POST-C DONE"
