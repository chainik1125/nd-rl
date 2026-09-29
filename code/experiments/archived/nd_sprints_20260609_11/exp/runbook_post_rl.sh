#!/bin/bash
# Post-RL analyses on pod-b. Run after ei_rl completes.
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1

# 1) forgetting check: iid greedy per round ckpt
for rnd in 0 1 2 3 4 5; do
  CK=$R/ei_rl/round$rnd/ckpt.pt
  [ "$rnd" = "0" ] && [ ! -f "$CK" ] && CK=$R/L8/ckpt.pt
  if [ -f "$CK" ]; then
    python3 -m train.evalgen --ckpt $CK --targets $D/eval_iid.jsonl \
      --k 1 --temp 0.0 --bs 1536 --out $RES/ei_rl_round${rnd}_iid_greedy.jsonl \
      > $RES/log_ei_iid_r$rnd.txt 2>&1
    echo "iid eval round $rnd done"
  fi
done

# 2) sharpening check: pass@64 on transfer for round-5 vs round-0
python3 -m train.evalgen --ckpt $R/ei_rl/round5/ckpt.pt --targets $D/eval_transfer.jsonl \
  --limit 500 --k 64 --temp 1.0 --bs 2048 --out $RES/ei_rl_round5_transfer_pass64.jsonl \
  > $RES/log_ei_r5_pass64.txt 2>&1
echo "round5 pass@64 done"

# 3) structure analysis of discoveries
python3 -m exp.analyze_ei --run $R/ei_rl --baseline $D/train_L8.jsonl \
  --out $RES/ei_rl_analysis.json > $RES/log_analyze_ei.txt 2>&1
echo "analyze_ei done"

echo "POST-RL ANALYSES DONE"
