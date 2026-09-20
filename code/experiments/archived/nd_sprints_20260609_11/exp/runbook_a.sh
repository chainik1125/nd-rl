#!/bin/bash
# pod-a runbook: finish cliff trainings, then run all cliff evals.
# Requires: L12 training already running/done on this pod; L8 ckpt arrives
# from pod-b into /root/nd/runs/L8/ckpt.pt (copied externally).
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1

while [ ! -f $R/L12/ckpt.pt ]; do sleep 20; done
echo "L12 done"
python3 -m train.train --train $D/train_L16.jsonl --out $R/L16 --iters 6000 --bs 128 > $R/L16.log 2>&1
echo "L16 done"
python3 -m train.train --train $D/train_L8_2xtrue.jsonl --out $R/L8_2xtrue --iters 6000 --bs 128 > $R/L8_2xtrue.log 2>&1
echo "L8_2xtrue done"
while [ ! -f $R/L8/ckpt.pt ]; do sleep 20; done
echo "L8 ckpt present"

for cond in L8 L12 L16 L8_2xtrue; do
  for ev in eval_iid eval_ood eval_local; do
    python3 -m train.evalgen --ckpt $R/$cond/ckpt.pt --targets $D/$ev.jsonl \
      --k 1 --temp 0.0 --bs 1536 --out $RES/cliff_${cond}_${ev}_greedy.jsonl \
      > $RES/log_${cond}_${ev}_greedy.txt 2>&1
    python3 -m train.evalgen --ckpt $R/$cond/ckpt.pt --targets $D/$ev.jsonl \
      --k 8 --temp 0.8 --bs 1536 --out $RES/cliff_${cond}_${ev}_pass8.jsonl \
      > $RES/log_${cond}_${ev}_pass8.txt 2>&1
    echo "eval $cond $ev done"
  done
done

# Yue-style large-k probe of the round-0 model on the transfer band
python3 -m train.evalgen --ckpt $R/L8/ckpt.pt --targets $D/eval_transfer.jsonl \
  --limit 500 --k 64 --temp 1.0 --bs 2048 \
  --out $RES/base_L8_transfer_pass64.jsonl > $RES/log_base_transfer64.txt 2>&1
echo "ALL CLIFF DONE"
