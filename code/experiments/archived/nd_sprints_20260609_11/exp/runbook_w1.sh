#!/bin/bash
# W1 style isolation: SAME theorems, model-style vs generator-style proofs.
# Add each (token-matched, same theorems) to base primitives; retrain depth-4;
# compare in-distribution retention + transfer. Predicts generator-style ->
# iid collapse, model-style -> retained (isolates STYLE; theorems held fixed).
cd /root/nd/code
D=/root/nd/data
R=/root/nd/runs
RES=/root/nd/results
export PYTHONUNBUFFERED=1
mkdir -p $RES

# baseline: base model iid/transfer (reference)
python3 -m train.evalgen --ckpt $R/L8off/ckpt.pt --targets $D/eval_iid.jsonl --k 1 --temp 0.0 --bs 1024 --out $RES/w1_base_iid.jsonl > $RES/w1_base_iid.log 2>&1
python3 -m train.evalgen --ckpt $R/L8off/ckpt.pt --targets $D/eval_transfer.jsonl --limit 500 --k 8 --temp 0.8 --bs 1024 --out $RES/w1_base_transfer.jsonl > $RES/w1_base_tr.log 2>&1

for arm in model gen; do
  python3 -m train.train --train $D/train_L8off.jsonl $D/w1_${arm}style.jsonl:8 \
    --out $R/w1_$arm --iters 6000 --bs 128 --layers 4 --seed 0 > $R/w1_$arm.log 2>&1
  python3 -m train.evalgen --ckpt $R/w1_$arm/ckpt.pt --targets $D/eval_iid.jsonl \
    --k 1 --temp 0.0 --bs 1024 --out $RES/w1_${arm}_iid.jsonl > $RES/w1_${arm}_iid.log 2>&1
  python3 -m train.evalgen --ckpt $R/w1_$arm/ckpt.pt --targets $D/eval_transfer.jsonl \
    --limit 500 --k 8 --temp 0.8 --bs 1024 --out $RES/w1_${arm}_transfer.jsonl > $RES/w1_${arm}_tr.log 2>&1
  echo "W1 $arm done"
done
echo "W1 DONE"
