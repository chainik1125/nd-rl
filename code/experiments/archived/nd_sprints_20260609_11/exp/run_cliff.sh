#!/bin/bash
# SFT length-generalization cliff: train 4 conditions, evaluate each on
# iid + ood eval sets (greedy and pass@8). Run from /root/nd/code on the pod.
set -e
cd "$(dirname "$0")/.."
D=../data
R=../runs

for cond in L8 L12 L16 L8_2x; do
  if [ ! -f $R/$cond/ckpt.pt ]; then
    python3 -m train.train --train $D/train_$cond.jsonl --out $R/$cond \
      --iters ${ITERS:-6000} --bs 128 2>&1 | tail -3
  fi
  for ev in eval_iid eval_ood; do
    python3 -m train.evalgen --ckpt $R/$cond/ckpt.pt --targets $D/$ev.jsonl \
      --k 1 --temp 0.0 --out ../results/cliff_${cond}_${ev}_greedy.jsonl
    python3 -m train.evalgen --ckpt $R/$cond/ckpt.pt --targets $D/$ev.jsonl \
      --k 8 --temp 0.8 --out ../results/cliff_${cond}_${ev}_pass8.jsonl
  done
done
echo CLIFF DONE
