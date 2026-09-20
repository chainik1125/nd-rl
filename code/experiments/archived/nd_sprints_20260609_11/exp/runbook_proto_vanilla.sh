#!/bin/bash
# Protocol experiment, vanilla-EI control arm (run after data+ckpt arrive).
cd /root/nd/code
export PYTHONUNBUFFERED=1
python3 -m exp.expert_iter --mode rl --rounds 5 --data /root/nd/data \
  --out /root/nd/runs/ei_rl --round0-ckpt /root/nd/runs/L8off_x/ckpt.pt \
  --primitives train_L8off.jsonl --iters 6000 --n-rollouts 32 \
  --n-targets 1200 --rep 8 --seed 0 > /root/nd/runs/ei_rl.log 2>&1
echo "VANILLA ARM DONE"
