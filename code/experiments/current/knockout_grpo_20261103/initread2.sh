#!/usr/bin/env bash
# Init reads on a free GPU (after the co-located attempt ran out of memory): the iteration-0 (GRPO it0 rolling)
# checkpoint through the snapshot read path -- eval.py at f35d3daf, k 64, batch 512, seed 0, plain, B64, T 0.8.
K=/scratch/cowifornia/nd-rl/knockout_grpo; tag=$1; gpu=$2
mkdir -p $K/evals/$tag
cd $K/snap/f35d3daf && CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  LEAN_GATE_WORKERS=8 ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$K/tmp timeout 1800 nice \
  /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m code.thinking.eval \
  --ckpt $K/initckpt/${tag}_L2_it0_rolling.pt --sets $K/sets/strict106.jsonl --out $K/evals/${tag}/init_n64 \
  --k 64 --budgets 64 --arms plain --batch 512 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10 \
  > $K/logs/initread2_${tag}.log 2>&1
echo "initread2 $tag rc $?" >> $K/logs/initread2_${tag}.log
