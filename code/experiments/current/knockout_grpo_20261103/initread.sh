#!/usr/bin/env bash
# Read a line's iteration-0 checkpoint (GRPO's it0 rolling save = the init) through the snapshot read path:
# eval.py at f35d3daf, k 64, batch 512, plain, B64, T 0.8, 512 / 96 / 10 -- line.py's eval_one with seed 0.
K=/scratch/cowifornia/nd-rl/knockout_grpo; tag=$1; gpu=$2
cd $K/snap/f35d3daf && CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  LEAN_GATE_WORKERS=8 ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$K/tmp timeout 2400 nice \
  /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m code.thinking.eval \
  --ckpt $K/runs/${tag}_L2/ckpts/${tag}_L2_it0_rolling.pt --sets $K/sets/strict106.jsonl --out $K/evals/${tag}/init_c0000 \
  --k 64 --budgets 64 --arms plain --batch 512 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10 \
  > $K/logs/initread_${tag}.log 2>&1
echo "initread $tag rc $?" >> $K/logs/initread_${tag}.log
