#!/usr/bin/env bash
# Pod setup: Dan's fork at long_rl's pinned commit (model, sampler, nd2lean), the long-RL checkpoints and run summaries
# from the HF repo, laid out as CL_CKPTS/<run>/<stem>.pt. No Lean: nothing here judges a proof.
set -euo pipefail
python -m pip install -q huggingface_hub numpy
cd /workspace
[ -d nd-takehome ] || git clone -q --filter=blob:none https://github.com/dan-pandori/nd-takehome
git -C nd-takehome checkout -q 51604b3feaa5e66821d1faba22238fe6d96d46d0
mkdir -p ckpts art
REPO=robbiethompson2018/nd-rl-checkpoints
for run in v4_untied-ei_s0 v4_untied-ei_s1 v5_untied-ei_s0 v5_untied-ei_s1 v2_untied-ei_s0 v2_untied-ei_s1; do
  hf download "$REPO" --include "runs/long-rl-20261014-$run/*.pt" --include "runs/long-rl-20261014-$run/summary.json" --local-dir /workspace/hf > /dev/null
  mkdir -p "ckpts/$run"
  for f in /workspace/hf/runs/long-rl-20261014-$run/*.pt /workspace/hf/runs/long-rl-20261014-$run/summary.json; do ln -sfn "$f" "ckpts/$run/$(basename "$f")"; done
done
ls ckpts/*/ | head -60
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__)"
