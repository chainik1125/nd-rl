#!/usr/bin/env bash
# Pod setup: Lean as on Dan's pods, Dan's fork at a pinned commit, the checkpoints from his public bucket.
set -euo pipefail
python -m pip install -q huggingface_hub tqdm numpy pytest
[ -x ~/.elan/bin/lean ] || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.1
~/.elan/bin/lean --version
cd /workspace
[ -d nd-takehome ] || git clone -q --filter=blob:none https://github.com/dan-pandori/nd-takehome
git -C nd-takehome checkout -q 4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c
mkdir -p ckpts results
B=hf://buckets/dan-pandori/nd-rl
get() { [ -s "ckpts/$(basename "$1")" ] || hf buckets cp "$B/$1" "ckpts/$(basename "$1")"; }
for s in 0 1 2; do
  get "trajectory/ckpts/tj/stage1_best12_s${s}_b1200.pt"
  for r in 1 2 3 4 6 8; do get "trajectory/ckpts/tj/ladder/la_T1_best12_s${s}_r${r}.pt"; done
  for r in 12 16; do get "rl-continue/ckpts/rc/ladder/la_T1_best12_s${s}_r${r}.pt"; done
done
if [ "${1:-}" = svd ]; then
  for p in 1600 5000 12000 16000; do
    get "trajectory/ckpts/tj/stage1_best12_s0_b1200_step${p}.pt"
    for r in 1 2 4 8; do get "rl-from-ckpt/ckpts/rfc/ladder/la_T1_best12_s0_p${p}_r${r}.pt"; done
  done
fi
ls ckpts | wc -l
