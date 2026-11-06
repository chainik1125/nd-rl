#!/usr/bin/env bash
# Runs on the pod: ../remote.sh's setup (the fork at the pinned commit, Lean for the accepted/rejected split), then
# tempkl.py on each line of <jobs> ("<HF url> <tag> <arch> <pretrain HF url> <pretrain tag>", from tempkl/launch.py).
# Checkpoints are cached under ckpt/ (a run's pretrain.pt is the reference for every one of its checkpoints). A
# failure doesn't stop the rest.
set -euo pipefail
FORK_COMMIT=51604b3feaa5e66821d1faba22238fe6d96d46d0   # same as ../remote.sh
export ND_FORK=/workspace/nd-takehome ND_RL=ei ND_DATA=/workspace/long_rl/data
cd /workspace/long_rl
[ -d "$ND_FORK/.git" ] || git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
git -C "$ND_FORK" cat-file -e "$FORK_COMMIT" 2>/dev/null || git -C "$ND_FORK" fetch -q origin
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
[ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
export LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG=/workspace/long_rl/tempkl/lean_gate.jsonl
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus $(nproc), $(~/.elan/bin/lean --version), python $($PY --version 2>&1), torch $($PY -c 'import torch; print(torch.__version__)')"
mkdir -p ckpt tempkl/results
while read -r url tag arch pre_url pre_tag; do
  [ -f "tempkl/results/$tag.json" ] && continue
  { [ -f "ckpt/$pre_tag.pt" ] || curl -sSfL -o "ckpt/$pre_tag.pt" "$pre_url"; } \
    && { [ -f "ckpt/$tag.pt" ] || curl -sSfL -o "ckpt/$tag.pt" "$url"; } \
    && ND_ARCH=$arch $PY tempkl/tempkl.py "ckpt/$tag.pt" --pre "ckpt/$pre_tag.pt" --tag "$tag" \
         --out "tempkl/results/$tag.json" < /dev/null \
    || echo "FAILED $tag"
done < "$1"
