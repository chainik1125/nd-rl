#!/usr/bin/env bash
# Runs on the pod. Checks out Dan's fork at the pinned commit, unpacks the Stage-1 file, runs the pipeline.
# Usage: bash remote.sh <out-dir> [pipeline.py args...]
set -euo pipefail
FORK_COMMIT=ab629c7d2cdf3dc50d471e302d14643ba0782749   # dan-pandori/nd-takehome, branch dan_ladder_a
export ND_FORK=/workspace/nd-takehome
OUT=$1; shift

if [ ! -d "$ND_FORK/.git" ]; then
  git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
fi
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
[ -f "$ND_FORK/data/train.jsonl" ] || gunzip -k "$ND_FORK/data/train.jsonl.gz"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cd /workspace/poc
python3 pipeline.py --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
