#!/usr/bin/env bash
# Runs on the pod. Checks out Dan's fork at the pinned commit (frozen data + verifier), runs the harness.
# Usage: bash remote.sh <out-dir> [harness.py args...]
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
cd /workspace/autoresearch
python3 harness.py --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
