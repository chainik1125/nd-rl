#!/usr/bin/env bash
# Runs on the pod. ../long_rl_20261014/remote.sh with one change: the last line runs knockout.py, the wrapper around
# that experiment's harness.py. Fork commit, Lean install and environment are the same.
# Usage: ND_ARCH=untied ND_RL=ei|grpo ND_KNOCKOUT=IMPE|IMPI|BOTE|DN|none bash remote_ko.sh <out-dir> [harness args...]
set -euo pipefail
FORK_COMMIT=51604b3feaa5e66821d1faba22238fe6d96d46d0   # dan-pandori/nd-takehome, branch dan_lean_seed2
export ND_FORK=/workspace/nd-takehome
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)

if [ ! -d "$ND_FORK/.git" ]; then
  git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
fi
git -C "$ND_FORK" cat-file -e "$FORK_COMMIT" 2>/dev/null || git -C "$ND_FORK" fetch -q origin
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
[ -f "$ND_FORK/data/train.jsonl" ] || gunzip -k "$ND_FORK/data/train.jsonl.gz"
[ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
export LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG="$OUT/lean_gate.jsonl" ND_DATA="$HERE/data"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus $(nproc), $(~/.elan/bin/lean --version), arm $ND_ARCH/$ND_RL, knockout $ND_KNOCKOUT, pool ${ND_POOL:-full}"
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
echo "python $($PY --version 2>&1), torch $($PY -c 'import torch; print(torch.__version__)')"
cd "$HERE"
$PY knockout.py --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
