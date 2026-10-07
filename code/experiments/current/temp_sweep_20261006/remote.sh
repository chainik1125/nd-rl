#!/usr/bin/env bash
# Runs on the pod: lays out ~/runs/proof-selection the way Dmitry's pool_lib.py hardcodes it (Dan's repo at the
# commit his sprint pinned, the r16 checkpoint), installs Lean, runs the sweep.
# Usage: bash remote.sh <out-dir> [sweep_remote.py args...]
set -euo pipefail
DAN_COMMIT=4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c   # dan-pandori/nd-takehome, branch dan
R=~/runs/proof-selection
OUT=$1; shift
mkdir -p "$R/ckpts" "$OUT"
[ -d "$R/nd-takehome/.git" ] || git clone -q https://github.com/dan-pandori/nd-takehome.git "$R/nd-takehome"
git -C "$R/nd-takehome" checkout -q "$DAN_COMMIT"
cp -n /workspace/tempsweep/la_T1_best12_s0_r16.pt "$R/ckpts/"
[ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
export PATH=$HOME/.elan/bin:$PATH LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG="$OUT/lean_gate.jsonl"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus $(nproc), $(lean --version)"
# Newer RunPod images put torch under `python` and leave `python3` as bare system Python.
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
cd /workspace/tempsweep
$PY sweep_remote.py --out "$OUT" "$@"
