#!/usr/bin/env bash
# Runs on the pod. Checks out Dan's fork at the pinned commit (frozen data + verifier + Lean
# tokenizer/gate), installs Lean, runs the harness.
# Usage: bash remote.sh <out-dir> [harness.py args...]
set -euo pipefail
# robbiethompson18/nd-takehome, branch robbie_seed2-prefilter: dan_lean_seed2's 51604b3 (lean_seq with the BOTE
# fix) plus Dan's sound Lean pre-filter, `lean -j 1` and quota-sized gate workers (verdicts unchanged)
FORK_COMMIT=0e920a55cee2c97d8ce7963b6d03877da3381a69
FORK_URL=https://github.com/robbiethompson18/nd-takehome.git
export ND_FORK=/workspace/nd-takehome
OUT=$1; shift

if [ ! -d "$ND_FORK/.git" ]; then
  git clone -q "$FORK_URL" "$ND_FORK"
fi
git -C "$ND_FORK" cat-file -e "$FORK_COMMIT" 2>/dev/null || git -C "$ND_FORK" fetch -q "$FORK_URL" robbie_seed2-prefilter
git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
[ -f "$ND_FORK/data/train.jsonl" ] || gunzip -k "$ND_FORK/data/train.jsonl.gz"
# Lean 4 core checks every sampled proof (fork lean_gate.py, called from sample.generate): the
# reward is Lean AND nd_verify. Installed before harness.py starts, so it is not charged to TOTAL_SECS.
[ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
# Gate workers default to the cgroup CPU quota, one `lean -j 1` each (fork lean_gate.cpu_quota). Before: one
# default-threaded Lean per visible core (nproc 96), which Dan measured 2-3x slower on RunPod A40s.
export LEAN_PREFILTER=on LEAN_GATE_LOG="$OUT/lean_gate.jsonl"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus $(nproc), quota $(cd "$ND_FORK" && python3 -c 'import lean_gate; print(lean_gate.WORKERS)' 2>/dev/null), prefilter $LEAN_PREFILTER, $(~/.elan/bin/lean --version)"
# Newer RunPod images put torch under `python` and leave `python3` as bare system Python.
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
echo "python $($PY --version 2>&1), torch $($PY -c 'import torch; print(torch.__version__)')"
cd /workspace/autoresearch
$PY harness.py --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
