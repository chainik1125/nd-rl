#!/usr/bin/env bash
# Runs on the pod, from /workspace/autoresearch (the frozen harness.py and its remote.sh, this
# folder's pretrain_coconut.py as pretrain.py, thought_sweep.py, textbook72/).
# Usage: ND_ARM=<arm> bash coconut_remote.sh <out-dir> [harness.py args...]
# Pretrain + RL through the frozen harness, then the thought sweep on pretrain.pt and final.pt.
# FINISH=1: <out-dir> holds an uploaded run that timed out before its dev eval. Skip the harness,
# run finish_eval.py on it, then the sweep on the final.pt that writes.
set -euo pipefail
OUT=$1; shift
cd /workspace/autoresearch
[ -n "${FINISH:-}" ] || bash remote.sh "$OUT" "$@"   # fork checkout, Lean install, harness.py
export ND_FORK=/workspace/nd-takehome LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG="$OUT/lean_gate_sweep.jsonl"
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
if [ -n "${FINISH:-}" ]; then
  $PY finish_eval.py "$OUT"
  rm -f "$OUT/pretrain.pt"   # its sweep is done already
fi
for stage in pretrain final; do
  if [ -f "$OUT/$stage.pt" ]; then
    $PY thought_sweep.py "$OUT" --stage $stage
  fi
done
