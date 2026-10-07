#!/usr/bin/env bash
# Runs on the pod, from /workspace/autoresearch (the frozen harness.py and its remote.sh, this
# folder's pretrain_looped.py as pretrain.py, loop_sweep.py, textbook72/).
# Usage: bash spike_remote.sh <out-dir> train|baseline|more|sweep [harness.py args...]
#   train     pretrain + RL through the frozen harness (ND_TRAIN_LOOPS picks the arm), then the
#             loop sweep on pretrain.pt and final.pt
#   baseline  download run 198 seed 0 from Hugging Face and run the same sweep on it
#   sweep     only the loop sweep, on a run that already exists in <out-dir>; `bar` is the same
#             after remote.sh's setup, for a fresh pod given an uploaded run
#   more      8 more expert-iteration rounds from <out-dir>/final.pt (more_rl.py), then the sweep
#             on the result
set -euo pipefail
OUT=$1 MODE=$2; shift 2
HF=https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs
export ND_FORK=/workspace/nd-takehome
cd /workspace/autoresearch
if [ "$MODE" = train ]; then
  bash remote.sh "$OUT" "$@"   # fork checkout, Lean install, harness.py
elif [ "$MODE" = baseline ] || [ "$MODE" = bar ]; then
  # remote.sh's setup without its harness run
  FORK_COMMIT=$(sed -n 's/^FORK_COMMIT=\([0-9a-f]*\).*/\1/p' remote.sh)
  [ -d "$ND_FORK/.git" ] || git clone -q https://github.com/dan-pandori/nd-takehome.git "$ND_FORK"
  git -C "$ND_FORK" checkout -q "$FORK_COMMIT"
  [ -f "$ND_FORK/data/train.jsonl" ] || gunzip -k "$ND_FORK/data/train.jsonl.gz"
  [ -x ~/.elan/bin/lean ] || curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0 > /dev/null
  mkdir -p "$OUT/source"
  for f in source/pretrain.py pretrain.pt final.pt found_targets.jsonl; do
    [ "$MODE" = bar ] || [ -f "$OUT/$f" ] || curl -sSfL -o "$OUT/$f" "$HF/198-claude_lean-muon-no-rms-retest_s0/$f"
  done
fi
export LEAN_GATE_WORKERS=$(nproc) LEAN_GATE_LOG="$OUT/lean_gate_sweep.jsonl"
PY=$(for p in python python3; do "$p" -c "import torch" 2>/dev/null && echo "$p" && break; done)
if [ "$MODE" = more ]; then
  $PY more_rl.py "$OUT" --rounds 8
  OUT=$OUT/more_rl
fi
# SWEEP_STAGES / SWEEP_LOOPS override which checkpoints and loop counts are swept
for stage in ${SWEEP_STAGES:-pretrain final}; do
  if [ -f "$OUT/$stage.pt" ] && [ "$MODE$stage" != morepretrain ]; then
    $PY loop_sweep.py "$OUT" --stage $stage ${SWEEP_LOOPS:+--loops $SWEEP_LOOPS} ${SWEEP_ARGS:-}
  fi
done
