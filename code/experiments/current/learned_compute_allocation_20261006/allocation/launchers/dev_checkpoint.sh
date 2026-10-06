#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES after checking coordinated GPU availability}"
: "${ALLOCATION_CHECKPOINT:?Set ALLOCATION_CHECKPOINT to the completed-step 100 or 200 checkpoint}"
: "${ALLOCATION_EVAL_OUT:?Set ALLOCATION_EVAL_OUT to a new immutable output folder}"
export ND_OFFLINE=1
exec /data/users/dmitry/venvs/proof-state/bin/python \
  code/experiments/current/learned_compute_allocation_20261006/allocation/evaluate_live_path.py \
  --source artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan \
  --base artifacts/learned_compute_allocation_20261006/allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt \
  --checkpoint "$ALLOCATION_CHECKPOINT" \
  --data artifacts/learned_compute_allocation_20261006/allocation/main_dev32_public_panel.jsonl \
  --out "$ALLOCATION_EVAL_OUT" --seed 52006 --repeats 1 --context 16384
