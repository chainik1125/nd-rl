#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES after checking coordinated GPU availability}"
export ND_OFFLINE=1
exec /data/users/dmitry/venvs/proof-state/bin/python \
  code/experiments/current/learned_compute_allocation_20261006/allocation/train_live.py \
  --source artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan \
  --base artifacts/learned_compute_allocation_20261006/allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt \
  --data artifacts/learned_compute_allocation_20261006/data/main_v1/train.jsonl \
  --out artifacts/learned_compute_allocation_20261006/allocation/fixed4/main_s101 \
  --seed 101 --cost 0 --updates 200 --episodes 8 --lr 1e-5 --head-lr 1e-3 \
  --context 16384 --teacher-diagnostics --save-every 50 --fixed-count 4
