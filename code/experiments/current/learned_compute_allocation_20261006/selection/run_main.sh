#!/usr/bin/env bash
set -euo pipefail
# Invoke only after main_train merge/render/audit and GPU occupancy checks.
# Arguments: assigned GPU (1 or 4), architecture (pooled or factorized).
sprint_gpu="$1"
architecture="$2"
[[ "$sprint_gpu" == 1 || "$sprint_gpu" == 4 ]]
[[ "$architecture" == pooled || "$architecture" == factorized ]]
artifact_root=artifacts/learned_compute_allocation_20261006
trainer=code/experiments/current/learned_compute_allocation_20261006/selection/train.py
for training_seed in 0 1 2; do
    output="$artifact_root/selectors/main_${architecture}_transitions_s${training_seed}"
    [[ ! -e "$output/config.json" ]]
    CUDA_VISIBLE_DEVICES="$sprint_gpu" ND_OFFLINE=1 \
        /data/users/dmitry/venvs/proof-state/bin/python "$trainer" \
        --train-public "$artifact_root/pools/main_train/public_transitions.jsonl" \
        --train-private "$artifact_root/pools/main_train/private.jsonl" \
        --dev-public "$artifact_root/pools/main_dev/public_transitions.jsonl" \
        --dev-private "$artifact_root/pools/main_dev/private.jsonl" \
        --out "$output" --architecture "$architecture" --view transitions \
        --context 8192 --epochs 5 --batch 2 --lr 0.00003 --seed "$training_seed" \
        > "$artifact_root/main_${architecture}_s${training_seed}.log" 2>&1
done
