#!/usr/bin/env bash
set -euo pipefail
sprint_gpu="$1"
[[ "$sprint_gpu" == 1 || "$sprint_gpu" == 4 ]]
artifact_root=artifacts/learned_compute_allocation_20261006
trainer=code/experiments/current/learned_compute_allocation_20261006/selection/train_bce.py
for training_seed in 0 1 2; do
    output="$artifact_root/selectors/main_factorized_bce_transitions_s${training_seed}"
    [[ ! -e "$output/config.json" ]]
    CUDA_VISIBLE_DEVICES="$sprint_gpu" ND_OFFLINE=1 \
        /data/users/dmitry/venvs/proof-state/bin/python "$trainer" \
        --train-public "$artifact_root/pools/main_train/public_transitions.jsonl" \
        --train-private "$artifact_root/pools/main_train/private.jsonl" \
        --dev-public "$artifact_root/pools/main_dev/public_transitions.jsonl" \
        --dev-private "$artifact_root/pools/main_dev/private.jsonl" \
        --out "$output" --architecture factorized --view transitions \
        --context 8192 --epochs 5 --batch 2 --lr 0.00003 --seed "$training_seed" \
        > "$artifact_root/main_factorized_bce_s${training_seed}.log" 2>&1
done
