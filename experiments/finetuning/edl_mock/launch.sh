#!/bin/sh
# edl_mock launch (2026-10-10). Run from repo root. Detached, logs to runs/. Uses Dan's worktree read-only.
set -e
cd "$(dirname "$0")"; E=$(pwd); C=$E/../../../code; CK=$HOME/.cache/nd-rl/dan_ckpts; D=$E/data
PY="$C/.venv/bin/python"; : > runs/pids.txt
run() { # name ckpt stream arm lr epochs
  THREADS=2 ND_OFFLINE=1 nohup "$PY" "$C/finetuning/edl_mock.py" --ckpt "$CK/$2" --train "$D/stream_$3_train.jsonl" \
    --held "$D/stream_$3_held.jsonl" --arm "$4" --lr "$5" --epochs "$6" --out "runs/$1.jsonl" > "runs/$1.log" 2>&1 &
  echo "$1 $!" >> runs/pids.txt
}
run full_base_new  stage1_best12_s0_b1200.pt new full 3e-4 2
run full_r2_new    la_T1_best12_s0_r2.pt     new full 3e-4 2
run full_r4_new    la_T1_best12_s0_r4.pt     new full 3e-4 2
run full_r8_new    la_T1_best12_s0_r8.pt     new full 3e-4 2
wait
run full_base_ctrl stage1_best12_s0_b1200.pt ctrl full 3e-4 2
run full_r8_ctrl   la_T1_best12_s0_r8.pt     ctrl full 3e-4 2
run m10k_base_new  stage1_best12_s0_b1200.pt new m10000 3e-3 8
run m10k_r8_new    la_T1_best12_s0_r8.pt     new m10000 3e-3 8
wait
echo "ALL_DONE $(date -u +%FT%TZ)" >> runs/launch.log
