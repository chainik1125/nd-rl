#!/usr/bin/env bash
# Calibration of the 21-30 test set on GPU 0 only: T0 s0 (no think, B 0) and T1v1 s0 (B 256), k 8, plain arm.
#   bash calib.sh SET.jsonl OUTDIR [K]
# Before each launch GPU 0 must show < 500 MiB used; it is polled every 60 s for up to 60 min (other agents run
# short jobs there).  Every command is wrapped in `timeout`.  GPUs 1-4 are never used.
set -uo pipefail
SET=$1
OUT=$2
K=${3:-8}
REPO=/scratch/cowifornia/nd-rl/think/nd-rl
cd "$REPO"
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean \
    TMPDIR=/scratch/cowifornia/nd-rl/think/runs/tmp CUDA_VISIBLE_DEVICES=0 LEAN_GATE_WORKERS=16

wait_gpu() {
    for i in $(seq 1 60); do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
        if [ "$m" -lt 500 ]; then echo "$(date -u +%H:%M:%S) gpu0 free (${m} MiB)"; return 0; fi
        echo "$(date -u +%H:%M:%S) gpu0 busy (${m} MiB), poll $i/60"
        sleep 60
    done
    echo "gpu0 not free after 60 min"; return 1
}

run() {  # name ckpt budgets
    wait_gpu || return 1
    timeout 7200 code/.venv/bin/python -m code.thinking.eval --ckpt "$2" --sets "$SET" --out "$OUT/$1" \
        --k "$K" --arms plain --budgets "$3" --max_steps 96 > "$OUT/$1.log" 2>&1
    echo "$(date -u +%H:%M:%S) $1 exit $?"; tail -2 "$OUT/$1.log"
}

mkdir -p "$OUT"
run T0_s0 /scratch/cowifornia/nd-rl/think/runs/stage1/T0_s0/model.pt 0
run T1v1_s0 /scratch/cowifornia/nd-rl/think/runs/stage1/T1v1_s0/model.pt 256
