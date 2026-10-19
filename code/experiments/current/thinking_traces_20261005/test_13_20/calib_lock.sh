#!/usr/bin/env bash
# One calibration eval on GPU 0 under the shared lock protocol.
#   bash calib_lock.sh NAME CKPT BUDGETS SET OUTDIR [K]
# Before the job: wait (poll every 60 s, at most --max 600 polls = 10 h) until /scratch/.../runs/gpu0.lock does not
# exist AND GPU 0 memory.used < 500 MiB, then create the lock atomically with this script's PID; the lock is
# removed when the job ends, also on failure or interruption (trap).  The eval runs under `timeout 3600`, plain arm,
# --max_steps 96, CUDA_VISIBLE_DEVICES=0 only.  GPUs 1-4 are never used.
set -uo pipefail
NAME=$1
CKPT=$2
BUDGETS=$3
SET=$4
OUT=$5
K=${6:-8}
LOCK=/scratch/cowifornia/nd-rl/think/runs/gpu0.lock
REPO=/scratch/cowifornia/nd-rl/think/nd-rl
cd "$REPO"
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean \
    TMPDIR=/scratch/cowifornia/nd-rl/think/runs/tmp CUDA_VISIBLE_DEVICES=0 LEAN_GATE_WORKERS=16

release() {
    if [ -e "$LOCK" ] && [ "$(cat "$LOCK" 2>/dev/null)" = "$$" ]; then rm -f "$LOCK"; fi
}

acquire() {
    for i in $(seq 1 600); do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
        if [ ! -e "$LOCK" ] && [ "$m" -lt 500 ]; then
            if (set -o noclobber; echo "$$" > "$LOCK") 2>/dev/null; then
                echo "$(date -u +%H:%M:%S) lock taken (pid $$, gpu0 ${m} MiB)"
                return 0
            fi
        fi
        if [ $((i % 10)) -eq 1 ]; then
            echo "$(date -u +%H:%M:%S) waiting: lock $([ -e "$LOCK" ] && cat "$LOCK" || echo none), gpu0 ${m} MiB"
        fi
        sleep 60
    done
    echo "gave up after 10 h"
    return 1
}

mkdir -p "$OUT"
acquire || exit 1
trap release EXIT INT TERM
timeout 3600 code/.venv/bin/python -m code.thinking.eval --ckpt "$CKPT" --sets "$SET" --out "$OUT/$NAME" \
    --k "$K" --arms plain --budgets "$BUDGETS" --max_steps 96 > "$OUT/$NAME.log" 2>&1
rc=$?
release
echo "$(date -u +%H:%M:%S) $NAME exit $rc"
grep -E '^\[plain' "$OUT/$NAME.log"
exit $rc
