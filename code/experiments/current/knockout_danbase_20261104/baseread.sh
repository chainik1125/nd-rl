#!/usr/bin/env bash
# One-off base read for a line whose segment-1 base read failed: LINE BASE SEED GPU (on this host).
# Same settings as dline.py's reads; refuses a GPU with another user's process.
LINE=$1; BASE=$2; SEED=$3; GPU=$4
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
KD=/scratch/cowifornia/nd-rl/knockout_danbase
cd $W && python3 -c "import gpu,sys; sys.exit(0 if gpu.free_for_us(gpu.HOST,$GPU) else 1)" || { echo "GPU $GPU not ours alone"; exit 2; }
OUT=$KD/runs/$LINE/seg1/reads/it0000.json
[ -f $OUT ] && { echo "exists"; exit 0; }
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=$GPU DANREAD_MEM_FRAC=0.3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
timeout 1800 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python $W/danread.py --ckpt $KD/bases/$BASE \
  --seed $SEED --round 0 --out $OUT --only106 --k 64 --batch 128 >> $KD/runs/$LINE/seg1/reads.log 2>&1
echo "$LINE rc $?"; ls $OUT
