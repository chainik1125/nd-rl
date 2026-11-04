#!/usr/bin/env bash
# Pilot: 10 GRPO iterations on Dan's full K12 base (stage1_full_s3) with his rl_targets; one GPU, gated.
GPU=$1; KD=/scratch/cowifornia/nd-rl/knockout_danbase
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
U=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print tolower($2)}')
for p in $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v b=$U 'tolower($2)==b{print $1}'); do
  [ "$(ps -o user:32= -p $p)" != "$USER" ] && { echo "GPU $GPU has another user's process $p: not launching"; exit 2; }
done
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=$GPU
export LEAN_GATE_DUMP=$KD/runs/pilot_full_s3/lean_dump.jsonl
mkdir -p $KD/runs/pilot_full_s3
cd $KD/snap/f35d3daf_danbase
exec timeout 5400 nice -n 5 $PY -m code.thinking.grpo --init $KD/bases/stage1_full_s3.pt --name pilot_full_s3 \
  --out $KD/runs/pilot_full_s3 --targets $KD/pools/dan_rl_targets.jsonl --iters 10 --prompts 256 --group 16 \
  --batch 1024 --lr 1.5e-4 --unlikely_beta 0.25 --overlong fail --save_every 10 --save_rolling 5 --keep_rolling 4 \
  --max_steps 48 --max_action 256 --seed 3 --budgets 0 --canary $KD/pools/canary_long.jsonl --canary_n 1000 \
  --canary_every 1 --rollback_on_alarm --max_rollbacks 3 --kl_anchor_coef 0 --kl_anchor_ref $KD/bases/stage1_full_s3.pt \
  --monitor --arm logical --max_rej 0 --rej_penalty 0.1 --snap_iters 5,10
