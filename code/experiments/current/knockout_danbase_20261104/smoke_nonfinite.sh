#!/usr/bin/env bash
# The non-finite stop on a real run: an lr large enough to blow up the weights (Amendment 2).
KD=/scratch/cowifornia/nd-rl/knockout_danbase
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
GPU=${1:?gpu}; S=$KD/smoke_guardoff; rm -rf $S/n $S/n.log
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=$GPU \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
cd $KD/snap/f35d3daf_danbase
python3 -c "import sys; sys.path.insert(0,'$W'); import gpu; sys.exit(0 if gpu.free_for_us('ynez',$GPU) else 1)" || { echo "GPU $GPU taken"; exit 2; }
timeout 600 $PY -m code.thinking.grpo --init $KD/bases/stage1_dose01_s2.pt --name n --out $S/n \
  --targets $KD/pools/dan_rl_targets.jsonl --prompts 32 --group 8 --batch 256 \
  --unlikely_beta 0.25 --overlong fail --save_every 6 --save_rolling 1 --keep_rolling 4 \
  --max_steps 48 --max_action 256 --seed 2 --budgets 0 --canary $KD/pools/canary_long.jsonl \
  --canary_n 100 --canary_every 1 --kl_anchor_coef 0 --kl_anchor_ref $KD/bases/stage1_dose01_s2.pt \
  --monitor --arm logical --max_rej 0 --rej_penalty 0.1 --snap_iters 2,4 \
  --iters 4 --lr 1e37 --stop_nonfinite --no_alarm_ckpt > $S/n.log 2>&1
echo "n rc $?"
