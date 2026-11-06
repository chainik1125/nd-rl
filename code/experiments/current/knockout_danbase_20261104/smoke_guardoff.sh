#!/usr/bin/env bash
# Short runs on one free ynez GPU that test the guard-off flags before the pilot (Amendment 2):
#   a  guard off, alarm checkpoints written; every iteration alarms (--canary_mult 0.5)
#   b  the same with --no_alarm_ckpt --stop_nonfinite: same rows as a = storage only
#   c  the same as a again: the run-to-run noise, if a and b differ
#   n  --stop_nonfinite with an absurd lr: exit code 4, row['nonfinite_stop'], no checkpoint
#   g  the old flags (rollback on) with the real thresholds: same rows as o until the first rollback
#   o  guard off with the real thresholds
set -u
KD=/scratch/cowifornia/nd-rl/knockout_danbase
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
GPU=${1:?gpu}; S=$KD/smoke_guardoff; rm -rf $S; mkdir -p $S
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=$GPU \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
cd $KD/snap/f35d3daf_danbase
free() { python3 -c "import sys; sys.path.insert(0,'$W'); import gpu; sys.exit(0 if gpu.free_for_us('ynez',$GPU) else 1)"; }
run() {
  name=$1; shift
  free || { echo "$name: another user's process on GPU $GPU; stop"; exit 2; }
  timeout 900 $PY -m code.thinking.grpo --init $KD/bases/stage1_dose01_s2.pt --name $name --out $S/$name \
    --targets $KD/pools/dan_rl_targets.jsonl --prompts 32 --group 8 --batch 256 \
    --unlikely_beta 0.25 --overlong fail --save_every 6 --save_rolling 2 --keep_rolling 4 \
    --max_steps 48 --max_action 256 --seed 2 --budgets 0 --canary $KD/pools/canary_long.jsonl \
    --canary_n 100 --canary_every 1 --kl_anchor_coef 0 --kl_anchor_ref $KD/bases/stage1_dose01_s2.pt \
    --monitor --arm logical --max_rej 0 --rej_penalty 0.1 --snap_iters 2,4 "$@" > $S/$name.log 2>&1
  echo "$name rc $? at $(date -u +%H:%M:%SZ)"
}
run a --iters 4 --lr 1.5e-4 --canary_mult 0.5
run b --iters 4 --lr 1.5e-4 --canary_mult 0.5 --no_alarm_ckpt --stop_nonfinite
run c --iters 4 --lr 1.5e-4 --canary_mult 0.5
run n --iters 4 --lr 1e40 --stop_nonfinite --no_alarm_ckpt
run o --iters 6 --lr 1.5e-4 --no_alarm_ckpt --stop_nonfinite
run g --iters 6 --lr 1.5e-4 --rollback_on_alarm --max_rollbacks 3
echo finished
