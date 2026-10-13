#!/bin/bash
# all.sh: dry-run every GRPO v2 arm, seeds 0-2, normal and SMOKE=1, plus the refusals and queue.sh; results under
# $H/res/. No python, no GPU (make.sh's stubs). Then check.py compares the records with the transcript evidence.
#   bash make.sh [DIR] && bash all.sh && python3 check.py
H=${DRY_HOME:-/scratch/cowifornia/nd-rl/think/dry}  # outputs and the built harness
D=$(cd "$(dirname "$0")" && pwd)  # this script's directory (run.sh)
BASE=(grpo_think grpo_think_v grpo_think_vs grpo_nothink grpo_nothink_v grpo_nothink_vs grpo_nothink_vs3
      ei_think ei_think_v ei_nothink ei_nothink_v)
HARMS=(grpo_think_h grpo_think_v_h grpo_think_vs_h grpo_nothink_h grpo_nothink_v_h grpo_nothink_vs_h grpo_nothink_vs3_h)
GRPO=(grpo_think grpo_think_v grpo_think_vs grpo_nothink grpo_nothink_v grpo_nothink_vs grpo_nothink_vs3 "${HARMS[@]}")
EIH=(ei_think_h ei_think_v_h ei_nothink_h ei_nothink_v_h)
EIB=(ei_think_b ei_think_v_b ei_nothink_b ei_nothink_v_b ei_think_h_b)
MODES=(normal smoke); SEEDS=(0 1 2)
OKLOG='guard it 39: kl anchor 0.0004 x 0.02 (gn policy 0.014); lr x1 (0 rollbacks); capped B0 0.0% B64 0.0% B256 0.0%'
RBLOG='guard it 39: kl anchor 0.0006 x 0.02 (gn policy 0.013); lr x0.5 (1 rollbacks); capped B0 0.0% B64 0.0% B256 0.0%'

setup() {  # NAME MODE SEED SRC CKPT(yes|no) GUARDLINE|none -> a SETUP file: SRC's stub L2 checkpoint and L2 log
  local f=$H/setups/$1.sh c=runs/grpo_v2 l=runs/logs it=39
  [ $2 = smoke ] && { c=runs/grpo_v2/smoke; l=$c/logs; it=2; }
  mkdir -p $H/setups
  { [ $5 = yes ] && echo "mkdir -p \"\$SB/$c/${4}_s${3}_L2/ckpts\" && : > \"\$SB/$c/${4}_s${3}_L2/ckpts/${4}_s${3}_L2_it$it.pt\""
    [ "$6" != none ] && printf 'mkdir -p "$SB/%s" && printf "%%s\\n" "it 39: dry" %q >> "$SB/%s/v2_%s_s%s_L2.log"\n' "$l" "$6" "$l" "$4" "$3"
    echo true; } > $f
  echo $f; }

rm -rf "$H/res" "$H/setups"; t0=$(date +%s); n=0
for mode in "${MODES[@]}"; do for seed in "${SEEDS[@]}"; do
  for arm in "${BASE[@]}" "${HARMS[@]}" "${EIH[@]}" "${EIB[@]}" bogus_arm; do
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm"; n=$((n+1))
  done
  for arm in "${GRPO[@]}"; do
    su=$(setup b_${mode}_s${seed}_$arm $mode $seed $arm yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ${arm}_b "$H/res/$mode/s$seed/${arm}_b" $su; n=$((n+1))
  done
  # ARM_b refusals: no L2 checkpoint; an L2 log at lr x0.5 (1 rollbacks); no L2 log
  su=$(setup nockpt_${mode}_s$seed $mode $seed grpo_think_h no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_h_b "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_h_b" $su
  su=$(setup rb_${mode}_s$seed $mode $seed grpo_think yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_b "$H/res/refuse/$mode/s$seed/rb_grpo_think_b" $su
  su=$(setup nolog_${mode}_s$seed $mode $seed grpo_think_vs yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_b "$H/res/refuse/$mode/s$seed/nolog_grpo_think_vs_b" $su
  n=$((n+3))
done; done
# seed 3 has no Stage-1 init in the sandbox; usage error
bash "$D/run.sh" "$H/line.sh.dry" normal 3 grpo_think "$H/res/refuse/normal/s3/grpo_think"
bash "$D/run.sh" "$H/line.sh.dry" normal "" "" "$H/res/refuse/normal/usage"
# the live run's GPU: grpo_think_h_b seed 1 on GPU 1 (ps evidence), normal
su=$(setup b_gpu1 normal 1 grpo_think_h yes "$OKLOG")
DRY_GPU=1 bash "$D/run.sh" "$H/line.sh.dry" normal 1 grpo_think_h_b "$H/res/gpu1/grpo_think_h_b_s1" $su
# waitfree with a busy GPU (another user's process): the stubbed nvidia-smi reports 23474 MiB and a pid
DRY_GPU_MEM=23474 DRY_GPU_PID=2116814 bash "$D/run.sh" "$H/line.sh.dry" normal 0 grpo_think "$H/res/busy/grpo_think_s0"
# queue.sh: two arms on GPU 2 after a wait line and a PRE_CMD, as the serrano queues were started
cat > $H/setups/queue.sh <<'EOF'
mkdir -p "$SB/runs/logs"; echo "00:00Z grpo_v2 line done (grpo_think seed 0)" > "$SB/runs/logs/v2_grpo_think_s0.log"
EOF
DRY_GPU=2 bash "$D/run.sh" queue normal - - "$H/res/queue/normal" $H/setups/queue.sh -- \
  2 "grpo_think_v:1 ei_nothink:2" "$H/sb/runs/logs/v2_grpo_think_s0.log" 'grpo_v2 line done (grpo_think seed 0)' 'echo pre ran'
n=$((n+7))
echo "$n dry runs in $(( $(date +%s) - t0 ))s under $H/res"
