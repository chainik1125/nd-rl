#!/usr/bin/env bash
# Calibration chain (GPU 0, shared lock; each job via calib_lock.sh, k 8, plain arm, max_steps 96):
#   1. T0 s0 B0 and 2. T1v1 s0 B0+B256 on the 600-item length-stratified sample of the 13-20 set;
#   3. T0 s0 B0 on the length-stratified sample of the 21-30 set (575: the 30-line bin has 35 items).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=/scratch/cowifornia/nd-rl/think/runs/stage1
P13=/scratch/cowifornia/nd-rl/think/runs/pools/test_13_20
P21=/scratch/cowifornia/nd-rl/think/runs/pools/test_21_30
bash "$HERE/calib_lock.sh" T0_s0 "$S/T0_s0/model.pt" 0 "$P13/calib600_13_20.jsonl" "$P13/calib"
bash "$HERE/calib_lock.sh" T1v1_s0 "$S/T1v1_s0/model.pt" 0,256 "$P13/calib600_13_20.jsonl" "$P13/calib"
bash "$HERE/calib_lock.sh" T0_s0_600 "$S/T0_s0/model.pt" 0 "$P21/calib600_21_30.jsonl" "$P21/calib"
