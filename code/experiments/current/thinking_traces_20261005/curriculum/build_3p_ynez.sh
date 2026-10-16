#!/usr/bin/env bash
# ynez rebuild (2026-10-13): 3-premise versions of the L2r/L3r pool rows, uncurry3.py's docstring commands
# (leon/thinking-curriculum 6bdbd78; run from leon/curriculum-rebuild-ynez, uncurry3.py unchanged), plus
# PYTHONHASHSEED=0 and the L2r/L3r build's time limits (PFG_SEARCH_SECS=12, PFG_NEC_SECS=14.5).
set -uo pipefail
cd /scratch/cowifornia/nd-rl/think/nd-rl-currb || exit 1
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PFG_MEM_GB=3.0 PYTHONHASHSEED=0
export PFG_SEARCH_SECS=12 PFG_NEC_SECS=14.5
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=code.experiments.current.thinking_traces_20261005
O=/scratch/cowifornia/nd-rl/think/runs/pools/curriculum_3p
for b in L2 L3; do
  PFG_MIN_L=6 PFG_MAX_L=15 PFG_MAX_GAP=9 timeout 7200 nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O \
      build --band $b --workers 12 || exit 1
  LEAN_GATE_WORKERS=12 timeout 7200 nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O lean --band $b || exit 1
done
timeout 7200 nice -n 10 $PY -m $X.curriculum.uncurry3 --out $O finish --workers 12 || exit 1
(cd ../nd-rl-v2 && timeout 1800 $PY -m $X.grpo_v2.preflight $O/L2_3p.jsonl $O/L3_3p.jsonl) > $O/preflight.log
timeout 600 $PY -m $X.curriculum.uncurry3 --out $O report
echo BUILD_3P_DONE
