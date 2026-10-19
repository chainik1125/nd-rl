#!/usr/bin/env bash
# Big-statement twins of the mini-run subsets and of the L2 / L3 test_1k splits (size_twins.py), CPU only.
#   bash build_size_twins.sh            # build every set, then Lean / env, finish, check, report
# SZ_WORKERS (default 7: with the mostly idle parent, at most 8 cores busy) workers at nice 10, each capped at PFG_MEM_GB of address space; every step
# under `timeout`. `build` resumes (rows already in work/gated.jsonl or rejects.jsonl are skipped), so
# a rerun after a crash continues where it stopped; logs are appended.
set -uo pipefail
WT=/scratch/cowifornia/nd-rl/think/nd-rl-size
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
M=code.experiments.current.thinking_traces_20261005.curriculum.size_twins
O=/scratch/cowifornia/nd-rl/think/runs/pools/size_twins
W=${SZ_WORKERS:-7}
cd $WT || exit 1
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= PFG_MEM_GB=1.5
mkdir -p $O/logs
say() { echo "$(date -u +%FT%TZ) $*"; }
for s in L2_mini L3_mini L3_3p_mini L2_test_1k L3_test_1k; do   # L3_3p_mini needs L3_mini's sigmas
  timeout 14400 nice -n 10 $PY -m $M --out $O build --set $s --workers $W >> $O/logs/build_$s.log 2>&1
  say "build $s exit $?"
done
LEAN_GATE_WORKERS=6 timeout 7200 nice -n 10 $PY -m $M --out $O lean >> $O/logs/lean.log 2>&1
say "lean exit $?"
timeout 7200 nice -n 10 $PY -m $M --out $O finish --workers $W >> $O/logs/finish.log 2>&1
say "finish exit $?"
timeout 3600 nice -n 10 $PY -m $M --out $O check >> $O/logs/check.log 2>&1
say "check exit $?"
timeout 3600 nice -n 10 $PY -m $M --out $O report >> $O/logs/report.log 2>&1
say "report exit $?"
say BUILD_SIZE_TWINS_DONE
