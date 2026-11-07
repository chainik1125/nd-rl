#!/usr/bin/env bash
# numbers -> figures -> the summary's generated sections -> figures copied to the summary folder
set -e
R=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo
W=$R/code/experiments/current/knockout_danbase_20261104
S=$R/experiment-summaries/2026-11-04-grpo-on-dan-knockout-base
cd $R
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python $W/tables.py > /dev/null
/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python $W/pilot.py > /dev/null
code/.venv/bin/python -m code.experiments.current.knockout_danbase_20261104.fig 2>&1 | grep -v "Warning\|savefig" || true
python3 $W/summary.py
for f in rl_needed_danbase cum_vs_x sigma_danbase; do cp $W/figs/$f.png $W/figs/$f.pdf $S/; done
echo "finalized $(date -u +%H:%M:%SZ)"
