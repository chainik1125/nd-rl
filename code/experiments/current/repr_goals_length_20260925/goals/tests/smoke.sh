#!/usr/bin/env bash
# GPU plumbing smoke (through the lock): harness --smoke with rl_goals.py on the 60-step smoke checkpoint, then the
# evaluator on a 3-theorem slice. Not part of the experiment.
set -x
G=/scratch/cowifornia/nd-rl/ab3/goals; PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
cd $G; export AB_WORKER=goals
rm -rf tests/out/rl_smoke
$PY tests/out/frozen_smoke/harness.py --rl $G/rl_goals.py --out tests/out/rl_smoke --seed 101 --smoke --no-score --gpu auto --gpu-wait 3600
$PY evaluate.py --ckpt tests/out/rl_smoke/final.pt --label smoke_rl --limit 3 --lengths 6,11,18 --sets std,matched --gpu auto --gpu-wait 3600 --force
echo SMOKE_DONE
