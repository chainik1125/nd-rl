#!/bin/bash
# score one pools_v2 pool with the teacher-forced scorers (GPU 7). Usage: run_scores.sh <stem>
set -e
S=/data/users/dmitry/ndrl-selection-sprint/code/experiments/current/proof_selection_20261006
R=$HOME/runs/proof-selection; PY=$HOME/venvs/proof-state/bin/python
export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast CUDA_VISIBLE_DEVICES=7
stem=$1
$PY $S/code/score_loglik.py --pool $R/pools_v2/$stem.jsonl --out_dir $R/scores 2>&1 | grep -v Warning | tee -a $R/logs/workerB_score_$stem.log
