#!/usr/bin/env bash
# SVD sweep, most wanted first. Each script skips results that already exist. Main convention: 1-D parameters move to
# round r (`--others full`), which reproduces Charles's Figure 4 at r16; held at the pretrain they break the output grammar.
cd /workspace/job
export ND_FORK=/workspace/nd-takehome ND_CKPTS=/workspace/ckpts ND_RESULTS=/workspace/results
NS="0 1 2 4 8 16 32 64 128"
python run_svd.py --seed 0 --rounds 16 8 4 2 1 --ns $NS --others full
python run_svd.py --seed 0 --rounds 12 6 3 --ns $NS --others full
for s in 1 2; do python run_svd.py --seed $s --rounds 16 8 4 2 1 --ns 1 4 16 64 --others full; done
echo QUEUE_DONE
