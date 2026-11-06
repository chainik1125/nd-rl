#!/usr/bin/env bash
# SAE arms: the wishlist x axis (rounds) at block 4, then a second block and more seeds at r16.
cd /workspace/job
export ND_FORK=/workspace/nd-takehome ND_CKPTS=/workspace/ckpts ND_RESULTS=/workspace/results
python -m pytest -q test_identities.py || exit 1
for r in 16 4 1 8 2; do python run_sae.py --seed 0 --round $r --block 4; done
python run_sae.py --seed 0 --round 16 --block 2 --ns 16 256 8192
for s in 1 2; do python run_sae.py --seed $s --round 16 --block 4 --ns 16 256 8192; done
echo QUEUE_DONE
