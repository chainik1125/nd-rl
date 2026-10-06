#!/usr/bin/env bash
# Run 2 addendum (p1): cap-8 set with box depth 4 REMOVED (f = 0 for depth-4 at cap 8, so 8-line proofs are in-distribution
# but no proof has a fourth box), two Stage-1 seeds, then EI + frozen + pass@2000 on the depth-4 pool.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "QUEUE DONE" artifacts/r2_arms_p1.log && grep -q "QUEUE DONE" artifacts/r2_ctl_p1.log; do sleep 60; done
mkdir -p data/r2 ckpts/r2 artifacts/r2
nice -n 5 python3 run2_sets.py --pool data/p2/pool_cap8.jsonl --outdir data/r2 --heldout data/p2/heldout_c8.jsonl --lens 2-8 --size 154994 \
  --exclude "data/r2/cands_*.jsonl" "data/r2/targets_*.jsonl" "data/r2/transfer_*.jsonl" --sets c8_depth4_f0:depth4 --seed 5 > artifacts/r2/sets_c8.log 2>&1
echo SETS_C8_DONE
cat > pod/r2/c8depth4_jobs.txt <<'Q'
stage1_r2_c8_depth4_f0_s0 python3 train.py --data data/r2/train_r2_c8_depth4_f0.jsonl --heldout data/p2/heldout_c8.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_c8_depth4_f0_s0.pt --cap 8 --seed 0
stage1_r2_c8_depth4_f0_s1 python3 train.py --data data/r2/train_r2_c8_depth4_f0.jsonl --heldout data/p2/heldout_c8.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_c8_depth4_f0_s1.pt --cap 8 --seed 1
ei_r2_depth4_c8f0_s0 until [ -f ckpts/r2/stage1_r2_c8_depth4_f0_s0.pt ]; do sleep 30; done; sleep 30; python3 expert_iter.py --init ckpts/r2/stage1_r2_c8_depth4_f0_s0.pt --name r2/ei_depth4_c8f0_s0 --targets data/r2/targets_depth4.jsonl --transfer data/r2/transfer_depth4.jsonl --heldout data/p2/heldout_c8.jsonl --train data/r2/train_r2_c8_depth4_f0.jsonl --rounds 8 --k 32 --temperature 0.8 --seed 0 --batch 768
ei_r2_depth4_c8f0_s1 until [ -f ckpts/r2/stage1_r2_c8_depth4_f0_s1.pt ]; do sleep 30; done; sleep 30; python3 expert_iter.py --init ckpts/r2/stage1_r2_c8_depth4_f0_s1.pt --name r2/ei_depth4_c8f0_s1 --targets data/r2/targets_depth4.jsonl --transfer data/r2/transfer_depth4.jsonl --heldout data/p2/heldout_c8.jsonl --train data/r2/train_r2_c8_depth4_f0.jsonl --rounds 8 --k 32 --temperature 0.8 --seed 1 --batch 768
frozen_r2_depth4_c8f0_s0 python3 expert_iter.py --init ckpts/r2/stage1_r2_c8_depth4_f0_s0.pt --name r2/frozen_depth4_c8f0_s0 --targets data/r2/targets_depth4.jsonl --transfer data/r2/transfer_depth4.jsonl --heldout data/p2/heldout_c8.jsonl --train data/r2/train_r2_c8_depth4_f0.jsonl --rounds 8 --k 32 --temperature 0.8 --seed 0 --batch 768 --no_train
frozen_r2_depth4_c8f0_s1 python3 expert_iter.py --init ckpts/r2/stage1_r2_c8_depth4_f0_s1.pt --name r2/frozen_depth4_c8f0_s1 --targets data/r2/targets_depth4.jsonl --transfer data/r2/transfer_depth4.jsonl --heldout data/p2/heldout_c8.jsonl --train data/r2/train_r2_c8_depth4_f0.jsonl --rounds 8 --k 32 --temperature 0.8 --seed 1 --batch 768 --no_train
cov_r2_depth4_c8f0_s0 python3 coverage.py --ckpt ckpts/r2/stage1_r2_c8_depth4_f0_s0.pt --in data/r2/targets_depth4.jsonl --out artifacts/r2/cov_depth4_c8f0_s0 --k 2000 --temperature 0.8 --limit 300 --batch 1024 --seed 0 --procs 3
cov_r2_depth4_c8f0_s1 python3 coverage.py --ckpt ckpts/r2/stage1_r2_c8_depth4_f0_s1.pt --in data/r2/targets_depth4.jsonl --out artifacts/r2/cov_depth4_c8f0_s1 --k 2000 --temperature 0.8 --limit 300 --batch 1024 --seed 0 --procs 3
Q
bash pod/runq.sh pod/r2/c8depth4_jobs.txt 2
echo C8DEPTH4_DONE
