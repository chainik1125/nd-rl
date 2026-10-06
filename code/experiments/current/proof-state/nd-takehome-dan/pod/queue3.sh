#!/usr/bin/env bash
# after the seed-0 arms finish: continue EI seed 0 for rounds 9-16 and the frozen control for rounds 9-16 (cumulative bookkeeping resumed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/nd-takehome
until grep -q 'round 8 done' artifacts/ei_abs_s0.log 2>/dev/null && grep -q 'round 8 done' artifacts/frozen_abs_s0.log 2>/dev/null; do sleep 60; done
sleep 60
python3 expert_iter.py --init ckpts/ei_abs_s0_r8.pt --name ei_abs_s0_cont --start_round 9 --resume_found artifacts/ei_abs_s0 --rounds 8 --k 32 --temperature 0.8 --seed 0 --batch 768 > artifacts/ei_abs_s0_cont.log 2>&1 < /dev/null &
sleep 5
python3 expert_iter.py --init ckpts/stage1_abs.pt --name frozen_abs_s0_cont --start_round 9 --resume_found artifacts/frozen_abs_s0 --rounds 8 --k 32 --temperature 0.8 --seed 0 --no_train --batch 768 > artifacts/frozen_abs_s0_cont.log 2>&1 < /dev/null &
wait
