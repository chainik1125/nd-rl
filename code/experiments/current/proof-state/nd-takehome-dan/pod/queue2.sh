#!/usr/bin/env bash
# waits for the seed-0 arms to finish, then runs EI-long + seed-1 EI + seed-1 frozen concurrently, plus the abs-fixed ablation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/nd-takehome
until grep -q 'round 8 done' artifacts/ei_abs_s0.log 2>/dev/null && grep -q 'round 8 done' artifacts/frozen_abs_s0.log 2>/dev/null; do sleep 60; done
sleep 20
python3 expert_iter.py --init ckpts/stage1_abs.pt --name ei_abs_long_s0 --rounds 8 --k 32 --temperature 0.8 --seed 0 --select longest --batch 768 > artifacts/ei_abs_long_s0.log 2>&1 < /dev/null &
sleep 5
python3 expert_iter.py --init ckpts/stage1_abs.pt --name ei_abs_s1 --rounds 8 --k 32 --temperature 0.8 --seed 1 --batch 768 > artifacts/ei_abs_s1.log 2>&1 < /dev/null &
sleep 5
python3 expert_iter.py --init ckpts/stage1_abs.pt --name frozen_abs_s1 --rounds 8 --k 32 --temperature 0.8 --seed 1 --no_train --batch 768 > artifacts/frozen_abs_s1.log 2>&1 < /dev/null &
sleep 5
bash -c 'python3 train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode abs --no_shift --steps 6000 --bs 128 --out ckpts/stage1_absfixed.pt --cap 6 --seed 0 && python3 eval_set.py --ckpt ckpts/stage1_absfixed.pt --in data/heldout.jsonl --out artifacts/stage1_absfixed_heldout_greedy.jsonl --temperature 0 --batch 512 --summary artifacts/stage1_absfixed_heldout_greedy.json && python3 eval_set.py --ckpt ckpts/stage1_absfixed.pt --in data/transfer.jsonl --out artifacts/stage1_absfixed_transfer2_greedy.jsonl --temperature 0 --batch 512 --summary artifacts/stage1_absfixed_transfer2_greedy.json && python3 eval_set.py --ckpt ckpts/stage1_absfixed.pt --in data/transfer.jsonl --out artifacts/stage1_absfixed_transfer2_k16.jsonl --k 16 --temperature 0.8 --seed 0 --batch 512 --summary artifacts/stage1_absfixed_transfer2_k16.json' > artifacts/stage1_absfixed.log 2>&1 < /dev/null &
wait
