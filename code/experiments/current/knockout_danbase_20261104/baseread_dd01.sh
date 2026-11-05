#!/usr/bin/env bash
# The base reads dd01_s1 / dd01_s2's seg1 never made (temescal's CUDA was broken), made as dline
# makes them (danread.py, k 64 on the 106, batch 128, the line's seed, round 0) on serrano GPU 1,
# which is ours (dd01_s0), gated on no other user's process there.
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
KD=/scratch/cowifornia/nd-rl/knockout_danbase
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=1 DANREAD_MEM_FRAC=0.3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
for s in 1 2; do
  python3 -c "import sys; sys.path.insert(0,'$W'); import gpu; sys.exit(0 if gpu.free_for_us('serrano',1) else 1)" || { echo "serrano GPU 1 has another user's process; stop"; exit 2; }
  out=$KD/collected/temescal/dd01_s$s/seg1/reads/it0000.json
  mkdir -p $(dirname $out)
  timeout 1800 nice -n 5 $PY $W/danread.py --ckpt $KD/bases/stage1_dose01_s$s.pt --seed $s --round 0 \
    --out $out --only106 --k 64 --batch 128 >> $KD/logs/dd01_s${s}_baseread.log 2>&1
  echo "dd01_s$s base read rc $?: $(python3 -c "import json; d=json.load(open('$out')); print(len(d['first32_solved']), 'first-32;', len(d['strict_solved']), 'of 64')")"
done
