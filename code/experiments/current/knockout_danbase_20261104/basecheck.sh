#!/usr/bin/env bash
# Dan's round-1 transfer draw on each base (his seed, batch 1024, all 832), serrano GPU 0, gated.
W=/scratch/cowifornia/nd-rl/worktrees/knockout-grpo/code/experiments/current/knockout_danbase_20261104
KD=/scratch/cowifornia/nd-rl/knockout_danbase
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=0 ND_REGISTRY=0
for b in ko_s0 ko_s1 ko_s2 full_s3; do
  s=${b##*_s}
  python3 -c "import sys; sys.path.insert(0,'$W'); import gpu; sys.exit(0 if gpu.free_for_us('serrano',0) else 1)" || { echo "GPU 0 not free, stop"; exit 2; }
  timeout 1800 nice -n 5 $PY $W/danread.py --ckpt $KD/bases/stage1_$b.pt --seed $s --round 1 --out $KD/reads/base/$b.json
done
