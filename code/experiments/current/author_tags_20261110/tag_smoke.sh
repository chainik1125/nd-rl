#!/usr/bin/env bash
# CPU smoke of every path, end to end, at toy sizes (no GPU: CUDA_VISIBLE_DEVICES is empty; 8 threads, 8 Lean workers).
#
#   tag_smoke.sh [all|pretrain|sample|line|read|check|variant]
#
# Writes only under /scratch/cowifornia/nd-rl/author_tags/smoke/. Steps:
#   pretrain  tag_pretrain.py: a 3 x 128 model, 800 steps on the first 4,000 tagged records (about 10 min)
#   sample    tag_sample.py: 16 short training-corpus theorems x 4 under A, B and none, Dan's sampler + Lean
#   line      tag_line.py --smoke in a smoke tree: GRPO under B (2 iterations of 4 x 4), the base read and the
#             snapshot reads under B (k 2 on the 106), the line view; then a second line of 60 iterations with a
#             tiny D, which the compute cap must stop (a rule check every 3 s kills GRPO)
#   read      tag_read.py gpu/rows on the CPU: 1 theorem per set, k 2, k_f 1, under A, B and none
#   check     tag_check.py: 20 records per set under A, B and none
#   variant   a tie-strength variant (p10; its corpus must be built): tag_pretrain.py --variant on the first 4,000
#             records, 200 steps, with Dan's held-out file (the variant's held-out rule); tag_check.py on it (the
#             variant's corpus from the checkpoint); tag_gate.py on that read (about 5 min)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
S=/scratch/cowifornia/nd-rl/author_tags/smoke
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8
export ND_OFFLINE=1 ND_REGISTRY_SYNC=0
cd "$HERE"
what=${1:-all}
BASE=$S/pre/stage1_small_s0.pt

if [ "$what" = all ] || [ "$what" = pretrain ]; then
  $PY tag_pretrain.py --seed 0 --out $BASE --steps 800 --limit 4000 --best_dims 3,128,4,384 --tok_budget 4096 \
    --curve_every 60 --heldout "" 2>&1 | grep -v -i warn | tail -5
fi
if [ "$what" = all ] || [ "$what" = sample ]; then
  $PY tag_sample.py --ckpt $BASE --n 16 --k 4 --max_lines 4 --out $S/sample_small.json 2>&1 | grep "^tag"
fi
if [ "$what" = all ] || [ "$what" = line ]; then
  for name in at at_tinyD; do
    AT=$S/$name; rm -rf "$AT"; mkdir -p $AT/bases $AT/logs
    ln -s $BASE $AT/bases/stage1_tagAB_s0.pt
    D=470535415; [ $name = at_tinyD ] && D=1000
    echo "{\"train_tokens\": $D, \"smoke\": true}" > $AT/bases/stage1_tagAB_s0.tag.json
    IT=2; [ $name = at_tinyD ] && IT=60   # the rule check every 3 s must end it early
    AT=$AT $PY tag_line.py --line tB_s0 --gpu 0 --smoke --iters $IT --rule_every 3 || true
    cat $AT/runs/tB_s0/seg1/end.json; echo
    AT=$AT $PY tag_measure.py tB_s0
  done
fi
if [ "$what" = all ] || [ "$what" = read ]; then
  rm -rf $S/read
  $PY tag_read.py gpu --seeds 0 --tags A,B,none --base $BASE --root $S/read --device cpu --limit 1 --k 2 --kf 1 \
    --batch 64 2>&1 | tail -12
  $PY tag_read.py rows --seeds 0 --tags A,B,none --base $BASE --root $S/read --limit 1 2>&1 | tail -4
fi
if [ "$what" = all ] || [ "$what" = check ]; then
  $PY tag_check.py --ckpt $BASE --out $S/check_small.json --n 20 --device cpu 2>&1 | tail -6
fi
if [ "$what" = all ] || [ "$what" = variant ]; then
  V=$S/variant; rm -rf "$V"; mkdir -p $V/checkdir
  $PY tag_pretrain.py --seed 0 --variant p10 --out $V/stage1_small_p10_s0.pt --steps 200 --limit 4000 \
    --best_dims 3,128,4,384 --tok_budget 4096 --curve_every 60 2>&1 | grep -v -i warn | tail -3
  $PY tag_check.py --ckpt $V/stage1_small_p10_s0.pt --out $V/checkdir/tagAB_p10_s0.json --n 20 --device cpu 2>&1 | tail -6
  $PY tag_gate.py --dir $V/checkdir
fi
