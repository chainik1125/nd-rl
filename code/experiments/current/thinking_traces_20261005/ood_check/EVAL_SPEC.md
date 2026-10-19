# Calibration eval spec (not run; for approval)

```bash
OUT=/scratch/cowifornia/nd-rl/think/ood_eval_20261006; mkdir -p $OUT/tmp $OUT/registry
cp -r /scratch/cowifornia/nd-rl/think/ood_check/sets $OUT/sets
cd /scratch/cowifornia/nd-rl/think/nd-takehome          # read-only; every output goes to $OUT
PY=/scratch/cowifornia/nd-rl/bench/nd-rl/code/.venv/bin/python
export ND_OFFLINE=1 ND_REGISTRY_DIR=$OUT/registry LEAN_GATE_LOG=$OUT/lean_gate.jsonl \
       TMPDIR=$OUT/tmp PYTHONDONTWRITEBYTECODE=1 PATH=$HOME/.elan/bin:$PATH
CK=/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state
S=$OUT/sets; SETS=$S/ood_or_reorder.jsonl,$S/ood_and_reorder.jsonl,$S/ood_commute.jsonl,$S/ood_case_product.jsonl,$S/charles_long_reach.jsonl
nvidia-smi    # pick one idle 3090 -> G
# smoke test (~2 min):
CUDA_VISIBLE_DEVICES=$G $PY guided_eval.py --ckpt $CK/ladder/la_T1_best12_s0_r8.pt --arm plain --k 4 --limit 8 --batch 512 --sets $SETS --out $OUT/smoke
for ck in ladder/la_T1_best12_s0_r8 stage1/stage1_best12_s0_b1200; do for arm in plain logical; do
  CUDA_VISIBLE_DEVICES=$G $PY guided_eval.py --ckpt $CK/$ck.pt --arm $arm --k 64 --seed 1 --batch 1024 \
    --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10 --sets $SETS --out $OUT/$(basename $ck)_$arm
done; done
git status --short --ignored    # must stay empty
```

Estimate on one 3090:

| job | minutes |
|---|---|
| r8 plain | about 27 |
| r8 logical | about 58 |
| pretrained plain | about 11 |
| pretrained logical | about 27 |
| **total** | **about 2 GPU-h** |

If batch 1024 runs out of memory, use 512.
