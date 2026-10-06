#!/usr/bin/env bash
# Run 2 (p3): assemble the four pretraining sets (candidate classes of every run-2 pool excluded), then 8 Stage-1 models.
cd /workspace/nd-takehome
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f data/r2/cands_impi_ore.jsonl ] && [ -f data/r2/cands_negi_ande_hyp.jsonl ] && [ -f data/r2/cands_depth4.jsonl ] && grep -q "NEC_DONE impe_chain4" artifacts/r2_targets2.log; do sleep 30; done
nice -n 5 python3 run2_sets.py --pool data/p2/pool_cap6_recon.jsonl --outdir data/r2 --heldout data/p2/heldout.jsonl \
  --exclude "data/r2/cands_*.jsonl" --sets struct:none,impi_ore_f0:impi_ore,negi_ande_hyp_f0:negi_ande_hyp,ori_ore_f0:ori_ore --seed 0 > artifacts/r2/sets.log 2>&1
echo SETS_DONE
mkdir -p ckpts/r2
cat > pod/r2/stage1.txt <<'Q'
stage1_r2_struct_s0 python3 train.py --data data/r2/train_r2_struct.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_struct_s0.pt --cap 6 --seed 0
stage1_r2_struct_s1 python3 train.py --data data/r2/train_r2_struct.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_struct_s1.pt --cap 6 --seed 1
stage1_r2_impi_ore_f0_s0 python3 train.py --data data/r2/train_r2_impi_ore_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_impi_ore_f0_s0.pt --cap 6 --seed 0
stage1_r2_impi_ore_f0_s1 python3 train.py --data data/r2/train_r2_impi_ore_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_impi_ore_f0_s1.pt --cap 6 --seed 1
stage1_r2_negi_ande_hyp_f0_s0 python3 train.py --data data/r2/train_r2_negi_ande_hyp_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_negi_ande_hyp_f0_s0.pt --cap 6 --seed 0
stage1_r2_negi_ande_hyp_f0_s1 python3 train.py --data data/r2/train_r2_negi_ande_hyp_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_negi_ande_hyp_f0_s1.pt --cap 6 --seed 1
stage1_r2_ori_ore_f0_s0 python3 train.py --data data/r2/train_r2_ori_ore_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_ori_ore_f0_s0.pt --cap 6 --seed 0
stage1_r2_ori_ore_f0_s1 python3 train.py --data data/r2/train_r2_ori_ore_f0.jsonl --heldout data/p2/heldout.jsonl --mode abs --steps 6000 --bs 128 --out ckpts/r2/stage1_r2_ori_ore_f0_s1.pt --cap 6 --seed 1
Q
bash pod/runq.sh pod/r2/stage1.txt 2
echo STAGE1_DONE
