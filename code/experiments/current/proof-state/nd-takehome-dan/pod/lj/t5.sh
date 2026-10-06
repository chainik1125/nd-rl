#!/usr/bin/env bash
# Acceptance test 5 of run lean-judge: one expert-iteration round end to end under the Lean-only judge.
# Checks: (a) no LEAN* marker in any training file, (b) every checked sample dumped with its Lean verdict so the old
# gate's count (Lean AND nd_verify) can be re-derived off the pod, (c) at least one hindsight-relabelled acceptance.
set -x
cd /workspace/nd-takehome
export PATH=$HOME/.elan/bin:$PATH
export LEAN_GATE_WORKERS=6 LEAN_GATE_CHUNK=400 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEAN_GATE_LOG=artifacts/lj/t5_gate.jsonl LEAN_GATE_DUMP=artifacts/lj/t5_dump.jsonl
mkdir -p artifacts/lj
python3 expert_iter.py --init ckpts/stage1_a3_s0.pt --name lj/t5_ei \
  --targets data/lj/targets200.jsonl --transfer data/lj/transfer50.jsonl --heldout data/lj/heldout200.jsonl \
  --train data/lj/train_retain.jsonl --rounds 1 --k 16 --temperature 0.8 --seed 0 --relabel \
  --retain 3000 --ft_steps 50 --batch 512 2>&1 | tee artifacts/lj/t5_ei.log
echo "=== marker grep over every file the round writes ==="
grep -c LEAN artifacts/lj/t5_ei/found_1.jsonl artifacts/lj/t5_ei/found_transfer_1.jsonl artifacts/lj/t5_ei/mix_1.jsonl 2>&1 | tee artifacts/lj/t5_marker_grep.txt
wc -l artifacts/lj/t5_ei/found_1.jsonl artifacts/lj/t5_ei/found_transfer_1.jsonl artifacts/lj/t5_ei/mix_1.jsonl artifacts/lj/t5_dump.jsonl | tee -a artifacts/lj/t5_marker_grep.txt
