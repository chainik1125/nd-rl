#!/usr/bin/env bash
# Block C targets: union of the 9-14 and 12-16-line strict-derived-ORE pools, keep only theorems with no <=8-line proof found.
cd /workspace/nd-takehome
set -e
cat data/p2/pool_long_c8.jsonl data/p2/pool_long_c8b.jsonl > data/p2/pool_long_c8_all.jsonl
cat data/p2/pool_long_c8_minlen_all.jsonl data/p2/pool_long_c8b_minlen.jsonl > data/p2/pool_long_c8_minlen_union.jsonl
python3 make_coverage_sets.py targets --pool data/p2/pool_long_c8_all.jsonl --minlen data/p2/pool_long_c8_minlen_union.jsonl --outdir data/p2 \
  --exclude data/p2/train_derived_ore_strict_f0_c8.jsonl data/p2/train_derived_ore_strict_f0.001_c8.jsonl data/p2/train_derived_ore_strict_f0.01_c8.jsonl data/p2/heldout_c8.jsonl \
  --require_long --min_ub 9 --patterns derived_ore_strict --n_targets 500 --n_transfer 250 --suffix _c8 --seed 0
cp data/p2/targets_derived_ore_strict_c8.jsonl data/p2/targets_c8.jsonl
cp data/p2/transfer_derived_ore_strict_c8.jsonl data/p2/transfer_c8.jsonl
wc -l data/p2/targets_c8.jsonl data/p2/transfer_c8.jsonl
echo BLOCKC_TARGETS3_DONE
