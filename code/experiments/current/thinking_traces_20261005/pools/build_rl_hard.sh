#!/usr/bin/env bash
# Assemble runs/pools/rl_hard.jsonl (+ .meta.json) from the measured candidates (CPU only, ~1 min).
#   rl_hard.jsonl           Leon's option B: short band (7-12 lines) filtered to T0 pass@1 <= 0.625,
#                           long band (13-20 lines) measured but not filtered; 2,500 each
#   rl_hard_filtered.jsonl  the same with the long band filtered too (pass@1 <= 0.625)
# Renaming classes of the eval sets, transfer pools and rl_targets are dropped (expected: 0 hits,
# prep.py already removed them), and so are the 21-30-line test set's classes when
# runs/pools/test_21_30/test_21_30.jsonl exists: rerun this script once it does.
set -eu
R=/scratch/cowifornia/nd-rl/think/runs/pools
F=/scratch/cowifornia/nd-rl/think/nd-takehome/data
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
EX=(/scratch/cowifornia/nd-rl/think/ood_check/sets/*.jsonl
    $F/eval_only/textbook72/textbook_dev.jsonl $F/eval_only/textbook72/textbook_train.jsonl
    /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl
    $F/ladder/transfer.jsonl $F/transfer.jsonl $F/ladder/rl_targets.jsonl $F/rl_targets.jsonl)
TEST=$R/test_21_30/test_21_30.jsonl
if [ -s "$TEST" ]; then EX+=("$TEST"); else echo "note: $TEST not found; not deduplicated against it"; fi
ROWS=($R/passrate/short_pilot/rows.jsonl $R/passrate/short/rows.jsonl
      $R/passrate/long/rows.jsonl $R/passrate/long2/rows.jsonl)
SETS=($R/short_pilot.jsonl $R/short_cands.jsonl $R/long_cands.jsonl $R/long2_cands.jsonl)
P="code/.venv/bin/python -m code.experiments.current.thinking_traces_20261005.pools.assemble"
$P rl --rows "${ROWS[@]}" --sets "${SETS[@]}" --short_lens 7-12 --long_lens 13-20 \
  --n_short 2500 --n_long 2500 --exclude_sets "${EX[@]}" --out $R/rl_hard.jsonl
$P rl --rows "${ROWS[@]}" --sets "${SETS[@]}" --short_lens 7-12 --long_lens 13-20 \
  --n_short 2500 --n_long 2500 --long_max_pass1 0.625 --exclude_sets "${EX[@]}" \
  --out $R/rl_hard_filtered.jsonl
