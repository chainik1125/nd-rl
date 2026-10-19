#!/usr/bin/env bash
# Regenerate the sealed 21-30-line test set (CPU only; about 3 h wall on 16-26 free cores), then calibrate it.
#   bash regenerate.sh
# Equivalent to the run that built /scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/test_21_30.jsonl on
# 2026-10-06 (that run interleaved selection, refinement and lower-bound batches as builds finished; README "Run
# log" lists the exact order).  Seeds are fixed, but generation and composition stop on wall-clock budgets
# (--secs / --max_secs), so a rerun reproduces the recipe and its distribution, not the identical set.
set -euo pipefail
REPO=/scratch/cowifornia/nd-rl/think/nd-rl
HERE=$REPO/code/experiments/current/thinking_traces_20261005/test_21_30
OUT=${OUT:-/scratch/cowifornia/nd-rl/think/runs/pools/test_21_30}
W=$OUT/work
PY=$REPO/code/.venv/bin/python
M=code.experiments.current.thinking_traces_20261005.test_21_30
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_LEAN=$HOME/.elan/bin/lean
mkdir -p "$W"
cd "$HERE"
mod() { (cd "$REPO" && $PY -m "$M.$1" "${@:2}"); }

# 0. exclusion index (K12 train / held-out, cap-6 Stage-1 set, every jsonl under the fork's data/, OOD sets, traces,
#    RL-pool files: raw/cands, cands_clean, long*_cands, rl_hard*)
mod seal index --out "$W/excl_index_final.pkl"

# 1. pieces: generator theorems with exact minimal lengths
$PY gen_pieces.py --out "$W/mains.jsonl"   --workers 8 --secs 1500 --seed 210000 --mode mix --goal any     --min_gen 9  --min_L 7
$PY gen_pieces.py --out "$W/lemmas.jsonl"  --workers 6 --secs 900  --seed 410000 --mode mix --goal literal --min_gen 7  --min_L 7
$PY gen_pieces.py --out "$W/mains2.jsonl"  --workers 8 --secs 1500 --seed 610000 --mode mix --goal any     --min_gen 11 --min_L 8
$PY gen_pieces.py --out "$W/lemmas2.jsonl" --workers 6 --secs 1500 --seed 810000 --mode mix --goal literal --min_gen 9  --min_L 8
mod prep_pieces --in "$W/mains.jsonl,$W/lemmas.jsonl,$W/mains2.jsonl,$W/lemmas2.jsonl" \
    --index "$W/excl_index_final.pkl" --out "$W/pieces_final.jsonl"

# 2. composites: one length-directed build over 21-30, then tail builds (prior builds' piece uses count)
B="$PY build_composites.py --pieces $W/pieces_final.jsonl"
$B --out "$W/comps.jsonl"       --lo 21 --hi 30 --per_len 600 --seed 2130 --max_uses 6 --max_secs 1500
$B --out "$W/comps_tail.jsonl"  --lo 24 --hi 30 --per_len 400 --seed 2631 --max_uses 8  --max_cuts 4 --max_fdepth 6 \
    --max_secs 1500 --max_tries 20000000
$B --out "$W/comps_tail2.jsonl" --lo 28 --hi 30 --per_len 250 --seed 2832 --max_uses 10 --max_cuts 5 --max_fdepth 6 \
    --max_secs 1500 --max_tries 20000000 --prior "$W/comps.jsonl,$W/comps_tail.jsonl"
$B --out "$W/comps_tail3.jsonl" --lo 28 --hi 30 --per_len 200 --seed 2933 --max_uses 10 --max_cuts 5 --max_fdepth 6 \
    --max_secs 1500 --max_tries 20000000 --prior "$W/comps.jsonl,$W/comps_tail.jsonl,$W/comps_tail2.jsonl"
# (the first three builds ran before the vacuous-antecedent filter existed; on 2026-10-06 their output was filtered
#  afterwards, with the same check that build_composites.py now applies: ndproof.unfold_discharged + semantic_flags)

# 3. select (<= 400 per reference length), refine (15 s vocabulary search), Lean, lower bounds (to 13 lines)
$PY assemble.py select --in "$W/comps.jsonl,$W/comps_tail.jsonl,$W/comps_tail2.jsonl,$W/comps_tail3.jsonl" \
    --out "$W/cands.jsonl" --lo 21 --hi 30 --per_len 400 --seed 0
$PY refine.py --in "$W/cands.jsonl" --out "$W/cands_ref.jsonl" --secs 15 --procs 16
LEAN_GATE_WORKERS=16 LEAN_GATE_LOG="$W/lean_gate.jsonl" mod lean_set --in "$W/cands_ref.jsonl" --out "$W/cands_lean.jsonl"
$PY minimality.py --in "$W/cands_lean.jsonl" --out "$W/cands_lb.jsonl" --bound 13 --stages 10:10,12:40,13:90 --procs 24

# 4. write (drops Lean rejects, proofs <= 13 lines, class / premise-permutation duplicates), seal, sidecar, profile
mod assemble write --in "$W/cands_lb.jsonl" --out "$W/test_21_30.unsealed.jsonl" --per_len 400 --seed 0
mod seal check --index "$W/excl_index_final.pkl" --set "$W/test_21_30.unsealed.jsonl" --report "$OUT/seal_report.json" \
    --rl_pool /scratch/cowifornia/nd-rl/think/runs/pools/rl_hard.jsonl --drop_out "$OUT/test_21_30.jsonl"
mod seal keys --set "$OUT/test_21_30.jsonl" --out "$OUT/test_21_30.keys.jsonl"
mod profile_set --set "$OUT/test_21_30.jsonl" --field reference_proof --k12 20000 --out "$OUT/profile.json"

# 5. calibration on GPU 0 (polls until GPU 0 < 500 MiB, <= 60 min per launch), report, power, meta
bash calib.sh "$OUT/test_21_30.jsonl" "$OUT/calib" 8
$PY calib_report.py --set "$OUT/test_21_30.jsonl" --rows T0=$OUT/calib/T0_s0/rows_plain_B0.jsonl \
    --rows T1v1_B256=$OUT/calib/T1v1_s0/rows_plain_B256.jsonl --out "$OUT/calib_report.json"
$PY power.py --a "$OUT/calib/T0_s0/rows_plain_B0.jsonl" --b "$OUT/calib/T1v1_s0/rows_plain_B256.jsonl" --out "$OUT/power.json"
$PY build_meta.py --out_dir "$OUT"
