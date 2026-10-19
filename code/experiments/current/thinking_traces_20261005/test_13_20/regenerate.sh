#!/usr/bin/env bash
# Regenerate the sealed 13-20-line companion set (CPU, about 3 h on 26 free cores), then calibrate it.
#   bash regenerate.sh
# The recipe is test_21_30's (scripts there are reused unchanged, plus `--prefix`/`cert` in assemble.py);
# 13-20 differs in the length range, the larger exclusion index (excl.py) and exact certification.
# Seeds are fixed, but composition / refinement / search stop on time budgets, so a rerun reproduces
# the recipe and its distribution, not the identical set.
set -euo pipefail
REPO=/scratch/cowifornia/nd-rl/think/nd-rl
EXP=$REPO/code/experiments/current/thinking_traces_20261005
T21=$EXP/test_21_30
HERE=$EXP/test_13_20
OUT=${OUT:-/scratch/cowifornia/nd-rl/think/runs/pools/test_13_20}
W=$OUT/work
W21=/scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/work
PY=$REPO/code/.venv/bin/python
M=code.experiments.current.thinking_traces_20261005
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_LEAN=$HOME/.elan/bin/lean
mkdir -p "$W"
mod() { (cd "$REPO" && $PY -m "$M.$1" "${@:2}"); }

# 0. exclusion index: test_21_30's 106 files + rl_hard12, hard12_test_1k, hard12 pools/work, other
#    runs/pools/*.jsonl, and the 21-30 test set (120 files)
mod test_13_20.excl --out "$W/excl_index.pkl"

# 1. pieces: test_21_30's generator runs (seeds 210000/410000/610000/810000 and the small-piece runs
#    950000/970000/960000/900000), re-filtered against this index
mod test_21_30.prep_pieces --index "$W/excl_index.pkl" --out "$W/pieces.jsonl" \
    --in "$W21/mains.jsonl,$W21/lemmas.jsonl,$W21/mains2.jsonl,$W21/lemmas2.jsonl,$W21/val_pieces.jsonl,$W21/small_lit.jsonl,$W21/pilot_lit.jsonl"

# 2. composites over 13-20 (final recipe, vacuous-antecedent filter included), refinement, selection
$PY "$T21/build_composites.py" --pieces "$W/pieces.jsonl" --out "$W/comps.jsonl" --lo 13 --hi 20 \
    --per_len 500 --seed 1320 --max_uses 3 --max_piece_L 12 --max_secs 900
$PY "$T21/refine.py" --in "$W/comps.jsonl" --out "$W/comps_ref.jsonl" --secs 15 --procs 24
$PY "$T21/assemble.py" select --in "$W/comps_ref.jsonl" --out "$W/sel.jsonl" --lo 13 --hi 20 \
    --per_len 380 --seed 0
LEAN_GATE_WORKERS=16 LEAN_GATE_LOG="$W/lean_gate.jsonl" mod test_21_30.lean_set --in "$W/sel.jsonl" \
    --out "$W/sel_lean.jsonl"

# 3. certification: exact search to lines - 1 at 13-17, lower bound to 15 at 18-20
$PY - <<PYEOF
import json
r = [json.loads(x) for x in open("$W/sel_lean.jsonl")]
n = lambda x: len(x["proof"].split(";")) - 1
open("$W/cert_in_13_17.jsonl", "w").writelines(json.dumps(x) + "\n" for x in r if n(x) <= 17)
open("$W/cert_in_18_20.jsonl", "w").writelines(json.dumps(x) + "\n" for x in r if n(x) >= 18)
PYEOF
$PY "$T21/minimality.py" --in "$W/cert_in_13_17.jsonl" --out "$W/cert_out_13_17.jsonl" --bound 16 \
    --stages 10:5,12:30,14:120,15:300,16:1800 --procs 16
$PY "$T21/minimality.py" --in "$W/cert_in_18_20.jsonl" --out "$W/cert_out_18_20.jsonl" --bound 15 \
    --stages 10:5,12:30,14:120,15:600 --procs 10
cat "$W/cert_out_13_17.jsonl" "$W/cert_out_18_20.jsonl" > "$W/cert_out.jsonl"
$PY "$HERE/certify.py" --in "$W/cert_out.jsonl" --out "$W/cert.jsonl"
LEAN_GATE_WORKERS=16 LEAN_GATE_LOG="$W/lean_gate.jsonl" mod test_21_30.lean_set --in "$W/cert.jsonl" \
    --out "$W/cert_lean.jsonl"

# 4. write, seal, ~F fix (negf.py: 2-line ( ~ F ) boxes), format limits re-applied, sidecar, profile
mod test_21_30.assemble write --in "$W/cert_lean.jsonl" --out "$W/test_13_20.unsealed.jsonl" \
    --lo 13 --hi 20 --per_len 380 --seed 0 --prefix t1320
mod test_21_30.seal check --index "$W/excl_index.pkl" --set "$W/test_13_20.unsealed.jsonl" \
    --report "$W/seal_report_pre.json" --drop_out "$W/test_13_20.pre_negf.jsonl"
LEAN_GATE_WORKERS=16 LEAN_GATE_LOG="$W/lean_gate.jsonl" mod test_13_20.negf \
    --set "$W/test_13_20.pre_negf.jsonl" --out "$W/test_13_20.pre_format.jsonl" --lo 13 --hi 20
$PY "$HERE/format_filter.py" --set "$W/test_13_20.pre_format.jsonl" --out "$OUT/test_13_20.jsonl"
mod test_21_30.seal check --index "$W/excl_index.pkl" --set "$OUT/test_13_20.jsonl" \
    --report "$OUT/seal_report.json"
mod test_21_30.seal keys --set "$OUT/test_13_20.jsonl" --out "$OUT/test_13_20.keys.jsonl"
mod test_21_30.profile_set --set "$OUT/test_13_20.jsonl" --field reference_proof --k12 20000 \
    --out "$OUT/profile.json"

# 5. calibration (GPU 0 lock protocol; 600 items stratified by length), report, power, meta.  The
#    2026-10-07 sample was drawn from the 3,040 candidates (cands_ood.jsonl) before certification so
#    that GPU time overlapped CPU work; 595 of its items are in the final set.
$PY "$HERE/subsample.py" --set "$W/cands_ood.jsonl" --out "$OUT/calib600_13_20.jsonl" --n 600
bash "$HERE/calib_all.sh"
$PY "$T21/calib_report.py" --set "$OUT/test_13_20.jsonl" --out "$OUT/calib_report.json" \
    --rows T0_B0="$OUT/calib/T0_s0/rows_plain_B0.jsonl" \
    --rows T1v1_B0="$OUT/calib/T1v1_s0/rows_plain_B0.jsonl" \
    --rows T1v1_B256="$OUT/calib/T1v1_s0/rows_plain_B256.jsonl"
$PY "$HERE/power13.py" --out_dir "$OUT"
$PY "$HERE/meta13.py" --out_dir "$OUT" --extra "$W/meta_extra.json"
