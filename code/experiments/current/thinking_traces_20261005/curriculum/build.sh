#!/usr/bin/env bash
# Length-curriculum RL pools L2 (10-12 lines, RL stage 1) and L3 (13-15 lines, RL stage 2): the steps of the
# 2026-10-09/10 build, in order. CPU only, at most 12 worker processes. Run from this worktree's root
# (leon/thinking-curriculum) with the main venv; see README.md. (L1, 7-9 lines, was started and dropped: see README.)
set -euo pipefail
C=/scratch/cowifornia/nd-rl/think/runs/pools/curriculum
H=/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=code.experiments.current.thinking_traces_20261005
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PFG_MEM_GB=3.0
N="nice -n 10"
B13="PFG_MIN_L=13 PFG_MAX_L=15"
ADAPT="PFG_CLASSICAL_VALUE=0 PFG_EXFALSO_VALUE=0 PFG_SCARCE_BONUS=0"

# 1. exclusion index: hard712's 364 files + hard712's outputs + every jsonl under runs/pools and the eval subsets
$N $PY -m $X.curriculum.excl --base $H/excl_index.pkl --out $C/work/excl_index.pkl

# 2. L2 source: hard712 rows no RL run or sealed split used, at 10-12 lines
$N $PY -m $X.curriculum.reuse --lo 10 --hi 12 --index $C/work/excl_index.pkl --out $C/L2/reused.jsonl

# 3. L3 sources (the production run interleaved these; the order and worker counts below are what ran)
#  a. fresh seeds 0..~1M at default recipe weights (pilot + first round), then adaptive weights valuing rows that are
#     neither classical-only nor ex falso; gap limit 1
G="$X.hard712.run --out $C/L3/gen generate --chunk 10000 --index $C/work/excl_index.pkl --target 12000"
env $B13 $N $PY -m $G --workers 6 --round 1000000 --stop_seed 1000000 --hours 0.6      # pilot (stopped mid-round)
env $B13 $ADAPT $N $PY -m $G --workers 8 --round 500000 --hours 5.0 --adaptive
#  b. hard712's 4,847 classes rejected only for a 13-line shortest proof, rebuilt from their seeds and certified
env $B13 PFG_MAX_GAP=1 $N $PY -m $X.curriculum.regen --lo 13 --hi 15 --status long \
  --index $C/work/excl_index.pkl --out $C/L3/regen --workers 8
#  c. generation goes on skipping classes 3b certifies, the held-out and sealed sets' keys (skip_keys.txt)
env $B13 $ADAPT $N $PY -m $G --workers 12 --round 500000 --hours 1.9 --adaptive --skip $C/L3/gen/skip_keys.txt
#  d. 3a's construction-gap rejects (constructed proof 2-3 lines over the shortest), gap limit 3, then 3a with gap 3
env $B13 PFG_MAX_GAP=3 $N $PY -m $X.curriculum.regen --lo 13 --hi 15 --status gap --source $C/L3/gen \
  --index $C/work/excl_index.pkl --out $C/L3/regen_gap --workers 12
env $B13 $ADAPT PFG_MAX_GAP=3 $N $PY -m $G --workers 12 --round 500000 --hours 0.78 --adaptive \
  --skip $C/L3/gen/skip_keys.txt

# 4. Lean on every new row; reservoirs (L3 first, so cross-band curried twins leave L2); mix gate; splits
export LEAN_GATE_WORKERS=12
for d in L3/regen L3/regen_gap L3/gen; do $N $PY -m $X.hard712.run --out $C/$d lean; done
$N $PY -m $X.curriculum.assemble --band_dir $C/L3 --lo 13 --hi 15 --fresh regen=$C/L3/regen/work/gated_lean.jsonl \
  regen_gap=$C/L3/regen_gap/work/gated_lean.jsonl gen=$C/L3/gen/work/gated_lean.jsonl
$N $PY -m $X.curriculum.assemble --band_dir $C/L2 --reused $C/L2/reused.jsonl --lo 10 --hi 12 \
  --avoid $C/L3/work/gated_lean.jsonl
$N $PY -m $X.curriculum.find_cap $C/L3 6000                       # prints the caps it settles on
# L2 with L3's classical-only, ex-falso, reductio and nested-negation shares as caps (CAPS_L2: see README)
$N $PY -m $X.hard712.run --out $C/L2 select --n 6000 --workers 12 --caps "$CAPS_L2"
for b in L2 L3; do
  $N $PY -m $X.hard712.run --out $C/$b split --n_test 1000 --n_dev 0 --prefix c$b --source curriculum_$b
  rm -f $C/$b/dev_500.jsonl $C/$b/dev_500.keys.jsonl
done

# 5. final checks (Lean on every final row, index, textbook72, curried held-out and sealed sets, disjointness)
LEAN_GATE_LOG=$C/work/lean_gate_report.jsonl $N $PY -m $X.curriculum.report --root $C
