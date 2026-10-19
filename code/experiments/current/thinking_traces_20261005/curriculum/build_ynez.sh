#!/usr/bin/env bash
# ynez rebuild (2026-10-13): NEW curriculum pools L2r (10-12 lines) and L3r (13-15 lines) with the recorded recipe
# (build.sh, README.md). serrano's L2/L3 were lost and cannot be reproduced exactly (time-limited gates, hash-order
# dependent shortest proofs, adaptive weights, an interrupted multi-run L3 generation), so these are new sets; their
# test_1k splits are NOT serrano's test splits. Differences from build.sh, all deterministic:
#   * PYTHONHASHSEED=0 everywhere; at most 20 worker processes, nice 10; every step bounded by `timeout`;
#   * time limits calibrated to serrano's timeout rates (calib_timeouts.py): PFG_SEARCH_SECS=12, PFG_NEC_SECS=14.5;
#   * generation stops at fixed seeds (--stop_seed, rounds of 500k from seed 0) instead of --hours / killed jobs,
#     over serrano's seed ranges: 0-1M default weights; 1M-8.5M adaptive, no skip; 8.5M-18.75M adaptive with
#     skip_keys; gap regen; 18.75M-25.75M adaptive, gap 3, skip_keys + regen_gap keys;
#   * skip_keys built once regen-long has finished (serrano grew it while regen-long ran);
#   * the exclusion index is rebuilt from the files that exist now (see REBUILD_NOTE.md);
#   * L2's caps are L3r's realized shares (the recipe), not serrano's L3 numbers.
#     bash build_ynez.sh PHASE      (PHASE: 1 regen-long + gen 0-8.5M | 2 skip keys + gen to 18.75M |
#                                    3 gap regen + gen to 25.75M | 4 Lean, assemble, select, split, report)
set -uo pipefail   # no -e: each step's exit status is logged; phase 4 stops on a failed step
WT=/scratch/cowifornia/nd-rl/think/nd-rl-currb
C=/scratch/cowifornia/nd-rl/think/runs/pools/curriculum
H=/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=code.experiments.current.thinking_traces_20261005
cd $WT
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= PFG_MEM_GB=3.0
export PFG_SEARCH_SECS=12 PFG_NEC_SECS=14.5
N="nice -n 10"
B13="PFG_MIN_L=13 PFG_MAX_L=15"
ADAPT="PFG_CLASSICAL_VALUE=0 PFG_EXFALSO_VALUE=0 PFG_SCARCE_BONUS=0"
G="$X.hard712.run --out $C/L3/gen generate --round 500000 --chunk 10000 --index $C/work/excl_index.pkl"
G="$G --target 999999 --hours 99"
L=$C/logs; mkdir -p $L $C/L3/gen
say() { echo "$(date -u +%FT%TZ) $*"; }

skip_add() {  # add the certified rows of a gated file (pkey + curried / held-out keys) to skip_keys.txt
  timeout 1800 $N $PY - "$1" <<'EOF'
import json, sys
from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
C = "/scratch/cowifornia/nd-rl/think/runs/pools/curriculum"
keys = {l.strip() for l in open(f"{C}/L3/gen/skip_keys.txt") if l.strip()}
n0 = len(keys)
for l in open(sys.argv[1]):
    r = json.loads(l)
    keys |= {r["pkey"]} | HO.keys(r["prompt"])
open(f"{C}/L3/gen/skip_keys.txt", "w").write("\n".join(sorted(keys)) + "\n")
print("skip keys", n0, "->", len(keys))
EOF
}

case ${1:?phase} in
1)
  say "phase 1: regen-long (10 workers) || gen 0-8.5M (10 workers)"
  ( env $B13 PFG_MAX_GAP=1 timeout 21600 $N $PY -m $X.curriculum.regen --lo 13 --hi 15 --status long \
      --index $C/work/excl_index.pkl --out $C/L3/regen --workers 10 > $L/regen_long.log 2>&1
    say "regen-long exit $?" ) &
  ( env $B13 timeout 7200 $N $PY -m $G --workers 10 --stop_seed 1000000 > $L/gen_a.log 2>&1
    say "gen 0-1M (default weights) exit $?"
    env $B13 $ADAPT timeout 21600 $N $PY -m $G --workers 10 --adaptive --stop_seed 8500000 > $L/gen_b.log 2>&1
    say "gen 1M-8.5M (adaptive) exit $?" ) &
  wait; say "phase 1 done" ;;
2)
  say "phase 2: skip keys, gen 8.5M-18.75M (20 workers)"
  timeout 1800 $N $PY - <<'EOF'
import json
from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
C = "/scratch/cowifornia/nd-rl/think/runs/pools/curriculum"
keys = set()
for l in open("/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work/rejects.jsonl"):
    r = json.loads(l)
    if r["status"] == "long":
        keys.add(r["pkey"])
n_long = len(keys)
for l in open(f"{C}/L3/regen/work/gated.jsonl"):
    r = json.loads(l)
    keys |= {r["pkey"]} | HO.keys(r["prompt"])
keys |= set(HO.heldout_index({**HO.FILES, **HO.SEALED}))
open(f"{C}/L3/gen/skip_keys.txt", "w").write("\n".join(sorted(keys)) + "\n")
print("long pkeys", n_long, "total skip keys", len(keys))
EOF
  env $B13 $ADAPT timeout 21600 $N $PY -m $G --workers 20 --adaptive --stop_seed 18750000 \
    --skip $C/L3/gen/skip_keys.txt > $L/gen_c.log 2>&1
  say "gen 8.5M-18.75M exit $?" ;;
3)
  say "phase 3: gap regen (20 workers), gen 18.75M-25.75M with gap 3 (20 workers)"
  env $B13 PFG_MAX_GAP=3 timeout 7200 $N $PY -m $X.curriculum.regen --lo 13 --hi 15 --status gap \
    --source $C/L3/gen --index $C/work/excl_index.pkl --out $C/L3/regen_gap --workers 20 > $L/regen_gap.log 2>&1
  say "regen_gap exit $?"
  skip_add $C/L3/regen_gap/work/gated.jsonl
  env $B13 $ADAPT PFG_MAX_GAP=3 timeout 14400 $N $PY -m $G --workers 20 --adaptive --stop_seed 25750000 \
    --skip $C/L3/gen/skip_keys.txt > $L/gen_d.log 2>&1
  say "gen 18.75M-25.75M exit $?" ;;
4)
  say "phase 4: Lean, assemble, select, split, report"
  export LEAN_GATE_WORKERS=20
  for d in L3/regen L3/regen_gap L3/gen; do
    timeout 7200 $N $PY -m $X.hard712.run --out $C/$d lean > $L/lean_$(basename $d).log 2>&1 \
      || { say "lean $d FAILED"; exit 1; }
    say "lean $d done"
  done
  timeout 1800 $N $PY -m $X.curriculum.assemble --band_dir $C/L3 --lo 13 --hi 15 \
    --fresh regen=$C/L3/regen/work/gated_lean.jsonl regen_gap=$C/L3/regen_gap/work/gated_lean.jsonl \
    gen=$C/L3/gen/work/gated_lean.jsonl | tee $L/assemble_L3.log || { say "assemble L3 FAILED"; exit 1; }
  timeout 1800 $N $PY -m $X.curriculum.assemble --band_dir $C/L2 --reused $C/L2/reused.jsonl --lo 10 --hi 12 \
    --avoid $C/L3/work/gated_lean.jsonl | tee $L/assemble_L2.log || { say "assemble L2 FAILED"; exit 1; }
  timeout 7200 $N $PY -m $X.curriculum.find_cap $C/L3 6000 | tee $L/find_cap_L3.log || { say "find_cap FAILED"; exit 1; }
  # L2's caps: L3r's realized shares of the attributes the recipe matches (classical-only, ex falso, reductio,
  # nested negation, nested negation without reuse), read from L3's final select_report.json
  CAPS_L2=$(timeout 300 $PY - <<'EOF'
import json
r = json.load(open("/scratch/cowifornia/nd-rl/think/runs/pools/curriculum/L3/work/select_report.json"))
sh = r["shares_selected"]
print(json.dumps({k: round(float(sh[k]), 4) for k in ("classical", "ex_falso", "reductio", "nested_neg", "nn_plain")}))
EOF
)
  say "CAPS_L2 $CAPS_L2" | tee -a $L/caps_L2.log
  timeout 3600 $N $PY -m $X.hard712.run --out $C/L2 select --n 6000 --workers 20 --caps "$CAPS_L2" > $L/select_L2.log 2>&1 \
    || { say "select L2 FAILED"; exit 1; }
  for b in L2 L3; do
    timeout 1800 $N $PY -m $X.hard712.run --out $C/$b split --n_test 1000 --n_dev 0 --prefix c$b \
      --source curriculum_$b > $L/split_$b.log 2>&1 || { say "split $b FAILED"; exit 1; }
    rm -f $C/$b/dev_500.jsonl $C/$b/dev_500.keys.jsonl
  done
  LEAN_GATE_LOG=$C/work/lean_gate_report.jsonl timeout 7200 $N $PY -m $X.curriculum.report --root $C \
    > $L/report.log 2>&1
  say "phase 4 done" ;;
esac
