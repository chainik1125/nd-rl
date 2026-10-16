#!/usr/bin/env bash
# L4 (16-18 lines): the next band of the length curriculum, built with L3r's recipe (build_ynez.sh) so that a GRPO
# stage on it differs from L3 in proof length only. Same generator (hard712/pfgen.py), same gates (hard712/pfgates.py,
# band moved to 16-18 through PFG_MIN_L / PFG_MAX_L), same adaptive-weight values, same assemble / select / split /
# report steps, PYTHONHASHSEED=0, CPU only, nice 10, every step under `timeout`. Differences from L3r, all forced by
# the band or stricter:
#   * time limits PFG_SEARCH_SECS=60, PFG_NEC_SECS=60 (L3r: 12 and 14.5). At 12 s, seeds 0-1M gave 0 certified rows
#     (50 of 252 certified classes timed out; re-run at 240 s, 7 of them are 16-line rows, searches 14-21 s;
#     calib_L4.py, L4/pilot_12s/). The shortest-proof search grows ~1.7x per line (L3r medians 2.3 / 4.3 / 7.0 s at
#     13 / 14 / 15 lines), so 12 s at 13-15 lines is about 60 s at 16-18;
#   * sources: L3r's 4,847 hard712 `long` classes have 13-line shortest proofs, so L4 has no such reservoir. The
#     analogue is L3r generation's own `long` rejects (12 classes, all 16 lines; regen.py --source L3/gen
#     --status long); everything else is fresh generation (seeds from 0) and its gap rejects (gap 3), as for L3r;
#   * exclusion index: the curriculum index's files (hard712's index, every jsonl under runs/pools and
#     runs/eval/matched/sets, sealed_parts) with nothing omitted (charles_1002 and t2130_n1000 are in), plus L2r /
#     L3r's pools, test splits and reservoirs and curriculum_3p's files (excl.py --extra);
#   * assemble avoids L3r's and L2r's reservoirs (renaming class and curried form), so L4 shares no class with any
#     earlier band, its test split included;
#   * the mix caps are L3r's realized shares (as L2r's were). The reservoir cannot fill 6,000 at any caps, so the
#     selection is the largest one that fills at those caps without relaxing them (find_n.py), and the test split is a
#     sixth of it (L3r's 1,000 of 6,000; still written as test_1k.jsonl, the name the readers expect);
#   * generation length: rounds of 500k seeds up to L4_STOP_B (no skip) / L4_STOP_C (skip keys) / L4_STOP_D (gap 3).
#     bash build_L4.sh PHASE   (PHASE: 0 index | 1 regen-long + gen 0-1M | 2 skip keys + gen to L4_STOP_C |
#                               3 gap regen + gen to L4_STOP_D | 4 Lean, assemble, select, split, report)
set -uo pipefail   # no -e: each step's exit status is logged; phase 4 stops on a failed step
WT=${L4_WT:-/scratch/cowifornia/nd-rl/think/nd-rl-long}
R=/scratch/cowifornia/nd-rl/think/runs
C=$R/pools/curriculum
D=$C/L4
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=code.experiments.current.thinking_traces_20261005
cd $WT
export PYTHONPATH=$WT ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= PFG_MEM_GB=3.0
export PFG_SEARCH_SECS=${L4_SEARCH_SECS:-60} PFG_NEC_SECS=${L4_NEC_SECS:-60}
W=${L4_WORKERS:-20}
STOP_B=${L4_STOP_B:-8500000}; STOP_C=${L4_STOP_C:-$STOP_B}; STOP_D=${L4_STOP_D:-200000000}
HOURS_D=${L4_HOURS_D:-4}  # phase 3's generation stops after the round that crosses this many hours (or at STOP_D)
N="nice -n 10"
B16="PFG_MIN_L=16 PFG_MAX_L=18"
ADAPT="PFG_CLASSICAL_VALUE=0 PFG_EXFALSO_VALUE=0 PFG_SCARCE_BONUS=0"
IDX=$D/work/excl_index.pkl
G="$X.hard712.run --out $D/gen generate --round 500000 --chunk 10000 --index $IDX --target 999999 --hours 99"
L=$D/logs; mkdir -p $L $D/gen $D/work
say() { echo "$(date -u +%FT%TZ) $*"; }

skip_add() {  # add the certified rows of a gated file (pkey + curried / held-out keys) to skip_keys.txt
  timeout 1800 $N $PY - "$1" <<'EOF'
import json, sys
from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
D = "/scratch/cowifornia/nd-rl/think/runs/pools/curriculum/L4"
keys = {l.strip() for l in open(f"{D}/gen/skip_keys.txt") if l.strip()}
n0 = len(keys)
for l in open(sys.argv[1]):
    r = json.loads(l)
    keys |= {r["pkey"]} | HO.keys(r["prompt"])
open(f"{D}/gen/skip_keys.txt", "w").write("\n".join(sorted(keys)) + "\n")
print("skip keys", n0, "->", len(keys))
EOF
}

case ${1:?phase} in
0)
  say "phase 0: L4 exclusion index"
  ex=()
  for b in L2 L3; do ex+=($C/$b/pool.jsonl $C/$b/test_1k.jsonl $C/$b/work/gated_lean.jsonl); done
  timeout 3600 $N $PY -m $X.curriculum.excl --base $R/pools/hard712/work/excl_index.pkl --out $IDX \
    --extra $C/work/sealed_parts.jsonl "${ex[@]}" > $L/index.log 2>&1
  say "index exit $?"; tail -1 $L/index.log ;;
1)
  say "phase 1: regen-long from L3r generation (2 workers) || gen 0-1M default weights ($((W - 2)) workers)"
  ( env $B16 PFG_MAX_GAP=1 timeout 7200 $N $PY -m $X.curriculum.regen --lo 16 --hi 18 --status long \
      --source $C/L3/gen --index $IDX --out $D/regen --workers 2 > $L/regen_long.log 2>&1
    say "regen-long exit $?" ) &
  ( env $B16 timeout 14400 $N $PY -m $G --workers $((W - 2)) --stop_seed 1000000 > $L/gen_a.log 2>&1
    say "gen 0-1M (default weights) exit $?" ) &
  wait; say "phase 1 done" ;;
1b)
  say "phase 1b: gen 1M-$STOP_B adaptive, no skip ($W workers)"
  env $B16 $ADAPT timeout 86400 $N $PY -m $G --workers $W --adaptive --stop_seed $STOP_B > $L/gen_b.log 2>&1
  say "gen 1M-$STOP_B (adaptive) exit $?" ;;
2)
  say "phase 2: skip keys, gen $STOP_B-$STOP_C ($W workers)"
  timeout 1800 $N $PY - <<'EOF'
import json
from code.experiments.current.thinking_traces_20261005.curriculum import heldout as HO
C = "/scratch/cowifornia/nd-rl/think/runs/pools/curriculum"
keys = set()
for l in open(f"{C}/L4/regen/work/gated.jsonl"):
    r = json.loads(l)
    keys |= {r["pkey"]} | HO.keys(r["prompt"])
n_regen = len(keys)
for b in ("L2", "L3"):  # earlier bands' reservoirs: assemble drops their classes and curried twins anyway
    for l in open(f"{C}/{b}/work/gated_lean.jsonl"):
        r = json.loads(l)
        keys |= {r["pkey"]} | HO.keys(r["prompt"])
keys |= set(HO.heldout_index({**HO.FILES, **HO.SEALED}))
open(f"{C}/L4/gen/skip_keys.txt", "w").write("\n".join(sorted(keys)) + "\n")
print("regen keys", n_regen, "total skip keys", len(keys))
EOF
  if [ $STOP_C -gt $STOP_B ]; then  # (L4 as built: STOP_C = STOP_B, straight to the gap-3 phase)
    env $B16 $ADAPT timeout 86400 $N $PY -m $G --workers $W --adaptive --stop_seed $STOP_C \
      --skip $D/gen/skip_keys.txt > $L/gen_c.log 2>&1
    say "gen $STOP_B-$STOP_C exit $?"
  fi ;;
3)
  say "phase 3: gap regen ($W workers), gen $STOP_C-$STOP_D with gap 3 ($W workers)"
  env $B16 PFG_MAX_GAP=3 timeout 14400 $N $PY -m $X.curriculum.regen --lo 16 --hi 18 --status gap \
    --source $D/gen --index $IDX --out $D/regen_gap --workers $W > $L/regen_gap.log 2>&1  # (L4: before gen gap 3)
  say "regen_gap exit $?"
  skip_add $D/regen_gap/work/gated.jsonl
  env $B16 $ADAPT PFG_MAX_GAP=3 timeout $((HOURS_D * 3600 + 3600)) $N $PY -m $G --workers $W --adaptive \
    --stop_seed $STOP_D --hours $HOURS_D --skip $D/gen/skip_keys.txt > $L/gen_d.log 2>&1
  say "gen $STOP_C-$STOP_D (or $HOURS_D h) exit $?" ;;
4)
  say "phase 4: Lean, assemble, select, split, report"
  export LEAN_GATE_WORKERS=$W
  srcs=()
  for s in regen regen_gap gen; do
    [ -f $D/$s/work/gated.jsonl ] || continue
    timeout 7200 $N $PY -m $X.hard712.run --out $D/$s lean > $L/lean_$s.log 2>&1 \
      || { say "lean $s FAILED"; exit 1; }
    say "lean $s done"; srcs+=($s=$D/$s/work/gated_lean.jsonl)
  done
  timeout 1800 $N $PY -m $X.curriculum.assemble --band_dir $D --lo 16 --hi 18 --fresh "${srcs[@]}" \
    --avoid $C/L3/work/gated_lean.jsonl $C/L2/work/gated_lean.jsonl | tee $L/assemble.log \
    || { say "assemble FAILED"; exit 1; }
  # caps: L3r's realized shares of the attributes the recipe matches (as L2r's caps were), from L3's select_report
  CAPS=$(timeout 300 $PY - <<'EOF'
import json
r = json.load(open("/scratch/cowifornia/nd-rl/think/runs/pools/curriculum/L3/work/select_report.json"))
sh = r["shares_selected"]
print(json.dumps({k: round(float(sh[k]), 4) for k in ("classical", "ex_falso", "reductio", "nested_neg", "nn_plain")}))
EOF
)
  say "CAPS_L4 $CAPS" | tee -a $L/caps.log
  # the reservoir cannot fill 6,000: the largest selection at those caps, none of them relaxed (find_n.py)
  timeout 7200 $N $PY -m $X.curriculum.find_n $D "$CAPS" > $L/find_n.log 2>&1 || { say "find_n FAILED"; exit 1; }
  NSEL=$($PY -c "import json; print(json.load(open('$D/work/select_report.json'))['n_selected'])")
  NTEST=${L4_NTEST:-$(( (NSEL + 3) / 6 ))}  # L3r's 1,000 of 6,000
  say "selected $NSEL; test $NTEST, pool $((NSEL - NTEST))"; tail -2 $L/find_n.log
  timeout 1800 $N $PY -m $X.hard712.run --out $D split --n_test $NTEST --n_dev 0 --prefix cL4 \
    --source curriculum_L4 > $L/split.log 2>&1 || { say "split FAILED"; exit 1; }
  rm -f $D/dev_500.jsonl $D/dev_500.keys.jsonl
  LEAN_GATE_LOG=$D/work/lean_gate_report.jsonl timeout 7200 $N $PY -m $X.curriculum.report --root $C --bands L4 \
    --index $IDX --out $D > $L/report.log 2>&1
  say "report exit $?"
  timeout 1800 $N $PY -m $X.grpo_v2.preflight $C/L2/pool.jsonl $C/L3/pool.jsonl $D/pool.jsonl \
    > $L/preflight.log 2>&1
  say "preflight exit $?"; tail -1 $L/preflight.log
  timeout 1800 $N $PY -m $X.grpo_v2.preflight $C/L2/pool.jsonl $C/L3/pool.jsonl $D/test_1k.jsonl \
    > $L/preflight_test.log 2>&1  # the L4 test split against every other evaluation set
  say "preflight (L4 test split) exit $?"; tail -1 $L/preflight_test.log
  say "phase 4 done" ;;
esac
