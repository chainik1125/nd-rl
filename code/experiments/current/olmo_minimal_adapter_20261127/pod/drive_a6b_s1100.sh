#!/bin/bash
# Amendment 6(b) at s1100 (as 5(b), 16 reach draws), with the audit's T1-T3 and T10 fixes. SERRANO side; NOT run yet: it
# runs only after 6(a) step 2 has been read (PREREG 6(b) item 2), Leon has seen the plan and the coordinator says go.
#   MODE=train  <alias>   fresh H100 (or H200): setup, download s1100 (64.5 GB), train plan_s1100_a6b (own, unsel, final
#                         at 3e-4 x 64; own 1e-3 x 64; own 3e-4 x 192) [+ plan_s1100_a6b_seed2 if SEED2=1], fetch
#   MODE=sample <alias>   fresh H200 class, after MODE=train has fetched (TRAIN_ALIAS=<its alias>): setup, dev batch
#                         (seed 660012, 32 x 4) -> dev_check winner, prompt control plain:answeronly (48 x 8, seed 651100),
#                         reach batch (final, own, unsel [+ winner if not r1p] [+ seed-2 own and final]; 48 x 16, seed
#                         660011), fetch
#   MODE=all    <alias>   both on one pod
# SEED2=1 only if 6(a) reads "seed" or "both" (6(b) item 3; about $7 more).
#   MODE=train nice -n 10 timeout 6000  bash pod/drive_a6b_s1100.sh <alias>   (about 1 h 10 min at most)
#   MODE=sample TRAIN_ALIAS=<a> nice -n 10 timeout <OUTER> bash pod/drive_a6b_s1100.sh <alias>
# Caps (2026-11-27 4:12 pm PT) are sized from rp-om5's measurements: about 7k tokens per draw at about 1,450 tok/s,
# i.e. about 1,000 draws per 80 min. Pod-side caps sit about 1.5x the expected finish, so they
# never fire at normal throughput. Money is bounded by the wait deadlines (an OVERRUN stops our chain) and by the outer
# timeout <OUTER>, set from the approved cap minus about 20 min. When the outer timeout ends the driver, the chain on the
# pod keeps running until its own cap, and the EXIT line asks the coordinator to decide (delete the pod).
set -uo pipefail
A=${1:?alias of a FRESH pod}
MODE=${MODE:?MODE=train|sample|all}
SEED2=${SEED2:-0}
CK=s1100
X=$(cd "$(dirname "$0")/.." && pwd)
V2=${V2:-/scratch/cowifornia/nd-rl/olmo_elicit/v2}
PY=${PY:-/scratch/cowifornia/venvs/olmo-edl/bin/python}
cd "$X" || exit 2
# shellcheck source=drive_lib.sh
. pod/drive_lib.sh
log "drive_a6b_s1100.sh MODE=$MODE SEED2=$SEED2 on $A; code at $(git rev-parse --short HEAD), $(git status --porcelain -- . | wc -l) uncommitted paths"
fresh_alias
OWN=s1100__own__A16__n64__r1p
step "setup" timeout 1800 bash pod/mini_pod.sh setup "$A"

if [ "$MODE" = train ] || [ "$MODE" = all ]; then
  step "train start" env TO_TRAIN=5400 timeout 900 bash pod/mini_pod.sh train "$A" s1100 --plan "$V2/plan_s1100_a6b.json" \
    --train-args "--ce-checkpoint --max-memory-gb 78"
  wait_stage s1100 train_plan_s1100_a6b 7200 || { log "STOP: training did not end DONE"; fetch_ok s1100; exit 4; }
  if [ "$SEED2" = 1 ]; then
    step "seed-2 train start" env TO_TRAIN=2400 timeout 900 bash pod/mini_pod.sh train "$A" s1100 \
      --plan "$V2/plan_s1100_a6b_seed2.json" --train-args "--ce-checkpoint --max-memory-gb 78"
    wait_stage s1100 train_plan_s1100_a6b_seed2 3000 || { log "STOP: seed-2 training did not end DONE"; fetch_ok s1100; exit 4; }
  fi
  fetch_ok s1100 || { log "STOP: training fetch not OK; keep $A until fetched"; exit 6; }
  if [ "$MODE" = train ]; then READY=1; exit 0; fi
fi

RT=$V2/runs_${TRAIN_ALIAS:-$A}
for c in $OWN s1100__unsel__A16__n64__r1p s1100__final__A16__n64__r1p s1100__own__A16__n64__lr1e-3 s1100__own__A16__n64__3ep; do
  test -f "$RT/s1100/train/cells/$c/DONE" || { log "STOP: $c not DONE under $RT"; exit 3; }
done

# ---- dev batch (tuned-own rule, 5(b))
DB=$V2/stage/batch_dev_s1100
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells $OWN,s1100__own__A16__n64__lr1e-3,s1100__own__A16__n64__3ep --stage "$DB" --force 2>&1 | tail -n 3
[ "$(staged "$DB")" = 3 ] || { log "STOP: dev batch staged $(staged "$DB") of 3"; exit 3; }
step "dev sample start" env TO_GATE=900 TO_SAMPLE=4200 timeout 900 bash pod/mini_pod.sh sample "$A" s1100 "$DB" \
  --seed 660012 --pids "$V2/stage/pids_dev_s1100_own.txt"
wait_stage s1100 sample_batch_dev_s1100 6000  # 384 draws: about 31 min expected (+ download, gate)
W=$OWN
if fetch_ok s1100; then
  S=$V2/runs_$A/s1100/sample/batch_dev_s1100
  OMP_NUM_THREADS=3 nice -n 10 timeout 1800 "$PY" dev_check.py --ckpt s1100 --pool "$V2/pools/s1100/own" \
    --cells "$OWN=$S/$OWN" "s1100__own__A16__n64__lr1e-3=$S/s1100__own__A16__n64__lr1e-3" \
    "s1100__own__A16__n64__3ep=$S/s1100__own__A16__n64__3ep" \
    --stage "$DB" --gate "$V2/runs_$A/s1100/gate/batch_dev_s1100.json" --out "$V2/a6/dev_check_s1100.json" 2>&1 | tail -n 5
  if [ "${PIPESTATUS[0]}" = 0 ]; then
    W=$(python3 -c "import json; print(json.load(open('$V2/a6/dev_check_s1100.json'))['winner'])")
  else log "dev_check gave no winner: no tuned cell"; fi
else
  log "dev fetch not OK: no tuned cell"
fi
log "tuned-own winner at s1100: $W"

# ---- prompt control (8 draws)
step "prompt control start" env TO_SAMPLE=4200 timeout 900 bash pod/mini_pod.sh plain "$A" s1100 answeronly \
  --problems "$V2/stage/problems_answeronly.jsonl" --pids "$V2/stage/pids_answeronly.txt" --seed 651100
wait_stage s1100 plain_answeronly 5400 ||  # 384 draws: about 31 min expected log "prompt control did not end DONE: reported as such"

# ---- reach batch (16 draws)
RB=$V2/stage/batch_a6b_s1100
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells s1100__final__A16__n64__r1p --stage "$RB" 2>&1 | tail -n 1
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells $OWN,s1100__unsel__A16__n64__r1p --stage "$RB" --force 2>&1 | tail -n 2
want=3
if [ "$W" != $OWN ]; then
  OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells "$W" --stage "$RB" --force 2>&1 | tail -n 1
  want=$((want + 1))
fi
if [ "$SEED2" = 1 ]; then
  OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells s1100__final__A16__n64__r1p_s523373504 --stage "$RB" 2>&1 | tail -n 1
  OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$RT" --ckpt s1100 --cells s1100__own__A16__n64__r1p_s523373504 --stage "$RB" --force 2>&1 | tail -n 1
  want=$((want + 2))
fi
[ "$(staged "$RB")" = $want ] || { log "STOP: reach batch staged $(staged "$RB") of $want"; fetch_ok s1100; exit 3; }
TS=$((want * 5600))  # 768 draws x 7k tokens / 1,450 tok/s = about 3,700 s per cell expected; cap 1.5x
step "reach sample start" env TO_GATE=900 TO_SAMPLE=$TS timeout 900 bash pod/mini_pod.sh sample "$A" s1100 "$RB" \
  --seed 660011 --pids "$V2/stage/pids_L16.txt"
wait_stage s1100 sample_batch_a6b_s1100 $((TS + 1200))
rws=$?
fetch_ok s1100 || { log "STOP: final fetch not OK; keep $A until fetched"; exit 6; }
[ $rws = 0 ] || { log "reach batch did not end DONE (rc $rws); everything sampled is fetched"; exit 7; }
READY=1
