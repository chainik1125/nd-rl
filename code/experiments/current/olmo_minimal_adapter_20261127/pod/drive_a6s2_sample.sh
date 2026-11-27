#!/bin/bash
# Amendment 6(a) step 2 + 6(c), with Amendment 8 (corner-A replicate, audit fixes T1-T3, T5, T10), at dpo, on ONE fresh
# H200-class pod. SERRANO side; the tracked copy is the one that runs (audit T12). Run from the working folder:
#   nice -n 10 timeout 11400 bash pod/drive_a6s2_sample.sh <alias> > $V2/logs/drive_a6s2_<alias>.log 2>&1
# Steps:
#   0. checks (fresh alias, every input cell DONE on serrano), stage both batches before the pod does anything
#   1. setup (about 10 min)
#   2. dev batch batch_dev_dpo: R-own, own lr1e-3, own 3ep on the 32 own dev prompts x 4, seed 660007
#      (pod caps TO_GATE 900, TO_SAMPLE 1500; wait deadline 2700 s)
#   3. fetch (must read FETCH_OK) -> dev_check.py -> winner (eligible = 4 draws on every prompt; highest pass count; tie
#      to the earlier-listed). Any failure here only means "no tuned cell": B, C and A' still run (T3).
#   4. reach batch batch_a6s2: B, C, A' (+ the winner if it is not R-own's recipe); 48 reach prompts x 8, seed 660005
#      (pod caps TO_GATE 900, TO_SAMPLE 7200; wait deadline 8400 s)
#   5. fetch (must read FETCH_OK) -> POD READY TO DELETE (only then; T2). Then, on serrano: bash a6s2_analyze.sh <alias>
# Budget: outer timeout 11,400 s = 3 h 10 min, about $16.8 at $5.29/h, under the $18 cap. Expected about 2.4-2.7 h.
set -uo pipefail
A=${1:?alias of a FRESH H200-class pod}
CK=dpo
X=$(cd "$(dirname "$0")/.." && pwd)
V2=${V2:-/scratch/cowifornia/nd-rl/olmo_elicit/v2}
PY=${PY:-/scratch/cowifornia/venvs/olmo-edl/bin/python}
cd "$X" || exit 2
# shellcheck source=drive_lib.sh
. pod/drive_lib.sh
log "drive_a6s2_sample.sh on $A; code at $(git rev-parse --short HEAD), $(git status --porcelain -- . | wc -l) uncommitted paths in the folder"

R4=$V2/runs_rp-om4; R3=$V2/runs_rp-om3; R2=$V2/runs_rp-om2
B=dpo__final25__A16__n64__r1p
C=dpo__final__A16__n64__r1p_s523373504
AP=dpo__final25__A16__n64__r1p_s523373504  # A': our trainer's replicate of the 25 Nov adapter (step-1 cell)
ROWN=dpo__own__A16__n64__r1p
for x in "$R4:$B" "$R4:$C" "$R3:$AP" "$R2:$ROWN" "$R4:dpo__own__A16__n64__lr1e-3" "$R4:dpo__own__A16__n64__3ep"; do
  test -f "${x%%:*}/dpo/train/cells/${x#*:}/DONE" || { log "STOP: ${x#*:} not DONE under ${x%%:*}"; exit 3; }
done
fresh_alias

# ---- 0. stage both batches on serrano first
DB=$V2/stage/batch_dev_dpo
RB=$V2/stage/batch_a6s2
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$R2" --ckpt dpo --cells $ROWN --stage "$DB" --force 2>&1 | tail -n 1
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$R4" --ckpt dpo --cells dpo__own__A16__n64__lr1e-3,dpo__own__A16__n64__3ep --stage "$DB" --force 2>&1 | tail -n 2
[ "$(staged "$DB")" = 3 ] || { log "STOP: dev batch staged $(staged "$DB") of 3"; exit 3; }
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$R4" --ckpt dpo --cells $B,$C --stage "$RB" 2>&1 | tail -n 2
OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$R3" --ckpt dpo --cells $AP --stage "$RB" 2>&1 | tail -n 1
[ "$(staged "$RB")" = 3 ] || { log "STOP: reach batch staged $(staged "$RB") of 3 (B, C, A')"; exit 3; }

# ---- 1. setup
step "setup" timeout 1800 bash pod/mini_pod.sh setup "$A"

# ---- 2-3. 6(c) dev batch and the winner
step "dev sample start" env TO_GATE=900 TO_SAMPLE=1500 timeout 900 bash pod/mini_pod.sh sample "$A" dpo "$DB" \
  --seed 660007 --pids "$V2/stage/pids_dev_dpo_own.txt"
wait_stage dpo sample_batch_dev_dpo 2700
dws=$?
W=$ROWN
DEVF=0
devcheck() {
  local S=$V2/runs_$A/dpo/sample/batch_dev_dpo
  OMP_NUM_THREADS=3 nice -n 10 timeout 1800 "$PY" dev_check.py --ckpt dpo --pool "$V2/pools/dpo/own" \
    --cells "$ROWN=$S/$ROWN" "dpo__own__A16__n64__lr1e-3=$S/dpo__own__A16__n64__lr1e-3" \
    "dpo__own__A16__n64__3ep=$S/dpo__own__A16__n64__3ep" \
    --stage "$DB" --gate "$V2/runs_$A/dpo/gate/batch_dev_dpo.json" --out "$V2/a6/dev_check_dpo.json" 2>&1 | tail -n 5
  return "${PIPESTATUS[0]}"
}
if fetch_ok dpo; then
  DEVF=1
  if devcheck; then
    W=$(python3 -c "import json; print(json.load(open('$V2/a6/dev_check_dpo.json'))['winner'])")
  else
    log "dev_check gave no winner (dev stage rc $dws): no tuned cell; B, C and A' still run"
  fi
else
  log "dev fetch not OK: no dev_check now and no tuned cell; the final fetch re-syncs the dev batch"
fi
log "6(c) winner: $W (dev stage rc $dws)"

# ---- 4. step-2 reach batch
want=3
if [ "$W" != $ROWN ]; then
  OMP_NUM_THREADS=1 python3 stage_cells.py --runs "$R4" --ckpt dpo --cells "$W" --stage "$RB" --force 2>&1 | tail -n 1
  want=4
fi
n=$(staged "$RB")
if [ "$n" != $want ]; then
  if [ "$want" = 4 ] && [ "$n" = 3 ]; then log "the tuned cell $W was refused by stage_cells: continuing with B, C, A'"
  else log "STOP: reach batch staged $n of $want"; exit 3; fi
fi
step "reach sample start" env TO_GATE=900 TO_SAMPLE=7200 timeout 900 bash pod/mini_pod.sh sample "$A" dpo "$RB" --seed 660005
wait_stage dpo sample_batch_a6s2 8400
rws=$?

# ---- 5. final fetch
fetch_ok dpo || { log "STOP: final fetch not OK; keep $A until fetched"; exit 6; }
if [ $DEVF = 0 ]; then devcheck || log "dev_check after the final fetch: no winner"; fi
[ $rws = 0 ] || { log "reach batch did not end DONE (rc $rws); everything sampled is fetched"; exit 7; }
READY=1
log "next, on serrano (CPU): bash a6s2_analyze.sh $A"
