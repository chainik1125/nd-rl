#!/usr/bin/env bash
# Sequential queue for the fable_rl final test (see TEST_PLAN.md). Run on serrano. Log: run/queue_fable.log
set -u
cd /scratch/cowifornia/nd-rl/fable_rl/testside
F=/scratch/cowifornia/nd-rl/fable_rl/final
Q=run/queue_fable.log; mkdir -p run
step() {  # step NAME env... -- args to final_test.sh
  local name=$1; shift
  if grep -q "^DONE $name " $Q 2>/dev/null; then return 0; fi
  echo "START $name $(date -u +%FT%TZ)" >> $Q
  env "$@" > run/queue_$name.out 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $name rc=$rc $(date -u +%FT%TZ)" >> $Q; echo "QUEUE_STOPPED $name" >> $Q; exit $rc; fi
  echo "DONE $name $(date -u +%FT%TZ)" >> $Q
}
for s in 0 1 2; do
  step fable_full_s$s SOLVER_MODE=full SOLVER_MAX_SECONDS=0 SETS=t832,tb72 TIMEOUT_S=7200 ./final_test.sh $F/solver $F/final_s$s.pt fable_full_s$s $s
done
echo "PHASE full_short done $(date -u +%FT%TZ)" >> $Q
for s in 0 1 2; do
  step fable_plain_s$s SOLVER_MODE=plain SOLVER_MAX_SECONDS=0 TIMEOUT_S=7200 ./final_test.sh $F/solver $F/final_s$s.pt fable_plain_s$s $s
done
step fable_ablplain_s0 SOLVER_MODE=plain SOLVER_MAX_SECONDS=0 TIMEOUT_S=7200 ./final_test.sh $F/solver $F/final_abl.pt fable_ablplain_s0 0
echo "PHASE plain done $(date -u +%FT%TZ)" >> $Q
for s in 0 1 2; do
  step fable_fullL_s$s SOLVER_MODE=full SOLVER_MAX_SECONDS=0 K=8 SETS=t1320,t2130 FORCE_LONG=fb TIMEOUT_S=14400 ./final_test.sh $F/solver $F/final_s$s.pt fable_fullL_s$s $s
done
echo "PHASE full_long done $(date -u +%FT%TZ)" >> $Q
echo "QUEUE_FINISHED $(date -u +%FT%TZ)" >> $Q
