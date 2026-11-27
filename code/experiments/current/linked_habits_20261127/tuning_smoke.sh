#!/usr/bin/env bash
# CPU smoke of the Stage-2 tuning CLIs after PREREG Amendment 3 (TUNING.md, "Smokes"), on the real base (the Nov 14
# authors_half base, trained on the no-notes proofs) and reiteration twins of l19's `ta_proof` (lh_tune.py toy
# --habit reit: lh_twin, the A-form member preferred). One run after another, one thread each, `nice -n 5`, pinned
# to one core (CORE, default 6), every run under `timeout` (serrano's load limit of 2026-11-27: at most 3 cores):
#   DPO seed 0; swap-random seed 0 and seed 1 (one orientation, --swap-seed 0; seed 1 only 3 steps);
#   the reward model (fit unless RM exists) and its probe on c24 twins of other habits;
#   SFT on the preferred members, then on the sampled members.
# Writes LH/smoke_tuning/v2/{toy,probe.jsonl,probe.json,dpo_s0,dposwap_s0,dposwap_s1,sft_pref,sft_sampled,logs}.
# Refuses if v2/dpo_s0 exists.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
LH=${LH:-/scratch/cowifornia/nd-rl/linked_habits}
S=$LH/smoke_tuning/v2
RM=${RM:-$LH/smoke_tuning/rm_reit.pt}
BASE=${BASE:-/scratch/cowifornia/nd-rl/two_authors/bases/stage1_authors_half_s0.pt}
STEPS=${STEPS:-6}
SNAPS=${SNAPS:-3,6}
PROBE_N=${PROBE_N:-20}
BATCH=${BATCH:-8}
CORE=${CORE:-6}
mkdir -p "$S/logs"
[ -e "$S/dpo_s0" ] && { echo "$S/dpo_s0 exists"; exit 1; }
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=(nice -n 5 taskset -c "$CORE" timeout 3600 "$PY")
T="$S/toy"
[ -f "$T/sft_sampled.jsonl" ] || "${RUN[@]}" "$HERE/lh_tune.py" toy --habit reit --out "$T" --n 160 --heldout 40 \
    --kl 64
[ -f "$S/probe.jsonl" ] || "${RUN[@]}" "$HERE/lh_tune.py" probe-toy --out "$S/probe.jsonl" --n "$PROBE_N" \
    --habits note_atom,note_vcc,note_~a,conj,disjunct,halves,restate
COMMON=(--base "$BASE" --heldout "$T/pairs_held.jsonl" --device cpu --threads 1 --snap-steps "$SNAPS"
        --eval-every 3 --kl-states "$T/kl_states.jsonl" --kl-n 32 --batch "$BATCH")
step() {  # name, then the command's arguments; a failure is logged and the next run goes on
  local name=$1; shift
  local rc=0
  "${RUN[@]}" "$@" > "$S/logs/$name.log" 2>&1 || rc=$?
  echo "$name exit $rc" | tee -a "$S/logs/$name.log"
}
step dpo_s0 "$HERE/lh_dpo.py" "${COMMON[@]}" --max-steps "$STEPS" --pairs "$T/pairs_train.jsonl" \
    --out "$S/dpo_s0" --seed 0
step dposwap_s0 "$HERE/lh_dpo.py" "${COMMON[@]}" --max-steps "$STEPS" --pairs "$T/pairs_train.jsonl" \
    --out "$S/dposwap_s0" --seed 0 --swap-random --swap-seed 0
step dposwap_s1 "$HERE/lh_dpo.py" "${COMMON[@]}" --max-steps 3 --pairs "$T/pairs_train.jsonl" \
    --out "$S/dposwap_s1" --seed 1 --swap-random --swap-seed 0
if [ ! -f "$RM" ] && [ "${RM_FIT:-1}" = 1 ]; then
  step rm_fit "$HERE/lh_rm.py" fit --base "$BASE" --pairs "$T/pairs_train.jsonl" --heldout "$T/pairs_held.jsonl" \
      --out "$RM" --device cpu --threads 1 --batch "$BATCH" --max-steps 40 --eval-every 20
fi
PRM=$RM
[ -f "$PRM" ] || PRM="$RM.failed"  # the probe never gates: on a model that missed the bar it still runs
if [ -f "$PRM" ]; then
  step rm_probe "$HERE/lh_rm.py" probe --rm "$PRM" --pairs "$S/probe.jsonl" --out "$S/probe.json" --device cpu \
      --threads 1 --reference "$T/pairs_held.jsonl"
fi
for f in pref sampled; do
  step "sft_$f" "$HERE/lh_sft.py" "${COMMON[@]}" --max-steps "$STEPS" --data "$T/sft_$f.jsonl" \
      --out "$S/sft_$f" --seed 0
done
