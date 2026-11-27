#!/usr/bin/env bash
# CPU smoke of the Stage-2 tuning CLIs after PREREG Amendment 7 (TUNING.md, "Smokes"), on the real base (the Nov 14
# authors_half base, trained on the no-notes proofs) and reiteration twins of l19's `ta_proof` (lh_tune.py toy
# --habit reit: lh_twin, the A-form member preferred). One run after another, one thread each, `nice -n 5`, pinned
# to one core (CORE, default 6), every run under `timeout` (serrano's load limit of 2026-11-27: at most 3 cores).
# Every trainer runs on its new defaults (lr 3e-5, the grid 5 ... 400 cut at --max-steps):
#   DPO seed 0 (the step-0 check on the real base); swap-random seeds 0 and 1 (swap seed = run seed, so two
#   orientations; every batch half swapped within each sampled side; seed 1 only 3 steps);
#   the reward model's probe (RM must exist: the v2 smoke's rm_reit.pt) on c24 twins of other habits plus toy
#   probe:length rows, --min-n lowered to the smoke's 20 twins per habit;
#   SFT on the preferred members, then on the sampled members, both divided by --norm-file = the sampled file.
# Reuses the v2 toy pairs and probe twins when they are there (TOY, PROBE_TWINS). Writes LH/smoke_tuning/v3/{probe.jsonl,
# probe.json,dpo_s0,dposwap_s0,dposwap_s1,sft_pref,sft_sampled,logs}. Refuses if v3/dpo_s0 exists.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
LH=${LH:-/scratch/cowifornia/nd-rl/linked_habits}
S=$LH/smoke_tuning/v3
RM=${RM:-$LH/smoke_tuning/rm_reit.pt}
BASE=${BASE:-/scratch/cowifornia/nd-rl/two_authors/bases/stage1_authors_half_s0.pt}
STEPS=${STEPS:-6}
SFT_STEPS=${SFT_STEPS:-3}
PROBE_N=${PROBE_N:-20}
BATCH=${BATCH:-8}
CORE=${CORE:-6}
T=${TOY:-$LH/smoke_tuning/v2/toy}
PT=${PROBE_TWINS:-$LH/smoke_tuning/v2/probe.jsonl}
mkdir -p "$S/logs"
[ -e "$S/dpo_s0" ] && { echo "$S/dpo_s0 exists"; exit 1; }
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=(nice -n 5 taskset -c "$CORE" timeout 3600 "$PY")
[ -f "$T/sft_sampled.jsonl" ] || "${RUN[@]}" "$HERE/lh_tune.py" toy --habit reit --out "$T" --n 160 --heldout 40 \
    --kl 64
[ -f "$PT" ] || "${RUN[@]}" "$HERE/lh_tune.py" probe-toy --out "$PT" --n "$PROBE_N" \
    --habits note_atom,note_vcc,note_~a,conj,disjunct,halves,restate
if [ ! -f "$S/probe.jsonl" ]; then
  "${RUN[@]}" "$HERE/lh_tune.py" probe-toy --out "$S/probe_length.jsonl" --habits "" --length-n "$PROBE_N"
  cat "$PT" "$S/probe_length.jsonl" > "$S/probe.jsonl"
fi
COMMON=(--base "$BASE" --heldout "$T/pairs_held.jsonl" --device cpu --threads 1 --eval-every 3
        --kl-states "$T/kl_states.jsonl" --kl-n 32 --batch "$BATCH")
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
    --out "$S/dposwap_s1" --seed 1 --swap-random --swap-seed 1
PRM=$RM
[ -f "$PRM" ] || PRM="$RM.failed"  # the probe never gates: on a model that missed the bar it still runs
if [ -f "$PRM" ]; then
  step rm_probe "$HERE/lh_rm.py" probe --rm "$PRM" --pairs "$S/probe.jsonl" --out "$S/probe.json" --device cpu \
      --threads 1 --reference "$T/pairs_held.jsonl" --min-n "$PROBE_N"
fi
for f in pref sampled; do
  step "sft_$f" "$HERE/lh_sft.py" "${COMMON[@]}" --max-steps "$SFT_STEPS" --data "$T/sft_$f.jsonl" \
      --out "$S/sft_$f" --seed 0 --norm-file "$T/sft_sampled.jsonl"
done
