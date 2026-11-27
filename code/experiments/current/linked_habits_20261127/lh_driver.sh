#!/usr/bin/env bash
# lh_driver.sh: the whole run of ONE corpus on ONE pod (linked habits version 2, SPEC2.md "Pods"), run ON THE POD
# (started detached by `lh_pod.sh start`).
#
#   lh_driver.sh <corpus>          corpus: one of lh_core.CORPORA (c24 | c9 | c4 | c0)
#
# Each stage is a detached lh_launch.py job (lock, every step under `timeout`, progress log LH/logs/<job>_<UTC>.log)
# that this driver waits for with `lh_launch.py wait` (bounded):
#  Stage 1
#   1 pretrain   lh_launch.py pretrain --corpus B --seed 0 for every base B of the pod: C, and on the c0 pod also c0n
#                (lh_core.bases_on_pod; EXTRA env replaces the list) (skipped when LH/bases/stage1_B_s0.pt exists)
#   2 measure    lh_launch.py measure --base B_s0 --skip_done    (k 8, tags A/B/none, marker_plain + route_unseen, step
#                limit 51; then Dan's strict 106 at k 64) for every base B; on the NOV14 pod (NOV14=1; lh_pod.sh start
#                sets it for lh_core.NOV14_POD) also authors_half_s0 and control_half_s0 (the performance control)
#   3 gate       writes LH/pod/C.GATE_READY and WAITS for LH/pod/C.GO (run stage 2) or LH/pod/C.NOGO (stop here);
#                the orchestrator fetches, runs lh_gate.py on serrano and `lh_pod.sh go|nogo`
#  Stage 2
#   4 prep2      lh_launch.py prep2 --corpus C   (anchor states, the base's proofs on Dan's RL pool, the pairs per rater;
#                skipped when every pair summary exists)
#   5 stage2     ND_GUARD=off lh_launch.py stage2 --corpus C --parallel $PARALLEL (2): one job holding the GPU, at most
#                two GPU tasks at once: rl_C_rlhfreit_s0 (reward model fit and probe, GRPO with its measurements),
#                rl_C_anc_s0, the offline runs (reit: dpo / dposwap _s0 and _s1, sft / sftsam _s0; atom on c24 and c0:
#                dpo / dposwap _s0; rated runs probed), then full_dpo / full_sft_C_<R>_s<S> (the measurement at the
#                bracketing snapshots and the last).
#                Tasks already finished are not passed again. A task that did not finish is recorded in
#                LH/pod/C.FAILED; the others still run.
#   6 done       writes LH/pod/C.DONE
# Heartbeat: LH/pod/C.heartbeat.json (UTC, stage, job, the job log's last line), rewritten every $POLL seconds while a
# job runs or the gate is awaited. Driver log: LH/pod/C.driver.log (lh_pod.sh start redirects there). A second driver for
# the same corpus refuses (lock LH/pod/C.driver.lock). The driver never creates or deletes a pod.
# Environment: LH, PY, GPU (0), POLL (60 s), GATE_WAIT_S (43200 = 12 h), MAX_H_PRETRAIN (4), MAX_H_MEASURE (3.5),
# MAX_H_PREP (6.5), MAX_H_STAGE2 (15) (each wait 0.5 h or more above its job's own cap), PARALLEL (2), NOV14 (0/1),
# LH_CORPORA; LH_LAUNCH, LH_TASKS and EXTRA (the pod's extra bases) replace the launcher, the task list and the base
# list (tests).
set -uo pipefail
c=${1:?usage: lh_driver.sh <corpus>}
[[ $c =~ ^[a-z][a-z0-9]*$ ]] || { echo "bad corpus name"; exit 2; }
W=$(cd "$(dirname "$0")" && pwd)
LH=${LH:-/scratch/cowifornia/nd-rl/linked_habits}
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
read -r -a LAUNCH <<< "${LH_LAUNCH:-$PY $W/lh_launch.py}"
GPU=${GPU:-0}
POLL=${POLL:-60}
GATE_WAIT_S=${GATE_WAIT_S:-43200}
POD=$LH/pod
HB=$POD/$c.heartbeat.json
export LH
mkdir -p "$POD"
utc() { date -u +%FT%TZ; }
say() { echo "[$(utc)] $c: $*"; }
hb() {  # hb <stage> [job]
  printf '{"utc": "%s", "corpus": "%s", "stage": "%s", "job": "%s", "pid": %d}\n' "$(utc)" "$c" "$1" "${2:-}" "$$" > "$HB.tmp" && mv "$HB.tmp" "$HB"
}
if ! mkdir "$POD/$c.driver.lock" 2>/dev/null; then
  say "another driver holds $POD/$c.driver.lock ($(cat "$POD/$c.driver.lock/pid" 2>/dev/null)): not running"; exit 3
fi
echo "$(hostname) $$ $(utc)" > "$POD/$c.driver.lock/pid"
trap 'rm -f "$POD/$c.driver.lock/pid"; rmdir "$POD/$c.driver.lock" 2>/dev/null' EXIT
die() { say "STOP: $*"; echo "$(utc) $*" >> "$POD/$c.FAILED"; hb "failed: $*"; exit 1; }

run_job() {  # run_job <stage> <job name> <max hours> <launcher args...>: launch detached, wait for its end, return its rc
  local stage=$1 job=$2 maxh=$3; shift 3
  hb "$stage" "$job"
  say "$stage: ${LAUNCH[*]} $* --gpu $GPU"
  "${LAUNCH[@]}" "$@" --gpu "$GPU" || { say "$stage: the launch refused or failed"; return 1; }
  "${LAUNCH[@]}" wait --job "$job" --max_h "$maxh" --every "$POLL" --heartbeat "$HB" --stage "$stage"
}
measured() {  # measured <base id>: the base's six summaries and its strict-106 rows exist
  local n; n=$(ls "$LH/measure/$1/"*.summary.json 2>/dev/null | wc -l)
  (( n >= 6 )) && [[ -s $LH/measure/$1/strict106__none.jsonl ]]
}

say "driver start on $(hostname), GPU $GPU, LH $LH"
# ---------------------------------------------------------------- stage 1
if [[ -n ${EXTRA+x} ]]; then pod_bases="$c $EXTRA"; else
  pod_bases=$(cd "$W" && "$PY" -c "import lh_core as C; print(' '.join(C.bases_on_pod('$c')))") || die "base list"
fi
bases=()
for b in $pod_bases; do
  if [[ -s $LH/bases/stage1_${b}_s0.pt && -s $LH/bases/stage1_${b}_s0.tag.json ]]; then
    say "pretrain $b: base exists, skipped"
  else
    run_job "pretrain $b" "pretrain_${b}_s0" "${MAX_H_PRETRAIN:-4}" pretrain --corpus "$b" --seed 0 || die "pretrain $b"
  fi
  bases+=("${b}_s0")
done
if [[ ${NOV14:-0} == 1 ]]; then bases+=(authors_half_s0 control_half_s0); fi
for b in "${bases[@]}"; do
  if measured "$b"; then say "measure $b: done before, skipped"
  else run_job "measure $b" "measure_$b" "${MAX_H_MEASURE:-3.5}" measure --base "$b" --skip_done || die "measure $b"; fi
done
utc > "$POD/$c.GATE_READY"
say "gate: wrote $POD/$c.GATE_READY; waiting up to ${GATE_WAIT_S} s for $POD/$c.GO or $POD/$c.NOGO"
t0=$SECONDS
while [[ ! -e $POD/$c.GO && ! -e $POD/$c.NOGO ]]; do
  hb "gate: waiting for GO or NOGO" ""
  if (( SECONDS - t0 > GATE_WAIT_S )); then die "gate: no GO or NOGO within ${GATE_WAIT_S} s"; fi
  sleep "$POLL"
done
if [[ -e $POD/$c.NOGO ]]; then
  say "gate: NOGO; stage 2 does not run"; hb "stopped at the gate (NOGO)"; utc > "$POD/$c.STOPPED"; exit 0
fi
say "gate: GO"
# ---------------------------------------------------------------- stage 2
if [[ -n ${LH_TASKS:-} ]]; then tasks=$LH_TASKS; raters=${LH_RATERS:-reit}; else
  tasks=$(cd "$W" && "$PY" -c "import lh_core as C; print(' '.join(C.tasks_of('$c')))") || die "task list"
  raters=$(cd "$W" && "$PY" -c "import lh_core as C; print(' '.join(C.raters_of('$c')))") || die "rater list"
fi
need_prep=0
for r in $raters; do [[ -s $LH/pairs/${c}_s0/$r/summary.json ]] || need_prep=1; done
[[ -s $LH/data/anchor_$c.jsonl ]] || need_prep=1
if (( need_prep )); then
  run_job prep2 "prep2_${c}_s0" "${MAX_H_PREP:-6.5}" prep2 --corpus "$c" || die "prep2"
else
  say "prep2: anchor and pairs exist, skipped"
fi
not_run() {  # not_run <task>: decided not to run (the rlhf line whose reward model missed 0.90): never retried
  [[ -s $LH/arms/$1.not_run.json ]]
}
finished() {  # finished <task>: a GRPO line whose seg1/end.json says done; a rated offline run (dpo, sft) with done.json
  local t=$1   # and its probe; a no-preference run (dposwap, sftsam) with done.json; a full measurement whose
  #              LH/arms/<task>.json says done
  if [[ $t == rl_* ]]; then
    local how; how=$(sed -n 's/.*"how": "\([a-z]*\)".*/\1/p' "$LH/runs/$t/seg1/end.json" 2>/dev/null | head -1)
    [[ $how == done || $how == rule ]]
  elif [[ $t == full_* ]]; then
    grep -q '"done": true' "$LH/arms/$t.json" 2>/dev/null
  elif [[ $t == dposwap_* || $t == sftsam_* ]]; then
    [[ -s $LH/arms/$t/done.json ]]
  else
    [[ -s $LH/arms/$t/done.json && -s $LH/arms/$t/probe/probe.json ]]
  fi
}
pending=()
for t in $tasks; do
  if finished "$t"; then say "stage2: $t finished before, skipped"
  elif not_run "$t"; then say "stage2: $t NOT RUN (decided before: $LH/arms/$t.not_run.json), skipped"
  else pending+=("$t"); fi
done
if (( ${#pending[@]} > 0 )); then
  joined=$(IFS=,; echo "${pending[*]}")
  ND_GUARD=off run_job "stage2 $joined" "stage2_${c}_s0" "${MAX_H_STAGE2:-15}" stage2 --corpus "$c" --tasks "$joined" \
    --parallel "${PARALLEL:-2}" || say "stage2: the job did not end rc 0"
fi
for t in "${pending[@]}"; do
  if finished "$t"; then say "stage2: $t finished"
  elif not_run "$t"; then say "stage2: $t NOT RUN"; echo "$(utc) stage2 $t not run: $(cat "$LH/arms/$t.not_run.json" | tr -d '\n')" >> "$POD/$c.NOT_RUN"
  else say "stage2: $t FAILED"; echo "$(utc) stage2 $t did not finish" >> "$POD/$c.FAILED"; fi
done
utc > "$POD/$c.DONE"
hb "done"
say "driver done"
