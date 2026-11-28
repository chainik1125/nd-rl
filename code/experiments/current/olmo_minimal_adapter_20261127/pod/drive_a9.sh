#!/bin/bash
# Amendment 9 (PREREG, 2026-11-28): the rank-1 adapter and the 16- and 4-example adapters, sampled on ONE H200-class pod
# that another job also uses (one job on the GPU at a time). SERRANO side. Run from the working folder:
#   nice -n 10 timeout 11400 bash pod/drive_a9.sh <alias> > $V2/logs/drive_a9_<alias>.log 2>&1
# Steps (RUNBOOK_A9.md has the expected durations):
#   0. checks on serrano, before the pod does anything: the plan file is a9.py's plan; the two reused adapters are DONE
#      with the recorded md5; no stage/batch_a9 yet; the alias is new and the pod has no /workspace/mini; the GPU is free
#      (checked again before training, before sampling, and on the pod when each chain starts);
#      what a push would copy holds no sealed-set or credential name
#   1. setup (Dan's pinned stack; idempotent, so a pod the other job has set up is reused)
#   2. budget decision (a9.py decide --phase train): train the 16-example adapter only if training plus three cells
#      ends inside the money stop
#   3. train plan_dpo_a9 (download first if /workspace/models/dpo is not there), fetch (must read FETCH_OK). The wait
#      ends at the minute after which three cells would no longer fit; training is then stopped and step 4 goes on
#      with the two reused cells.
#   4. budget decision (a9.py decide --phase sample), taken BEFORE the engine starts: three cells, or the rank-1 and
#      4-example cells only, or nothing. No deadline is used to cut draws in a healthy run.
#   5. stage exactly the decided cells as batch_a9, gate + sample (48 reach prompts x 8, seed 660009)
#   6. final fetch (must read FETCH_OK). Then, on serrano: RUNBOOK_A9.md "After the fetch".
# env: REF2=1          also sample the reference adapter again (only if the coordinator approved it; needs A9_CAP_USD raised)
#      A9_CAP_USD=15   the money stop for this job's pod time; A9_RATE=5.29 ($/h)
#      A9_T0=<epoch s> when this job's pod time started (default: now)
#      A9_RESTAGE=1    move an existing $V2/stage/batch_a9 aside (a second attempt on a new alias)
#      A9_GPU_FREE_MIB=2000   "the GPU is free" means less than this in use, here and again on the pod when a chain starts
#      A9_OTHER_LOCK=/workspace/wi/gpu.lock   the other job's GPU lock: our chains hold it too while they run, if the file
#                      exists (never created or written). Empty = do not take it.
# Never stops or deletes a pod. On the money stop it stops only this job's chain, fetches and reports.
set -uo pipefail
A=${1:?alias of the pod (an entry of ~/.ssh/config; no /workspace/mini on the pod yet)}
CK=dpo
X=$(cd "$(dirname "$0")/.." && pwd)
V2=${V2:-/scratch/cowifornia/nd-rl/olmo_elicit/v2}
PY=${PY:-python3}  # a9.py and stage_cells.py need the standard library only
REF2=${REF2:-0}
CAP=${A9_CAP_USD:-15}
RATE=${A9_RATE:-5.29}
T0=${A9_T0:-$(date +%s)}
FREE_MIB=${A9_GPU_FREE_MIB:-2000}
OTHER_LOCK=${A9_OTHER_LOCK-/workspace/wi/gpu.lock}  # the other job's GPU lock; A9_OTHER_LOCK= (empty) leaves it alone
cd "$X" || exit 2
# shellcheck source=drive_lib.sh
. pod/drive_lib.sh

STATE=open  # open: a step failed or a fetch is missing; done: all sampled and fetched; stopped: ended early, all fetched
WHY=""
UNFETCHED=0  # 1 while a chain of this job has written files on the pod that no FETCH_OK has covered yet
stop_early() {  # stop_early <rc> <why>: "stopped" only if nothing of ours is left unfetched on the pod
  WHY=$2
  [ $UNFETCHED = 0 ] && STATE=stopped
  log "STOP: $WHY"
  exit "$1"
}
on_exit() {  # replaces drive_lib.sh's message: this pod is shared, so "ready to delete" is not this driver's call
  local rc=$?
  case $STATE in
    done) log "A9 DONE on $A: every fetch FETCH_OK. This job needs the pod no longer; the pod is shared, so the coordinator decides when it is deleted." ;;
    stopped) log "A9 STOPPED rc=$rc on $A: $WHY. Everything this job wrote on the pod so far is fetched. This driver will not use the pod again; the coordinator decides what happens next (RUNBOOK_A9.md)." ;;
    *) log "EXIT rc=$rc: pod $A NEEDS A DECISION from the coordinator: ${WHY:-a step failed or a fetch is missing}. This driver never stops or deletes a pod. Fetch by hand first: bash pod/mini_pod.sh fetch $A $CK" ;;
  esac
}
elapsed() { echo $(($(date +%s) - T0)); }
spent() { "$PY" -c "print(f'{$(elapsed) / 3600 * $RATE:.2f}')"; }
gpu_idle() {  # the other job must not be on the GPU: under $FREE_MIB MiB in use (the other job's own gate), up to 2 min of waiting
  local n i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    n=$(ssh "${SSHO[@]}" "$A" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1' 2>/dev/null)
    [ "${n:-x}" -lt "$FREE_MIB" ] 2>/dev/null && return 0
    [ "$i" = 13 ] || sleep 10
  done
  stop_early 4 "the GPU of $A is not idle (${n:-no answer} MiB in use, limit $FREE_MIB); another job may be running"
}

RANK1=$("$PY" -c "import a9; print(a9.RANK1)")
N16=$("$PY" -c "import a9; print(a9.N16)")
N4=$("$PY" -c "import a9; print(a9.N4)")
REFC=$("$PY" -c "import a9; print(a9.REF)")
SEED=$("$PY" -c "import a9; print(a9.SEED_SAMPLE)")
TARGS=$("$PY" -c "import a9; print(a9.TRAIN_ARGS)")
STOP_S=$("$PY" -c "import a9; print(int(a9.stop_s($CAP, $RATE)))")
FETCH_S=$("$PY" -c "import a9; print(int(a9.FETCH_S))")
[ -n "$RANK1" ] && [ -n "$STOP_S" ] || { log "STOP: a9.py did not load"; exit 2; }
PLAN=$V2/plan_dpo_a9.json
RB=$V2/stage/batch_a9
R1=$V2/runs_rp-om1
R2=$V2/runs_rp-om2
REFFLAG=()
[ "$REF2" = 1 ] && REFFLAG=(--ref)
log "drive_a9.sh on $A; code at $(git rev-parse --short HEAD 2>/dev/null), $(git status --porcelain -- . 2>/dev/null | wc -l) uncommitted paths in the folder; REF2=$REF2; money stop \$$CAP = $STOP_S s of pod time from $(date -u -d "@$T0" +%FT%TZ)"

# ---- 0. checks on serrano
"$PY" a9.py plan --check "$PLAN" | tail -n 1
[ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: $PLAN is not a9.py's plan (build it: python3 a9.py plan --out $PLAN)"; exit 3; }
"$PY" a9.py verify --v2 "$V2" "${REFFLAG[@]}" | tail -n 4
[ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: a reused adapter is missing or differs from its recorded md5"; exit 3; }
if [ -e "$RB" ]; then
  [ "${A9_RESTAGE:-0}" = 1 ] || { log "STOP: $RB exists (an earlier attempt?); A9_RESTAGE=1 moves it aside"; exit 3; }
  mv "$RB" "$RB.old.$(date +%s)" || exit 3
fi
mkdir -p "$V2/a9" "$V2/logs"
fresh_alias
gpu_idle
timeout 300 bash pod/mini_pod.sh push "$A" --dry-run > /dev/null 2>&1
"$PY" a9.py scan --stage-dir "$V2/pod_stage/$A" | tail -n 3
[ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: the push would copy a sealed-set or credential name; nothing was sent"; exit 3; }

# ---- 1. setup
step "setup" timeout 1800 bash pod/mini_pod.sh setup "$A"
log "setup done; $(elapsed) s, \$$(spent) so far"

# ---- 2-3. the 16-example adapter
HAVE16=0
D=$("$PY" a9.py decide --phase train --elapsed-s "$(elapsed)" "${REFFLAG[@]}" --cap-usd "$CAP" --rate "$RATE")
echo "$D" | head -n 1 > "$V2/a9/decide_train_$A.json"
log "decision before training: $(echo "$D" | tail -n 1)"
case $(echo "$D" | tail -n 1) in
  *"$N16"*)
    gpu_idle
    UNFETCHED=1
    step "train start" env GPU_FREE_MIB="$FREE_MIB" ALSO_LOCK="$OTHER_LOCK" TO_DOWNLOAD=1500 TO_TRAIN=3600 timeout 900 bash pod/mini_pod.sh train "$A" dpo --plan "$PLAN" --train-args "$TARGS"
    # wait only as long as three cells can still fit afterwards; then our chain is stopped and the reused cells go on
    TD=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['train_deadline_s'])" "$V2/a9/decide_train_$A.json")
    log "training started; this driver stops waiting for it $TD s from now"
    wait_stage dpo train_plan_dpo_a9 "${TD:-1800}"
    tws=$?
    if fetch_ok dpo; then
      UNFETCHED=0
      if [ $tws = 0 ] && [ -f "$V2/runs_$A/dpo/train/cells/$N16/DONE" ]; then HAVE16=1
      else log "the 16-example adapter is not trained (stage rc $tws): reported as not trained; the reused cells go on"; fi
    else
      log "training fetch not OK: the 16-example adapter cannot be staged; the reused cells go on"
    fi ;;
  CELLS=) stop_early 8 "neither three nor two cells fit inside the money stop after setup; nothing was trained or sampled" ;;
  *) log "training plus three cells does not fit inside the money stop: the 16-example adapter is not trained" ;;
esac
log "after training: $(elapsed) s, \$$(spent) so far; 16-example adapter available: $HAVE16"

# ---- 4. which cells are sampled, decided before the engine starts
NO16=()
[ $HAVE16 = 1 ] || NO16=(--no-n16)
D=$("$PY" a9.py decide --phase sample --elapsed-s "$(elapsed)" "${NO16[@]}" "${REFFLAG[@]}" --cap-usd "$CAP" --rate "$RATE")
echo "$D" | head -n 1 > "$V2/a9/decide_sample_$A.json"
CELLS=$(echo "$D" | tail -n 1)
CELLS=${CELLS#CELLS=}
log "decision before sampling: cells=${CELLS:-none} ($(echo "$D" | head -n 1))"
[ -n "$CELLS" ] || stop_early 8 "not even the rank-1 and 4-example cells fit inside the money stop; nothing was sampled"

# ---- 5. stage exactly these cells, then gate + sample
want=0
for c in ${CELLS//,/ }; do
  case $c in
    "$RANK1" | "$N4") runs=$R1; force=() ;;
    "$N16") runs=$V2/runs_$A; force=(--force) ;;  # a did_not_train adapter is still sampled, with its flag (9(b))
    "$REFC") runs=$R2; force=() ;;
    *) log "STOP: unknown cell $c"; exit 3 ;;
  esac
  OMP_NUM_THREADS=1 "$PY" stage_cells.py --runs "$runs" --ckpt dpo --cells "$c" --stage "$RB" "${force[@]}" 2>&1 | tail -n 1
  want=$((want + 1))
done
[ "$(staged "$RB")" = "$want" ] || stop_early 3 "batch_a9 staged $(staged "$RB") of $want cells; nothing was sampled"
"$PY" a9.py scan --stage-dir "$RB" | tail -n 1
[ "${PIPESTATUS[0]}" = 0 ] || stop_early 3 "the staged batch holds a sealed-set or credential name; nothing was sent"
gpu_idle
DL=$((STOP_S - FETCH_S - $(elapsed)))  # the money stop, less the time kept for the last fetch
[ "$DL" -gt 600 ] || stop_early 8 "no time left before the money stop; nothing was sampled"
UNFETCHED=1
# pod-side caps are a runaway guard only: later than this driver's own stop, so they never cut a healthy run
step "sample start" env GPU_FREE_MIB="$FREE_MIB" ALSO_LOCK="$OTHER_LOCK" TO_GATE=900 TO_SAMPLE=$((DL + 1200)) timeout 900 bash pod/mini_pod.sh sample "$A" dpo "$RB" --seed "$SEED"
log "sampling started: $want cells, seed $SEED; this driver stops the chain $DL s from now if it is still running"
wait_stage dpo sample_batch_a9 "$DL"
rws=$?

# ---- 6. final fetch
fetch_ok dpo || { WHY="the final fetch is not OK; keep $A until it is fetched"; log "STOP: $WHY"; exit 6; }
UNFETCHED=0
log "final fetch OK; $(elapsed) s, \$$(spent) of pod time"
case $rws in
  0) STATE=done
    log "next, on serrano (CPU): RUNBOOK_A9.md, section 'After the fetch' (batch dir $V2/runs_$A/dpo/sample/batch_a9)" ;;
  2) STATE=stopped
    WHY="the money stop arrived during sampling and this job's chain was stopped; unfinished draws are reported as bounds (RUNBOOK_A9.md, 'If sampling was cut')"
    exit 7 ;;
  *) WHY="the sample stage did not end DONE (rc $rws) and it was not the money stop; everything sampled is fetched; a resume is the coordinator's decision"
    exit 7 ;;
esac
