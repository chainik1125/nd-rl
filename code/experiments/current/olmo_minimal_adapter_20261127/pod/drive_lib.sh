# drive_lib.sh: shared helpers for the serrano-side drivers of Amendments 6 and 8 (audit T1, T2, T10). Sourced by
# pod/drive_a6s2_sample.sh and pod/drive_a6b_s1100.sh after they set A (pod alias), CK (checkpoint), V2 and cd to the
# working folder. Never stops or deletes a pod: wait_stage's OVERRUN stops only our own chain on the pod (mini_pod.sh
# stop: SIGTERM to the recorded chain's process groups), and every exit that is not a clean finish prints a line that
# tells the coordinator the pod needs a decision.
# shellcheck shell=bash

log() { echo "$(date -u +%FT%TZ) $*"; }
SSHO=(-o BatchMode=yes -o ConnectTimeout=15)
READY=0

on_exit() {
  local rc=$?
  if [ "$READY" = 1 ]; then
    log "POD READY TO DELETE: $A (every fetch FETCH_OK)"
  else
    log "EXIT rc=$rc: pod $A NEEDS A DECISION from the coordinator. This driver never stops or deletes a pod." \
      "Delete it once its last fetch line reads FETCH_OK, or fetch by hand first: bash pod/mini_pod.sh fetch $A $CK"
  fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT

fresh_alias() {  # T10: a reused alias would overwrite v2/runs_<alias>; chain_guard passes on a pod without /workspace/mini
  [ -e "$V2/runs_$A" ] && { log "STOP: $V2/runs_$A exists (alias reused?)"; exit 2; }
  local r
  r=$(ssh "${SSHO[@]}" "$A" 'test -e /workspace/mini && echo USED || echo FRESH' 2>/dev/null)
  [ "$r" = FRESH ] || { log "STOP: $A is not a fresh pod (probe: ${r:-no answer})"; exit 2; }
}

probe() {  # probe <ckpt> <stage stem>: DONE | FAILED | NOTRUN (chain done, stage absent) | RUNNING <heartbeat age s> | DEAD | SSH_FAIL
  local st=/workspace/mini/runs/$1/state
  ssh "${SSHO[@]}" "$A" "cd $st 2>/dev/null || { echo DEAD; exit 0; }
    if [ -f chain.DONE ]; then
      if [ -f $2.DONE ]; then echo DONE; elif [ -f $2.FAILED ]; then echo FAILED; else echo NOTRUN; fi
    else
      p=\$(cut -d' ' -f1 chain.pid 2>/dev/null)
      if [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null; then
        echo RUNNING \$(( \$(date +%s) - \$(cut -d' ' -f1 heartbeat 2>/dev/null || echo 0) ))
      else echo DEAD; fi
    fi" 2>/dev/null || echo SSH_FAIL
}

wait_stage() {  # wait_stage <ckpt> <stage stem> <deadline s>: 0 DONE; 1 FAILED, NOTRUN or DEAD; 2 OVERRUN (chain stopped)
  local end=$(($(date +%s) + $3)) v dead=0 age
  while :; do
    v=$(probe "$1" "$2")
    case $v in
      DONE) log "$2: DONE"; return 0 ;;
      FAILED | NOTRUN) log "$2: $v"; return 1 ;;
      DEAD)
        dead=$((dead + 1))
        [ $dead -ge 2 ] && { log "$2: chain DEAD without chain.DONE"; return 1; } ;;
      RUNNING*)
        dead=0; age=${v#RUNNING }
        [ "$age" -gt 600 ] 2>/dev/null && log "STALL $2: heartbeat $age s old" ;;
      *) dead=0; log "probe $2: ${v:-no answer}" ;;
    esac
    if [ "$(date +%s)" -gt "$end" ]; then
      log "OVERRUN $2 after $3 s: stopping our chain on $A (the pod itself keeps running)"
      timeout 300 bash pod/mini_pod.sh stop "$A" "$1" 2>&1 | tail -n 2
      return 2
    fi
    sleep 60
  done
}

fetch_ok() {  # fetch_ok <ckpt>: 0 when a fetch reads FETCH_OK (two tries)
  local f i
  for i in 1 2; do
    f=$(timeout 900 bash pod/mini_pod.sh fetch "$A" "$1" 2>&1 | tail -n 1)
    log "fetch $1 (try $i): $f"
    case $f in FETCH_OK*) return 0 ;; esac
    sleep 30
  done
  return 1
}

step() {  # step <what> <command...>: run a mini_pod.sh step, show its last 2 lines, STOP (exit 5) if it failed
  local what=$1
  shift
  "$@" 2>&1 | tail -n 2
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: $what failed"; exit 5; }
}

staged() {  # staged <batch dir>: number of cells in its manifest
  python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cells']))" "$1/cells_manifest.json" 2>/dev/null || echo 0
}
