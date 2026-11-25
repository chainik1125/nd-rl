#!/usr/bin/env bash
# olmo_chain.sh <alias> <ckpt>[:<stages>] ... [--deadline <epoch>] [--n-h 8|4] [--edl-args "<flags>"] [--dry-run]
# SERRANO-side driver: the checkpoints one after another on ONE pod: run -> wait (status every 2 min) -> fetch
# (md5-checked) -> next. <stages> per checkpoint (default edl,T,H), e.g. the mini run:
#   olmo_chain.sh rp-ol1 dpo:edl,rep s0100 s0500 --deadline $(date -d '2026-11-25 13:30 PST' +%s)
# One log line per event in $OE_DATA/logs/chain_<alias>.log (UTC and PT). Watchdog: heartbeat older than 10 min ->
# STALL line; older than 30 min -> the chain is stopped (SIGTERM), fetched, and the driver moves on. At the deadline no
# new checkpoint starts; a running one is stopped and what exists is fetched (PREREG: unfinished is reported as such).
# Never creates or deletes a pod.
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)
OE_DATA=${OE_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit}
h=${1:?usage: olmo_chain.sh <alias> <ckpt>[:stages] ... [--deadline epoch]}
shift
specs=(); deadline=0; nh=8; edl_args="--max-tok 6144"; dry=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --deadline) deadline=$2; shift 2 ;;
    --n-h) nh=$2; shift 2 ;;
    --edl-args) edl_args=$2; shift 2 ;;
    --dry-run) dry=--dry-run; shift ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) specs+=("$1"); shift ;;
  esac
done
[[ ${#specs[@]} -gt 0 ]] || { echo "no checkpoints given"; exit 2; }
[[ $deadline =~ ^[0-9]+$ ]] || { echo "--deadline must be epoch seconds"; exit 2; }
POLL=${POLL:-120}
mkdir -p "$OE_DATA/logs"
LOG=$OE_DATA/logs/chain_$h.log
ev() { local m="$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date +'%-I:%M %p PT')) $h $*"; echo "$m" | tee -a "$LOG"; }
past() { [[ $deadline -gt 0 && $(date +%s) -ge $deadline ]]; }
pod() { bash "$P/olmo_pod.sh" "$@" $dry; }

ev "DRIVER_START specs=${specs[*]} deadline=$([[ $deadline -gt 0 ]] && TZ=America/Los_Angeles date -d @"$deadline" +'%F %-I:%M %p PT' || echo none) n_h=$nh edl_args='$edl_args'"
for spec in "${specs[@]}"; do
  ck=${spec%%:*}; st=edl,T,H; [[ $spec == *:* ]] && st=${spec#*:}
  if past; then ev "DEADLINE_SKIP $ck (not started)"; continue; fi
  ev "RUN $ck stages=$st"
  if ! out=$(pod run "$h" "$ck" --stages "$st" --n-h "$nh" --edl-args "$edl_args" 2>&1); then
    ev "RUN_FAIL $ck: $(echo "$out" | tail -2 | tr '\n' ' ')"; continue
  fi
  echo "$out" >> "$LOG"
  [[ -n $dry ]] && { ev "DRY_RUN: not waiting"; continue; }
  stalled_since=0; last=""
  while true; do
    sleep "$POLL"
    s=$(pod status "$h" "$ck" 2>/dev/null)
    v=$(echo "$s" | python3 "$P/oe_status.py" --verdict "$ck" --stages "$st")
    word=${v%% *}
    summ=$(echo "$v" | cut -d' ' -f4-)
    [[ $summ != "$last" ]] && { ev "STATUS $ck $v"; last=$summ; }
    [[ $word == FINISHED ]] && break
    hb=$(echo "$v" | sed -n 's/.* hb=\([0-9na]*\).*/\1/p')
    if [[ $word == NOSTATUS ]]; then
      ev "NOSTATUS $ck (ssh or status failed)"
      stalled_since=$(( stalled_since == 0 ? $(date +%s) : stalled_since ))
    elif [[ $hb =~ ^[0-9]+$ && $hb -gt 600 ]]; then
      ev "STALL $ck heartbeat ${hb}s old"
      stalled_since=$(( stalled_since == 0 ? $(date +%s) - hb : stalled_since ))
    else
      stalled_since=0
    fi
    if [[ $stalled_since -gt 0 && $(( $(date +%s) - stalled_since )) -gt 1800 ]]; then
      ev "STALL_STOP $ck (no heartbeat / status for 30 min): stopping"; pod stop "$h" "$ck" >> "$LOG" 2>&1; sleep 30; break
    fi
    if past; then ev "DEADLINE_STOP $ck: stopping the running chain"; pod stop "$h" "$ck" >> "$LOG" 2>&1; sleep 30; break; fi
  done
  for t in 1 2 3; do
    f=$(pod fetch "$h" "$ck" 2>&1 | tail -1)
    ev "FETCH $ck try $t: $f"
    [[ $f == FETCH_OK* || $f == FETCH_PARTIAL* ]] && break
    sleep 60
  done
  ev "DONE $ck $(pod status "$h" "$ck" 2>/dev/null | python3 "$P/oe_status.py" --verdict "$ck" --stages "$st")"
  if past; then ev "DEADLINE reached after $ck"; fi
done
ev "DRIVER_END"
