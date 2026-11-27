#!/usr/bin/env bash
# m2_watch.sh <alias> <job> <expected-minutes>   (SERRANO side; never kills or stops anything, it only logs)
# Polls `m2_pod.sh status` every POLL seconds (default 120) and appends to $M2_DATA/logs/watch_<alias>.log (UTC and PT):
#   STALL    heartbeat older than 10 min, or the newest output file has not grown for 15 min (logged once per episode)
#   OVERRUN  job running longer than 1.6 x expected minutes (logged once)
#   NOSTATUS ssh or status failed
#   EXIT     the job ended (state DONE / FAILED / STALE / ABSENT); the watcher then exits
# env: M2_DATA, POLL, HB_STALL (600), GROW_STALL (900) in seconds, M2_POD_CMD (default: m2_pod.sh next to this file; a test can point it elsewhere).
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)
h=${1:?usage: m2_watch.sh <alias> <job> <expected-minutes>}; job=${2:?job}; exp_min=${3:?expected minutes}
[[ $exp_min =~ ^[0-9]+$ ]] || { echo "expected-minutes must be an integer"; exit 2; }
M2_DATA=${M2_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit/metrics2}
POD=${M2_POD_CMD:-$P/m2_pod.sh}
POLL=${POLL:-120}; HB_STALL=${HB_STALL:-600}; GROW_STALL=${GROW_STALL:-900}  # seconds
mkdir -p "$M2_DATA/logs"
LOG=$M2_DATA/logs/watch_$h.log
ev() { local m="$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date +'%-I:%M %p PT')) $h $job $*"; echo "$m" | tee -a "$LOG"; }
field() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<< "$2" | head -1; }
ev "WATCH_START expected=${exp_min}min stall: hb>600s or no growth 900s; overrun at $((exp_min * 16 / 10))min"
t_start=$(date +%s); last_sig=""; last_change=$t_start; stall_hb=0; stall_gr=0; overrun=0; nostat=0
while true; do
  line=$(bash "$POD" status "$h" "$job" 2>/dev/null | grep "^$job " | head -1 || true)
  now=$(date +%s)
  if [[ -z $line ]]; then
    ((nostat++ == 0)) && ev "NOSTATUS (ssh or status failed)"
    sleep "$POLL"; continue
  fi
  nostat=0
  state=$(field state "$line"); hb=$(field hb "$line"); hb=${hb%s}; prog=$(field progress "$line")
  case $state in
    DONE|FAILED|STALE|ABSENT)
      ev "EXIT state=$state rc=$(field rc "$line") progress=$prog elapsed=$(( (now - t_start) / 60 ))min"; exit 0 ;;
  esac
  # job age: from the pod's own start stamp when present
  st=$(field started "$line"); [[ $st =~ ^[0-9]+$ ]] || st=$t_start
  age_min=$(( (now - st) / 60 ))
  if (( age_min * 10 >= exp_min * 16 )) && (( overrun == 0 )); then
    ev "OVERRUN running ${age_min}min >= 1.6 x ${exp_min}min progress=$prog"; overrun=1
  fi
  if [[ $hb =~ ^[0-9]+$ && $hb -gt $HB_STALL ]]; then
    ((stall_hb == 0)) && ev "STALL heartbeat ${hb}s old (state=$state progress=$prog)"; stall_hb=1
  else stall_hb=0; fi
  sig="$(field newest "$line"):$(field newest_size "$line")"
  if [[ $sig != "$last_sig" ]]; then last_sig=$sig; last_change=$now; stall_gr=0
  elif (( now - last_change > $GROW_STALL )); then
    ((stall_gr == 0)) && ev "STALL newest output file ($sig) unchanged for $(( (now - last_change) / 60 ))min (progress=$prog)"; stall_gr=1
  fi
  sleep "$POLL"
done
