#!/usr/bin/env bash
# authors_gate_driver.sh: the gate stage on one pod, run ON SERRANO by the orchestrator. For each corpus in turn:
# copy it, pretrain its seed-0 base, measure the base, fetch. One job at a time; stops at the first failure.
# It never creates or deletes a pod.
#
#   authors_gate_driver.sh <alias> [--setup] [--running <job>] <corpus> [<corpus> ...]
#     --setup          run `authors_pod.sh setup` and `check-env` first (a new pod)
#     --running <job>  the first corpus's pretraining was already started by hand as <job>: wait for it
set -uo pipefail
h=${1:?alias}
shift
W=$(cd "$(dirname "$0")" && pwd)
P="$W/authors_pod.sh"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
say() { echo "[$(date -u +%FT%TZ)] $h: $*"; }
die() { say "STOP: $*"; exit 1; }

wait_job() {  # wait_job <job name> <max seconds>: the job's log on the pod must end with rc 0
  local job=$1 max=$2 t0=$SECONDS line=""
  while (( SECONDS - t0 < max )); do
    line=$("${SSH[@]}" "$h" "L=\$(ls -t /workspace/ta/logs/${job}_*.log 2>/dev/null | head -1); [[ -n \$L ]] && grep -h 'job end rc' \$L | tail -1" 2>/dev/null || true)
    if [[ -n $line ]]; then
      say "$job: $line"
      [[ $line == *"job end rc 0"* ]] && return 0
      return 1
    fi
    sleep 60
  done
  say "$job: no end after ${max}s"
  return 1
}

setup=0
running=""
while [[ ${1:-} == --* ]]; do
  case $1 in
    --setup) setup=1; shift ;;
    --running) running=$2; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
[[ $# -ge 1 ]] || die "no corpus"

if (( setup )); then
  bash "$P" setup "$h" || die "setup"
fi
first=1
for c in "$@"; do
  if (( first )) && [[ -n $running ]]; then
    say "waiting for $running (started by hand)"
  else
    bash "$P" copy "$h" pretrain "$c" || die "copy $c"
    if (( first && setup )); then bash "$P" check-env "$h" || die "check-env"; fi
    bash "$P" start "$h" pretrain --corpus "$c" --seed 0 || die "start pretrain $c"
  fi
  first=0
  wait_job "pretrain_${c}_s0" 7500 || die "pretrain $c"
  bash "$P" start "$h" measure --base "${c}_s0" || die "start measure $c"
  wait_job "measure_${c}_s0" 7500 || die "measure $c"
  bash "$P" fetch "$h" || say "fetch after $c reported differences (it is repeated at the end)"
done
bash "$P" fetch "$h" || die "final fetch"
say "GATE STAGE DONE on this pod: $*"
