#!/usr/bin/env bash
# mini_pod.sh: drive one RunPod pod for the minimal-adapter experiment. Run ON SERRANO against an ssh alias rp-<name>
# (~/.ssh/config handles the key). Never creates, stops or deletes a pod; never reads or prints a key or token (models and
# Dan's bucket are public). Modelled on the 2026-11-25 olmo_pod.sh.
#
#   mini_pod.sh setup  <alias> [--no-wait]    push code + inputs to /workspace/mini (md5), run pod/mini_setup.sh detached
#                                             (Dan's pinned commits and versions; ~10 min), wait for SETUP_DONE
#   mini_pod.sh push   <alias>                push again (code, pools, plans, stage2 L inputs); md5 both sides. Code = this
#                                             folder's *.py / *.sh / *.md (not test_*), the 2026-11-25 folder's olmo_edl.py and
#                                             olmo_template.py copied next to mini_train.py, $V2/pools, $V2/plans, stage2 L inputs
#   mini_pod.sh train  <alias> <ckpt> --plan <file> [--train-args "<flags>"]
#                                             push, then one chain: download (weights kept) -> train
#                                             (mini_train.py run --model <weights> --ckpt <ck> --plan <plan> --root pools
#                                             --out runs/<ck>/train <train-args>); <file> is copied to plans/ on the pod
#   mini_pod.sh sample <alias> <ckpt> <batch dir>
#                                             push code, push the staged batch (<batch dir> = adapters/<batch>/<cell>/..., made
#                                             by stage_cells.py; md5-checked), then one chain: gate:<batch> -> sample:<batch>
#                                             (cells that FAIL the gate are not sampled; Dan's grade.py per cell)
#   mini_pod.sh plain  <alias> <ckpt> <name> --problems <file> --pids <file> --seed S
#                                             push code + the problems / pids files (-> stage/problems_<name>.jsonl, pids_<name>.txt,
#                                             seed_<name>.txt), then one chain: download -> plain:<name> (Dan's sampler unchanged, no
#                                             LoRA; his grade.py on the same problems file) -> runs/<ckpt>/plain_<name>/
#   mini_pod.sh status <alias> [<ckpt> ...]   one line per stage, per training cell, per chain: state, progress, heartbeat
#                                             age, GPU MiB, free GB
#   mini_pod.sh fetch  <alias> <ckpt>         rsync runs/<ckpt>/ -> $V2/runs_<alias>/<ckpt>/ (or $FETCH_DIR/<ckpt>/), md5 both sides, one verdict line:
#                                             FETCH_OK / FETCH_PARTIAL (chain still running) / FETCH_MISMATCH
#   mini_pod.sh stop   <alias> [<ckpt>]       SIGTERM to the recorded chain's process group (never pkill)
# Global flag (any position): --dry-run   print the ssh / rsync commands instead of running them.
# env: V2 (default /scratch/cowifornia/nd-rl/olmo_elicit/v2), OE_DATA (.../olmo_elicit), MINI_ROOT (pod root, /workspace/mini),
#      TO_DOWNLOAD / TO_TRAIN / TO_GATE / TO_SAMPLE (seconds), DELETE_WEIGHTS=1 (default: weights kept), VERIFY_SHA=1,
#      GPU_FREE_MIB / ALSO_LOCK (a pod shared with another job; see mini_chain.sh; unset = off).
set -euo pipefail
DRY=0
args=()
for a in "$@"; do [[ $a == --dry-run ]] && DRY=1 || args+=("$a"); done
set -- "${args[@]}"
cmd=${1:?usage: mini_pod.sh <setup|push|train|sample|status|fetch|stop> <alias> ...}
h=${2:?usage: mini_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
P=$(cd "$(dirname "$0")" && pwd)     # .../olmo_minimal_adapter_20261127/pod
X=$(dirname "$P")                    # our experiment folder
OLD=$(dirname "$X")/olmo_elicitation_20261125
OE_DATA=${OE_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit}
V2=${V2:-$OE_DATA/v2}
MINI=${MINI_ROOT:-/workspace/mini}
CKPTS="dpo s0050 s0100 s0200 s0500 s0750 s1100 s1600 s2300"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
utc() { date -u +%FT%TZ; }
x() {  # run, or print under --dry-run
  if [[ $DRY == 1 ]]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi
}
xs() {  # ssh <alias> <remote command string>
  if [[ $DRY == 1 ]]; then echo "+ ssh $h $1"; else "${SSH[@]}" "$h" "$1"; fi
}
md5_both() {  # md5_both <local root> <manifest file> <remote root> <label>: every file of the manifest equal on both sides
  local lroot=$1 man=$2 rroot=$3 label=$4
  [[ $DRY == 1 ]] && { echo "+ (md5 check of $(wc -l < "$man") files: $label)"; return 0; }
  (cd "$lroot" && xargs -d '\n' md5sum < "$man") > "$man.lab.md5"
  "${SSH[@]}" "$h" "cd $rroot && xargs -d '\n' md5sum" < "$man" > "$man.pod.md5"
  if ! cmp -s "$man.lab.md5" "$man.pod.md5"; then
    diff "$man.lab.md5" "$man.pod.md5" | head -20; echo "PUSH MD5 MISMATCH ($label): nothing started"; return 1
  fi
  echo "push md5 ok ($label): $(wc -l < "$man") files equal on serrano and $h ($(utc))"
}

push() {  # push [plan file]: stage code + inputs, rsync to $MINI, md5 every file both sides
  local ST=$V2/pod_stage/$h MAN=$V2/pod_stage/$h.manifest.txt plan=${1:-}
  rm -rf "$ST"; mkdir -p "$ST/pod" "$ST/stage2" "$ST/plans"
  (cd "$X" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) ! -name 'test_*' -exec cp -p {} "$ST/" \;)
  cp -p "$P"/*.sh "$P"/*.py "$ST/pod/"
  for f in olmo_edl.py olmo_template.py; do cp -p "$OLD/$f" "$ST/$f"; done   # next to mini_train.py (it imports them)
  for f in pids_L.txt seeds_L.csv; do [[ -s $OE_DATA/stage2/$f ]] && cp -p "$OE_DATA/stage2/$f" "$ST/stage2/$f"; done
  if [[ -d $V2/pools ]]; then mkdir -p "$ST/pools"; cp -rp "$V2/pools/." "$ST/pools/"; fi
  if [[ -d $V2/plans ]]; then cp -rp "$V2/plans/." "$ST/plans/"; fi
  [[ -n $plan ]] && cp -p "$plan" "$ST/plans/$(basename "$plan")"
  if [[ -n ${PLAIN_NAME:-} ]]; then   # plain stage inputs
    mkdir -p "$ST/stage"
    cp -p "$PLAIN_PROBLEMS" "$ST/stage/problems_$PLAIN_NAME.jsonl"; cp -p "$PLAIN_PIDS" "$ST/stage/pids_$PLAIN_NAME.txt"
    echo "$PLAIN_SEED" > "$ST/stage/seed_$PLAIN_NAME.txt"
  fi
  find "$ST" -name __pycache__ -prune -exec rm -rf {} +
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$MAN")
  echo "staged $(wc -l < "$MAN") files, $(du -sh "$ST" | cut -f1) for $h"
  xs "mkdir -p $MINI"
  x timeout 1800 "${RSYNC[@]}" "$ST/" "$h:$MINI/"
  md5_both "$ST" "$MAN" "$MINI" "code and inputs"
}

push_batch() {  # push_batch <batch dir>: adapters/<batch>/ -> $MINI/adapters/<batch>/, md5 both sides
  local src=$1 b; b=$(basename "$1")
  [[ $b =~ ^[A-Za-z0-9._-]+$ ]] || { echo "batch name must be [A-Za-z0-9._-]+ (got $b)"; exit 2; }
  [[ -f $src/cells_manifest.json || $DRY == 1 ]] || { echo "$src has no cells_manifest.json (stage_cells.py)"; exit 2; }
  local MAN=$V2/pod_stage/$h.batch.$b.txt; mkdir -p "$V2/pod_stage"
  [[ -d $src ]] && (cd "$src" && find . -type f | sed 's|^\./||' | sort > "$MAN")
  xs "mkdir -p $MINI/adapters/$b"
  x timeout 3600 "${RSYNC[@]}" "$src/" "$h:$MINI/adapters/$b/"
  md5_both "$src" "$MAN" "$MINI/adapters/$b" "batch $b"
}

chain_guard() {
  [[ $DRY == 1 ]] && return 0
  local alive
  alive=$("${SSH[@]}" "$h" "python3 $MINI/pod/mini_status.py 2>/dev/null | grep ' chain alive=1' || true")
  [[ -z $alive ]] || { echo "a chain is running on $h: $alive"; exit 1; }
  "${SSH[@]}" "$h" "test -f $MINI/meta/setup.done" || { echo "no setup.done on $h: run setup first"; exit 1; }
}

start_chain() {  # start_chain <ckpt> <stages> <total timeout s> <extra env string>
  local ck=$1 stages=$2 total=$3 envs=$4 out
  local base="VERIFY_SHA=${VERIFY_SHA:-0} DELETE_WEIGHTS=${DELETE_WEIGHTS:-0} TO_DOWNLOAD=${TO_DOWNLOAD:-2400} TO_TRAIN=${TO_TRAIN:-21600} TO_GATE=${TO_GATE:-3600} TO_SAMPLE=${TO_SAMPLE:-21600} GPU_FREE_MIB=${GPU_FREE_MIB:-} ALSO_LOCK=${ALSO_LOCK:-}"
  out=$(xs "mkdir -p $MINI/runs/$ck/logs && cd $MINI && (MINI_ROOT=$MINI setsid nohup timeout --signal=TERM --kill-after=120 $total env $base $envs bash pod/mini_chain.sh $ck $stages >> runs/$ck/logs/nohup.log 2>&1 < /dev/null &) ; sleep 3; tail -3 runs/$ck/logs/chain.log 2>/dev/null")
  echo "$out"
  if grep -q "LOCKED\|GPU_BAD\|GPU_BUSY" <<< "$(tail -1 <<< "$out")"; then echo "NOT STARTED: $ck on $h ($(utc))"; exit 1; fi
  echo "started $ck on $h: stages=$stages timeout=${total}s ($(utc))"
}

check_ck() { [[ " $CKPTS " == *" $1 "* ]] || { echo "unknown checkpoint $1 (one of: $CKPTS)"; exit 2; }; }

case $cmd in
  setup)
    xs "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1"
    push
    xs "mkdir -p $MINI/meta && cd $MINI && (MINI_ROOT=$MINI setsid nohup timeout 3000 bash pod/mini_setup.sh > meta/setup.log 2>&1 < /dev/null &) ; echo setup started"
    [[ ${1:-} == --no-wait || $DRY == 1 ]] && { echo "setup running detached; log $MINI/meta/setup.log"; exit 0; }
    echo "setup start $(utc) (about 10 min)"
    for _ in $(seq 1 100); do
      sleep 30
      out=$("${SSH[@]}" "$h" "tail -1 $MINI/meta/setup.log; pgrep -f 'pod/mini_setu[p].sh' > /dev/null && echo ALIVE || echo GONE" 2>/dev/null || echo "SSH_FAIL")
      if grep -q "SETUP_DONE" <<< "$out"; then
        "${SSH[@]}" "$h" "grep -E 'CODE_OK|VERSIONS|NLTK_OK|GRADER_OK|GPU_TEST|GPU_OK|GPU_BAD|SETUP_DONE' $MINI/meta/setup.log"
        echo "setup done $(utc)"; exit 0
      fi
      if grep -q "^GONE" <<< "$out"; then
        "${SSH[@]}" "$h" "tail -25 $MINI/meta/setup.log"; echo "SETUP FAILED ($(utc)): see $MINI/meta/setup.log"; exit 1
      fi
    done
    echo "SETUP TIMEOUT (50 min)"; exit 1 ;;
  push)
    push ;;
  train)
    ck=${1:?train <alias> <ckpt> --plan <file> [--train-args \"...\"]}; shift
    check_ck "$ck"
    plan=""; targs=""
    while [[ $# -gt 0 ]]; do
      case $1 in
        --plan) plan=$2; shift 2 ;;
        --train-args) targs=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    [[ -s $plan ]] || { echo "--plan <file> required (not found: ${plan:-unset})"; exit 2; }
    [[ $targs =~ ^[-A-Za-z0-9_.,=/:\ ]*$ ]] || { echo "--train-args: plain flags only"; exit 2; }
    [[ -d $V2/pools ]] || { echo "no $V2/pools (build_pools.py)"; exit 2; }
    grep -q "^$ck," "$OE_DATA/stage2/seeds_L.csv" || { echo "no seed for $ck in $OE_DATA/stage2/seeds_L.csv"; exit 2; }
    chain_guard
    push "$plan"
    total=$(( ${TO_DOWNLOAD:-2400} + ${TO_TRAIN:-21600} + 1200 ))
    pstem=$(basename "$plan" .json); [[ $pstem =~ ^[A-Za-z0-9._-]+$ ]] || { echo "plan file name must be [A-Za-z0-9._-]+.json"; exit 2; }
    start_chain "$ck" "download,train:$pstem" "$total" "PLAN=plans/$(basename "$plan") TRAIN_ARGS='$targs'" ;;  # one stage name per plan file, so a second plan is not skipped as DONE
  sample)
    ck=${1:?sample <alias> <ckpt> <batch dir> [--seed S] [--pids FILE]}; batch=${2:?sample <alias> <ckpt> <batch dir>}
    shift 2
    sseed=""; spids=""
    while [[ $# -gt 0 ]]; do
      case $1 in
        --seed) sseed=$2; shift 2 ;;
        --pids) spids=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    check_ck "$ck"
    b=$(basename "${batch%/}")
    [[ -s $OE_DATA/stage2/pids_L.txt && -s $OE_DATA/stage2/seeds_L.csv ]] || { echo "no L inputs in $OE_DATA/stage2"; exit 2; }
    grep -q "^$ck," "$OE_DATA/stage2/seeds_L.csv" || { echo "no seed for $ck in seeds_L.csv"; exit 2; }
    [[ -z $sseed || $sseed =~ ^[0-9]+$ ]] || { echo "--seed S (integer)"; exit 2; }
    [[ -z $spids || -s $spids ]] || { echo "--pids file missing or empty: $spids"; exit 2; }
    chain_guard
    push
    push_batch "${batch%/}"
    senv=""
    if [[ -n $spids ]]; then  # a custom pids file (smoke tests), copied next to the batch with a checksum compare
      xs "mkdir -p $MINI/stage"
      x timeout 600 "${RSYNC[@]}" -c "$spids" "$h:$MINI/stage/pids_sample_$b.txt"
      senv="PIDS_FILE=$MINI/stage/pids_sample_$b.txt"
    fi
    [[ -n $sseed ]] && senv="$senv SEED_SAMPLE=$sseed"
    total=$(( ${TO_DOWNLOAD:-2400} + ${TO_GATE:-3600} + ${TO_SAMPLE:-21600} + 3600 ))
    start_chain "$ck" "download,gate:$b,sample:$b" "$total" "$senv" ;;
  plain)
    ck=${1:?plain <alias> <ckpt> <name> --problems <file> --pids <file> --seed S}; name=${2:?plain <alias> <ckpt> <name> ...}; shift 2
    check_ck "$ck"
    [[ $name =~ ^[A-Za-z0-9.-]+$ ]] || { echo "name must be [A-Za-z0-9.-]+ (got $name)"; exit 2; }
    prob=""; pids=""; seed=""
    while [[ $# -gt 0 ]]; do
      case $1 in
        --problems) prob=$2; shift 2 ;;
        --pids) pids=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    [[ -s $prob && -s $pids ]] || { echo "--problems and --pids files required"; exit 2; }
    [[ $seed =~ ^[0-9]+$ ]] || { echo "--seed S (integer) required"; exit 2; }
    chain_guard
    PLAIN_NAME=$name PLAIN_PROBLEMS=$prob PLAIN_PIDS=$pids PLAIN_SEED=$seed push
    total=$(( ${TO_DOWNLOAD:-2400} + ${TO_SAMPLE:-21600} + 3600 ))
    start_chain "$ck" "download,plain:$name" "$total" "" ;;
  status)
    xs "MINI_ROOT=$MINI python3 $MINI/pod/mini_status.py $*" ;;
  fetch)
    ck=${1:?fetch <alias> <ckpt>}
    check_ck "$ck"
    L=${FETCH_DIR:-$V2/runs_$h}/$ck; F=$V2/pod_stage/$h.fetch.$ck  # one local tree per pod: two pods both write runs/<ck>/logs, state
    mkdir -p "$L" "$(dirname "$F")"
    if [[ $DRY == 1 ]]; then
      echo "+ ssh $h 'cd $MINI/runs/$ck && find . -type f | sort' > $F.list; md5sum both sides"
      echo "+ rsync --files-from=$F.list $h:$MINI/runs/$ck/ $L/"; exit 0
    fi
    alive=$("${SSH[@]}" "$h" "MINI_ROOT=$MINI python3 $MINI/pod/mini_status.py $ck 2>/dev/null | grep -c ' chain alive=1' || true")
    "${SSH[@]}" "$h" "cd $MINI/runs/$ck && find . -type f ! -name '*.lock' | sed 's|^\./||' | sort" > "$F.list" || true
    [[ -s $F.list ]] || { echo "FETCH_EMPTY $ck: nothing under $MINI/runs/$ck on $h"; exit 1; }
    "${SSH[@]}" "$h" "cd $MINI/runs/$ck && xargs -d '\n' md5sum" < "$F.list" > "$F.pod.md5"
    timeout 3600 "${RSYNC[@]}" --files-from="$F.list" "$h:$MINI/runs/$ck/" "$L/"
    (cd "$L" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5"
    n=$(wc -l < "$F.list"); sz=$(du -sh "$L" | cut -f1)
    if cmp -s "$F.pod.md5" "$F.lab.md5"; then
      if [[ ${alive:-0} != 0 ]]; then echo "FETCH_PARTIAL $ck: $n files ($sz) md5-equal, but the chain is still running ($(utc))"
      else echo "FETCH_OK $ck: $n files ($sz) md5-equal on $h and serrano -> $L ($(utc))"; fi
    else
      nd=$(diff "$F.pod.md5" "$F.lab.md5" | grep -c '^<' || true)
      echo "FETCH_MISMATCH $ck: $nd of $n files differ (chain alive=${alive:-?}; files still being written differ) ($(utc))"
      exit 1
    fi ;;
  stop)
    ck=${1:-}
    xs "cd $MINI/runs && for d in ${ck:-*}; do f=\$d/state/chain.pid; [ -f \$f ] || continue; set -- \$(cat \$f); [ -f \$d/state/chain.DONE ] && continue; kill -0 \$1 2>/dev/null || continue; echo \"SIGTERM chain \$d pid=\$1 pgid=\$2\"; G=\$(ps -eo pid=,ppid=,pgid= | awk -v r=\$1 '{p[\$1]=\$2; g[\$1]=\$3} END {for (x in p) {y=x; while (y in p && y != r && y > 1) y=p[y]; if (y == r) print g[x]}}' | sort -u); for gg in \$2 \$G; do kill -TERM -- -\$gg 2>/dev/null; done; kill -TERM \$1 2>/dev/null; echo \"process groups signalled: \$2 \$G\"; done; sleep 20; MINI_ROOT=$MINI python3 $MINI/pod/mini_status.py ${ck} | grep ' chain '" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
