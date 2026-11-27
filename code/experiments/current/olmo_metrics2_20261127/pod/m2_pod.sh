#!/usr/bin/env bash
# m2_pod.sh: drive one RunPod pod for the 2026-11-27 olmo measurements. Run ON SERRANO against an ssh alias rp-<name>
# (~/.ssh/config). Never creates, stops or deletes a pod; never reads or prints a key or token. Modelled on
# olmo_elicitation_20261125/pod/olmo_pod.sh. Pod root /workspace/m2; weights /workspace/models/<ckpt>.
#
#   m2_pod.sh setup    <alias> [--no-wait]   push, then pod/m2_setup.sh (runs the 20261125 oe_setup.sh with its two /workspace/oe
#                                            paths redirected to /workspace/m2), wait for SETUP_DONE (~10 min, idempotent)
#   m2_pod.sh push     <alias>               stage + rsync to /workspace/m2, md5 every file both sides; refuses on mismatch
#   m2_pod.sh download <alias> <ckpt> [--verify-sha] [--force]
#                                            job download_<ckpt> (own lock, not the GPU lock): pinned-commit fetch + verify
#   m2_pod.sh job      <alias> <job> <timeout-s> [--force] [--nogpu] -- <command...>
#                                            one job under setsid + nohup + timeout --kill-after, flock on gpu.lock,
#                                            heartbeat every 60 s, state/{RUNNING,DONE,FAILED}, logs/job.log; cwd /workspace/m2
#   m2_pod.sh chain    <alias> <chain-name> <chain-file>
#                                            jobs in order, detached; file lines: <job> <timeout-s> <continue-codes e.g. 0 or 0,11>
#                                            [--nogpu] -- <shell command>; stops at the first rc not in its list; state in
#                                            runs/_chains/<name>/{state,log}; a job already DONE is skipped
#   m2_pod.sh status   <alias> [<job> ...]   one line per job
#   m2_pod.sh fetch    <alias> <job>         rsync runs/<job>/ -> $M2_DATA/runs/<job>/, md5 both -> FETCH_OK|FETCH_PARTIAL|FETCH_MISMATCH
#   m2_pod.sh stop     <alias> <job>         SIGTERM to that job's recorded process groups only (never pkill)
# Global flag (any position): --dry-run   print the ssh / rsync commands instead of running them.
# env: M2_DATA (default /scratch/cowifornia/nd-rl/olmo_elicit/metrics2), OE_DATA (default .../olmo_elicit).
set -euo pipefail
DRY=0
args=()
for a in "$@"; do [[ $a == --dry-run ]] && DRY=1 || args+=("$a"); done
set -- "${args[@]}"
cmd=${1:?usage: m2_pod.sh <setup|push|download|job|status|fetch|stop> <alias> ...}
h=${2:?usage: m2_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
P=$(cd "$(dirname "$0")" && pwd)     # .../olmo_metrics2_20261127/pod
X=$(dirname "$P")                    # experiment folder
OE_DATA=${OE_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit}
M2_DATA=${M2_DATA:-$OE_DATA/metrics2}
OLD_POD=${OLD_POD:-$X/../olmo_elicitation_20261125/pod}
REMOTE=/workspace/m2
CKPTS="dpo s0050 s0100 s0200 s0500 s0750 s1100 s1600 s2300"
FORBIDDEN='textbook72|tb72|test_13_20|test_21_30'
CREDENTIAL='(^|/)\.env|\.netrc|(^|/)id_(rsa|ecdsa|ed25519|runpod|ynez|dsa)|\.pem$|\.key$|credential|secret|api[_-]?key|hf_token|(^|/)token\.(txt|json)$|(^|/)\.ssh/|authorized_keys'
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
utc() { date -u +%FT%TZ; }
x() { if [[ $DRY == 1 ]]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
xs() { if [[ $DRY == 1 ]]; then echo "+ ssh $h $1"; else "${SSH[@]}" "$h" "$1"; fi; }

stage() {  # stage <dir>: copy what the pod needs; echo the manifest path. Refuses forbidden / credential names.
  local ST=$1 MAN=$1.manifest.txt
  rm -rf "$ST"; mkdir -p "$ST/pod" "$ST/tools" "$ST/inputs" "$M2_DATA/inputs"
  (cd "$X" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) -exec cp -p {} "$ST/" \;)
  cp -rp "$P"/. "$ST/pod/"
  [[ -f $OLD_POD/oe_setup.sh ]] && cp -p "$OLD_POD/oe_setup.sh" "$ST/pod/oe_setup.sh"   # copy; the original stays untouched
  [[ -d $X/tools ]] && cp -rp "$X/tools/." "$ST/tools/"
  for sub in temp rank1 steer; do if [[ -d $X/$sub ]]; then mkdir -p "$ST/$sub"; cp -rp "$X/$sub/." "$ST/$sub/"; fi; done
  cp -rp "$M2_DATA/inputs/." "$ST/inputs/"
  [[ -d $OE_DATA/pool ]] && { mkdir -p "$ST/pool"; cp -rp "$OE_DATA/pool/." "$ST/pool/"; }
  find "$ST" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ST" -name '*.pyc' -delete
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort) > "$MAN"
  local bad
  bad=$(grep -iE "$FORBIDDEN" "$MAN" || true)
  [[ -z $bad ]] || { echo "REFUSED: held-out names in the staged list:"; echo "$bad"; return 1; }
  bad=$(grep -iE "$CREDENTIAL" "$MAN" || true)
  [[ -z $bad ]] || { echo "REFUSED: credential-like names in the staged list:"; echo "$bad"; return 1; }
  echo "staged $(wc -l < "$MAN") files, $(du -sh "$ST" | cut -f1) for $h"
}

push() {
  local ST=$M2_DATA/pod_stage/$h MAN=$M2_DATA/pod_stage/$h.manifest.txt
  mkdir -p "$M2_DATA/pod_stage"
  stage "$ST" || { echo "PUSH REFUSED: nothing sent"; exit 1; }
  xs "mkdir -p $REMOTE"
  x timeout 1800 "${RSYNC[@]}" "$ST/" "$h:$REMOTE/"
  [[ $DRY == 1 ]] && { echo "+ (md5 check of $(wc -l < "$MAN") files on both sides)"; return 0; }
  (cd "$ST" && xargs -d '\n' md5sum < "$MAN") > "$MAN.lab.md5"
  "${SSH[@]}" "$h" "cd $REMOTE && xargs -d '\n' md5sum" < "$MAN" > "$MAN.pod.md5"
  if ! cmp -s "$MAN.lab.md5" "$MAN.pod.md5"; then
    diff "$MAN.lab.md5" "$MAN.pod.md5" | head -20; echo "PUSH MD5 MISMATCH: nothing started"; exit 1
  fi
  echo "push md5 ok: $(wc -l < "$MAN") files equal on serrano and $h ($(utc))"
}

start_job() {  # start_job <job> <timeout> <force 0|1> <nogpu 0|1> <command words...>
  local job=$1 to=$2 force=$3 nogpu=$4; shift 4
  [[ $job =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "job name: letters, digits, _ . - only"; exit 2; }
  [[ $to =~ ^[0-9]+$ ]] || { echo "timeout must be seconds"; exit 2; }
  [[ $# -gt 0 ]] || { echo "no command after --"; exit 2; }
  local c; c=$(printf '%q ' "$@")
  if [[ $DRY != 1 ]]; then
    "${SSH[@]}" "$h" "test -f $REMOTE/meta/setup.done" || { echo "no setup.done on $h: run setup first"; exit 1; }
  fi
  local out
  out=$(xs "mkdir -p $REMOTE/runs/$job/logs && cd $REMOTE && (setsid nohup env FORCE=$force NOGPU=$nogpu bash pod/m2_job.sh $job $to -- $c >> runs/$job/logs/nohup.log 2>&1 < /dev/null &) ; sleep 4; tail -2 runs/$job/logs/runner.log 2>/dev/null")
  echo "$out"
  if [[ $DRY != 1 ]] && grep -qE "REFUSED_DONE|LOCKED|GPU_BAD" <<< "$(tail -1 <<< "$out")"; then echo "NOT STARTED: $job on $h ($(utc))"; exit 1; fi
  echo "started $job on $h: timeout=${to}s force=$force nogpu=$nogpu ($(utc))"
}

case $cmd in
  setup)
    xs "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1"
    push
    xs "mkdir -p $REMOTE/meta && cd $REMOTE && (setsid nohup timeout 3000 bash pod/m2_setup.sh > meta/setup.log 2>&1 < /dev/null &) ; echo setup started"
    [[ ${1:-} == --no-wait || $DRY == 1 ]] && { echo "setup running detached; log $REMOTE/meta/setup.log"; exit 0; }
    echo "setup start $(utc) (about 10 min)"
    for _ in $(seq 1 100); do
      sleep 30
      out=$("${SSH[@]}" "$h" "tail -1 $REMOTE/meta/setup.log; pgrep -f 'pod/m2_setu[p].sh' > /dev/null && echo ALIVE || echo GONE" 2>/dev/null || echo "SSH_FAIL")
      if grep -q "SETUP_DONE" <<< "$out"; then
        "${SSH[@]}" "$h" "grep -E 'CODE_OK|VERSIONS|NLTK_OK|GRADER_OK|GPU_TEST|GPU_OK|GPU_BAD|SETUP_DONE' $REMOTE/meta/setup.log"
        echo "setup done $(utc)"; exit 0
      fi
      if grep -q "^GONE" <<< "$out"; then
        "${SSH[@]}" "$h" "tail -25 $REMOTE/meta/setup.log"; echo "SETUP FAILED ($(utc)): see $REMOTE/meta/setup.log"; exit 1
      fi
    done
    echo "SETUP TIMEOUT (50 min)"; exit 1 ;;
  push) push ;;
  pushrun)  # pushrun <alias> <job> [subpath]: copy a FETCHED run folder (or a sub-path of it) from serrano to the pod, md5-checked
    # (orchestrator addition 2026-11-27, for moving adapters and steering vectors from rp-m2a to the H200 pod). Same name refusals as push.
    job=${1:?pushrun <alias> <job> [subpath]}; sub=${2:-.}
    [[ $job =~ ^[A-Za-z0-9_.-]+$ && $sub =~ ^[A-Za-z0-9_./-]+$ && $sub != *..* ]] || { echo "bad job or subpath"; exit 2; }
    L=$M2_DATA/runs/$job; [[ -e $L/$sub ]] || { echo "PUSHRUN_EMPTY: no $L/$sub on serrano"; exit 1; }
    F=$M2_DATA/pod_stage/$h.pushrun.$job; mkdir -p "$(dirname "$F")"
    (cd "$L" && find "$sub" -type f ! -name '*.lock' | sed 's|^\./||' | sort) > "$F.list"
    bad=$(grep -iE "$FORBIDDEN|$CREDENTIAL" "$F.list" || true)
    [[ -z $bad ]] || { echo "PUSHRUN REFUSED: forbidden or credential-like names:"; echo "$bad"; exit 1; }
    if [[ $DRY == 1 ]]; then echo "+ rsync --files-from=$F.list $L/ $h:$REMOTE/runs/$job/ ($(wc -l < "$F.list") files); md5 both sides"; exit 0; fi
    "${SSH[@]}" "$h" "mkdir -p $REMOTE/runs/$job"
    timeout 3600 "${RSYNC[@]}" --files-from="$F.list" "$L/" "$h:$REMOTE/runs/$job/"
    (cd "$L" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5"
    "${SSH[@]}" "$h" "cd $REMOTE/runs/$job && xargs -d '\n' md5sum" < "$F.list" > "$F.pod.md5"
    if cmp -s "$F.lab.md5" "$F.pod.md5"; then echo "PUSHRUN_OK $job/$sub: $(wc -l < "$F.list") files md5-equal on serrano and $h ($(utc))"
    else echo "PUSHRUN_MISMATCH $job/$sub ($(utc))"; exit 1; fi ;;
  download)
    ck=${1:?download <alias> <ckpt>}; shift
    [[ " $CKPTS " == *" $ck "* ]] || { echo "unknown checkpoint $ck (one of: $CKPTS)"; exit 2; }
    vs=0; force=0
    for a in "$@"; do case $a in --verify-sha) vs=1 ;; --force) force=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done
    start_job "download_$ck" "${TO_DOWNLOAD:-3600}" "$force" 1 env M2_ROOT=$REMOTE VERIFY_SHA=$vs bash pod/m2_download.sh "$ck" ;;
  job)
    job=${1:?job <alias> <job> <timeout-s> -- <command...>}; to=${2:?timeout seconds}; shift 2
    force=0; nogpu=0
    while [[ $# -gt 0 && $1 != -- ]]; do
      case $1 in --force) force=1 ;; --nogpu) nogpu=1 ;; *) echo "unknown option $1"; exit 2 ;; esac; shift
    done
    [[ ${1:-} == -- ]] || { echo "missing -- before the command"; exit 2; }
    shift
    start_job "$job" "$to" "$force" "$nogpu" "$@" ;;
  status)
    xs "python3 $REMOTE/pod/m2_status.py --root $REMOTE $*" ;;
  fetch)
    job=${1:?fetch <alias> <job>}
    [[ $job =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name"; exit 2; }
    L=$M2_DATA/runs/$job; F=$M2_DATA/pod_stage/$h.fetch.$job
    mkdir -p "$L" "$(dirname "$F")"
    if [[ $DRY == 1 ]]; then
      echo "+ ssh $h 'cd $REMOTE/runs/$job && find . -type f ! -name \"*.lock\" | sort' > $F.list; md5sum both sides"
      echo "+ rsync --files-from=$F.list $h:$REMOTE/runs/$job/ $L/"; exit 0
    fi
    st=$("${SSH[@]}" "$h" "python3 $REMOTE/pod/m2_status.py --root $REMOTE $job 2>/dev/null | head -1" || true)
    "${SSH[@]}" "$h" "cd $REMOTE/runs/$job && find . -type f ! -name '*.lock' | sed 's|^\./||' | sort" > "$F.list" || true
    [[ -s $F.list ]] || { echo "FETCH_EMPTY $job: nothing under $REMOTE/runs/$job on $h"; exit 1; }
    "${SSH[@]}" "$h" "cd $REMOTE/runs/$job && xargs -d '\n' md5sum" < "$F.list" > "$F.pod.md5"
    timeout 3600 "${RSYNC[@]}" --files-from="$F.list" "$h:$REMOTE/runs/$job/" "$L/"
    (cd "$L" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5"
    n=$(wc -l < "$F.list"); sz=$(du -sh "$L" | cut -f1)
    running=0; grep -q " state=RUNNING " <<< "$st " && running=1
    if cmp -s "$F.pod.md5" "$F.lab.md5"; then
      if [[ $running == 1 ]]; then echo "FETCH_PARTIAL $job: $n files ($sz) md5-equal, but the job is still running ($(utc))"
      else echo "FETCH_OK $job: $n files ($sz) md5-equal on $h and serrano -> $L ($(utc))"; fi
    else
      nd=$(diff "$F.pod.md5" "$F.lab.md5" | grep -c '^<' || true)
      echo "FETCH_MISMATCH $job: $nd of $n files differ (running=$running; files still being written differ) ($(utc))"; exit 1
    fi ;;
  chain)
    cn=${1:?chain <alias> <chain-name> <chain-file>}; cf=${2:?chain file}
    [[ $cn =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "chain name: letters, digits, _ . - only"; exit 2; }
    [[ -r $cf ]] || { echo "cannot read $cf"; exit 2; }
    n=0
    while IFS= read -r ln || [[ -n $ln ]]; do
      [[ -z ${ln//[[:space:]]/} || $ln =~ ^[[:space:]]*# ]] && continue
      n=$((n + 1))
      [[ $ln =~ ^[[:space:]]*[A-Za-z0-9_.-]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+(,[0-9]+)*([[:space:]]+--nogpu)?[[:space:]]+--[[:space:]]+[^[:space:]] ]] \
        || { echo "bad chain line: $ln"; exit 2; }
    done < "$cf"
    [[ $n -gt 0 ]] || { echo "chain file has no jobs"; exit 2; }
    if [[ $DRY != 1 ]]; then
      "${SSH[@]}" "$h" "test -f $REMOTE/meta/setup.done" || { echo "no setup.done on $h: run setup first"; exit 1; }
      "${SSH[@]}" "$h" "python3 $REMOTE/pod/m2_status.py --root $REMOTE" | grep -q "^chain:$cn state=RUNNING " \
        && { echo "chain $cn is already running on $h"; exit 1; }
    fi
    xs "mkdir -p $REMOTE/runs/_chains/$cn/state $REMOTE/runs/_chains/$cn/log"
    if [[ $DRY == 1 ]]; then echo "+ ssh $h 'cat > $REMOTE/runs/_chains/$cn/chain.txt' < $cf ($n jobs)"
    else
      "${SSH[@]}" "$h" "cat > $REMOTE/runs/_chains/$cn/chain.txt" < "$cf"
      [[ $(md5sum < "$cf") == "$("${SSH[@]}" "$h" "md5sum < $REMOTE/runs/_chains/$cn/chain.txt")" ]] || { echo "chain file md5 mismatch"; exit 1; }
    fi
    xs "cd $REMOTE && (setsid nohup bash pod/m2_chain.sh $cn >> runs/_chains/$cn/log/nohup.log 2>&1 < /dev/null &) ; sleep 3; tail -2 runs/_chains/$cn/log/chain.log 2>/dev/null"
    echo "chain $cn started on $h: $n jobs ($(utc))" ;;
  stop)
    if [[ ${1:-} == --chain ]]; then
      cn=${2:?stop <alias> --chain <name>}; [[ $cn =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad chain name"; exit 2; }
      xs "M2_ROOT=$REMOTE bash $REMOTE/pod/m2_stop.sh --chain $cn"; exit 0
    fi
    job=${1:?stop <alias> <job> | stop <alias> --chain <name>}
    [[ $job =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name"; exit 2; }
    xs "M2_ROOT=$REMOTE bash $REMOTE/pod/m2_stop.sh $job" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
