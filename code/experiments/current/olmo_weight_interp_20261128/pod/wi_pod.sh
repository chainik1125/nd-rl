#!/usr/bin/env bash
# wi_pod.sh: drive the one pod of the weight-interpolation run. Run ON SERRANO against an ssh alias rp-<name>
# (~/.ssh/config). Never creates, stops or deletes a pod; never reads or prints a key or token. A cut-down copy of
# metrics 2's pod/m2_pod.sh (same job runner, chain runner, status, stop and download scripts, pushed unedited) with the
# pod root /workspace/wi, this run's data root, and an explicit list of the files that are pushed (pod/STAGE_LIST.txt).
#
#   wi_pod.sh setup    <alias> [--no-wait]   push, then pod/m2_setup.sh (Dan's pinned stack: vLLM 0.11.0 etc.), wait for SETUP_DONE
#   wi_pod.sh push     <alias>               stage + rsync to /workspace/wi, md5 of every file on both sides; refuses on mismatch
#   wi_pod.sh podcheck <alias>               memory limit, CPU quota, bytes on /workspace, GPU; prints which schedule
#                                            fits: MODE lanes | MODE sequential | MODE none (RUNBOOK section 3)
#   wi_pod.sh job      <alias> <job> <timeout-s> [--force] [--nogpu] -- <command...>
#   wi_pod.sh chain    <alias> <chain-name> <chain-file>     jobs in order, detached (format of pod/m2_chain.sh)
#   wi_pod.sh status   <alias> [<job> ...]   one line per job and chain
#   wi_pod.sh fetch    <alias> <job>         rsync runs/<job>/ -> $WI_DATA/runs/<job>/, md5 both sides -> FETCH_OK|FETCH_PARTIAL|FETCH_MISMATCH
#   wi_pod.sh fetchall <alias>               fetch every run folder, the chain logs and meta/; one md5 list of everything
#   wi_pod.sh stop     <alias> <job> | --chain <name>        SIGTERM to that job's recorded process groups only (never pkill)
#   wi_pod.sh release  <alias>               only after FETCHALL_OK: pod/wi_release.sh removes this run's merged
#                                            checkpoints and the s2300 it downloaded, leaves dpo, writes /workspace/wi/GPU_FREE
# Global flag (any position): --dry-run   print the ssh / rsync commands instead of running them.
# env: WI_DATA (default /scratch/cowifornia/nd-rl/olmo_elicit/interp).
set -euo pipefail
DRY=0
args=()
for a in "$@"; do [[ $a == --dry-run ]] && DRY=1 || args+=("$a"); done
set -- "${args[@]}"
cmd=${1:?usage: wi_pod.sh <setup|push|podcheck|job|chain|status|fetch|fetchall|stop|release> <alias> ...}
h=${2:?usage: wi_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
P=$(cd "$(dirname "$0")" && pwd)     # .../olmo_weight_interp_20261128/pod
W=$(dirname "$P")                    # this experiment's folder
CUR=$(dirname "$W")                  # code/experiments/current
WI_DATA=${WI_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit/interp}
REMOTE=/workspace/wi
FORBIDDEN='textbook72|tb72|test_13_20|test_21_30|832|sealed'
CREDENTIAL='(^|/)\.env|\.netrc|(^|/)id_(rsa|ecdsa|ed25519|runpod|ynez|dsa)|\.pem$|\.key$|credential|secret|api[_-]?key|hf_token|(^|/)token\.(txt|json)$|(^|/)\.ssh/|authorized_keys'
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
utc() { date -u +%FT%TZ; }
x() { if [[ $DRY == 1 ]]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
xs() { if [[ $DRY == 1 ]]; then echo "+ ssh $h $1"; else "${SSH[@]}" "$h" "$1"; fi; }

stage() {  # stage <dir>: copy exactly the files of pod/STAGE_LIST.txt (+ $WI_DATA/inputs); refuses forbidden / credential names
  local ST=$1 MAN=$1.manifest.txt src dst
  rm -rf "$ST"; mkdir -p "$ST"
  while read -r src dst; do
    [[ -z $src || $src == \#* ]] && continue
    [[ -f $CUR/$src ]] || { echo "REFUSED: $CUR/$src is missing"; return 1; }
    mkdir -p "$ST/$(dirname "$dst")"; cp -p "$CUR/$src" "$ST/$dst"
  done < "$P/STAGE_LIST.txt"
  [[ -d $WI_DATA/inputs ]] || { echo "REFUSED: no $WI_DATA/inputs (run make_inputs.py)"; return 1; }
  mkdir -p "$ST/inputs"; cp -p "$WI_DATA"/inputs/* "$ST/inputs/"
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort) > "$MAN"
  local bad
  bad=$(grep -iE "$FORBIDDEN" "$MAN" || true)
  [[ -z $bad ]] || { echo "REFUSED: held-out names in the staged list:"; echo "$bad"; return 1; }
  bad=$(grep -iE "$CREDENTIAL" "$MAN" || true)
  [[ -z $bad ]] || { echo "REFUSED: credential-like names in the staged list:"; echo "$bad"; return 1; }
  (cd "$ST" && xargs -d '\n' sha256sum < "$MAN") > "$1.sha256.txt"
  echo "staged $(wc -l < "$MAN") files, $(du -sh "$ST" | cut -f1) for $h"
}

push() {
  local ST=$WI_DATA/pod_stage/$h MAN=$WI_DATA/pod_stage/$h.manifest.txt
  mkdir -p "$WI_DATA/pod_stage"
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
  out=$(xs "mkdir -p $REMOTE/runs/$job/logs && cd $REMOTE && (setsid nohup env M2_ROOT=$REMOTE FORCE=$force NOGPU=$nogpu bash pod/m2_job.sh $job $to -- $c >> runs/$job/logs/nohup.log 2>&1 < /dev/null &) ; sleep 4; tail -2 runs/$job/logs/runner.log 2>/dev/null")
  echo "$out"
  if [[ $DRY != 1 ]] && grep -qE "REFUSED_DONE|LOCKED|GPU_BAD" <<< "$(tail -1 <<< "$out")"; then echo "NOT STARTED: $job on $h ($(utc))"; exit 1; fi
  echo "started $job on $h: timeout=${to}s force=$force nogpu=$nogpu ($(utc))"
}

fetch_dir() {  # fetch_dir <remote dir under $REMOTE> <local dir> <tag>: rsync + md5 both sides; echoes OK|MISMATCH|EMPTY
  local rd=$1 L=$2 F=$WI_DATA/pod_stage/$h.fetch.$3
  mkdir -p "$L" "$(dirname "$F")"
  "${SSH[@]}" "$h" "cd $REMOTE/$rd 2>/dev/null && find . -type f ! -name '*.lock' | sed 's|^\./||' | sort" > "$F.list" || true
  [[ -s $F.list ]] || { echo EMPTY; return 0; }
  "${SSH[@]}" "$h" "cd $REMOTE/$rd && xargs -d '\n' md5sum" < "$F.list" > "$F.pod.md5"
  timeout 3600 "${RSYNC[@]}" --files-from="$F.list" "$h:$REMOTE/$rd/" "$L/"
  (cd "$L" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5"
  if cmp -s "$F.pod.md5" "$F.lab.md5"; then echo OK; else echo MISMATCH; fi
}

case $cmd in
  setup)
    xs "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1; nproc; free -g | sed -n 2p"
    push
    xs "mkdir -p $REMOTE/meta && cd $REMOTE && (setsid nohup env M2_ROOT=$REMOTE timeout 3000 bash pod/m2_setup.sh > meta/setup.log 2>&1 < /dev/null &) ; echo setup started"
    [[ ${1:-} == --no-wait || $DRY == 1 ]] && { echo "setup running detached; log $REMOTE/meta/setup.log"; exit 0; }
    echo "setup start $(utc) (about 2 to 10 min)"
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
  podcheck)
    # read only. A memory limit of "max" means the container has no limit of its own: the host's total is used.
    q='m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); t=$(awk "/MemTotal/{print \$2}" /proc/meminfo); [[ $m =~ ^[0-9]+$ && $m -lt $((t * 1024)) ]] || m=$((t * 1024)); echo "mem_bytes=$m"; echo "cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr " " /) nproc=$(nproc)"; echo "used_bytes=$(du -sb ${WI_VOLUME_ROOT:-/workspace} 2>/dev/null | cut -f1)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed "s/^/gpu process: /"'
    out=$(xs "$q"); echo "$out"
    [[ $DRY == 1 ]] && exit 0
    python3 "$W/cell_count.py" mode --mem-bytes "$(sed -n 's/^mem_bytes=//p' <<< "$out")" \
      --used-bytes "$(sed -n 's/^used_bytes=//p' <<< "$out" | grep -E '^[0-9]+$' || echo 0)" --volume-gb "${WI_VOLUME_GB:-500}" ;;
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
    L=$WI_DATA/runs/$job
    if [[ $DRY == 1 ]]; then echo "+ rsync $h:$REMOTE/runs/$job/ $L/ ; md5sum of every file on both sides"; exit 0; fi
    st=$("${SSH[@]}" "$h" "python3 $REMOTE/pod/m2_status.py --root $REMOTE $job 2>/dev/null | head -1" || true)
    r=$(fetch_dir "runs/$job" "$L" "$job")
    n=$(wc -l < "$WI_DATA/pod_stage/$h.fetch.$job.list")
    running=0; grep -q " state=RUNNING " <<< "$st " && running=1
    case $r in
      EMPTY) echo "FETCH_EMPTY $job: nothing under $REMOTE/runs/$job on $h"; exit 1 ;;
      OK) if [[ $running == 1 ]]; then echo "FETCH_PARTIAL $job: $n files md5-equal, but the job is still running ($(utc))"
          else echo "FETCH_OK $job: $n files md5-equal on $h and serrano -> $L ($(utc))"; fi ;;
      *) echo "FETCH_MISMATCH $job: files differ (running=$running; files still being written differ) ($(utc))"; exit 1 ;;
    esac ;;
  fetchall)
    if [[ $DRY == 1 ]]; then echo "+ for every folder of $h:$REMOTE/runs and for meta/: rsync to $WI_DATA, md5sum both sides"; exit 0; fi
    jobs=$("${SSH[@]}" "$h" "ls $REMOTE/runs 2>/dev/null")
    "${SSH[@]}" "$h" "python3 $REMOTE/pod/m2_status.py --root $REMOTE" > "$WI_DATA/pod_status_$h.txt" || true
    bad=0; : > "$WI_DATA/fetch_$h.md5"; rm -f "$WI_DATA/fetch_$h.ok"
    for job in $jobs; do
      [[ $job =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "skip odd name $job"; bad=1; continue; }
      r=$(fetch_dir "runs/$job" "$WI_DATA/runs/$job" "$job")
      echo "$r $job"; [[ $r == OK ]] || bad=1
      [[ -f $WI_DATA/pod_stage/$h.fetch.$job.lab.md5 ]] && sed "s|  |  runs/$job/|" "$WI_DATA/pod_stage/$h.fetch.$job.lab.md5" >> "$WI_DATA/fetch_$h.md5"
    done
    r=$(fetch_dir meta "$WI_DATA/pod_meta_$h" meta); echo "$r meta"; [[ $r == OK ]] || bad=1
    sed "s|  |  meta/|" "$WI_DATA/pod_stage/$h.fetch.meta.lab.md5" >> "$WI_DATA/fetch_$h.md5" 2>/dev/null || true
    if grep -q " state=RUNNING " "$WI_DATA/pod_status_$h.txt"; then echo "FETCHALL_PARTIAL: something is still running on $h"; exit 1; fi
    if [[ $bad == 0 ]]; then echo "$(utc) $(wc -l < "$WI_DATA/fetch_$h.md5") files" > "$WI_DATA/fetch_$h.ok"
      echo "FETCHALL_OK $h: $(wc -l < "$WI_DATA/fetch_$h.md5") files md5-equal on $h and serrano; list $WI_DATA/fetch_$h.md5 ($(utc))"
    else echo "FETCHALL_MISMATCH $h: see the lines above ($(utc))"; exit 1; fi ;;
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
    xs "cd $REMOTE && (setsid nohup env M2_ROOT=$REMOTE bash pod/m2_chain.sh $cn >> runs/_chains/$cn/log/nohup.log 2>&1 < /dev/null &) ; sleep 3; tail -2 runs/_chains/$cn/log/chain.log 2>/dev/null"
    echo "chain $cn started on $h: $n jobs ($(utc))" ;;
  stop)
    if [[ ${1:-} == --chain ]]; then
      cn=${2:?stop <alias> --chain <name>}; [[ $cn =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad chain name"; exit 2; }
      xs "M2_ROOT=$REMOTE bash $REMOTE/pod/m2_stop.sh --chain $cn"; exit 0
    fi
    job=${1:?stop <alias> <job> | stop <alias> --chain <name>}
    [[ $job =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name"; exit 2; }
    xs "M2_ROOT=$REMOTE bash $REMOTE/pod/m2_stop.sh $job" ;;
  release)
    # the last step on a shared pod: weights are deleted, so an md5-checked fetch of everything must be on record
    [[ -f $WI_DATA/fetch_$h.ok ]] || { echo "RELEASE REFUSED: no FETCHALL_OK on record for $h (run fetchall first)"; exit 1; }
    if [[ $DRY == 1 ]]; then echo "+ ssh $h 'cd $REMOTE && M2_ROOT=$REMOTE bash pod/wi_release.sh --fetched'"; exit 0; fi
    rc=0; "${SSH[@]}" "$h" "cd $REMOTE && M2_ROOT=$REMOTE bash pod/wi_release.sh --fetched" > "$WI_DATA/release_$h.txt" 2>&1 || rc=$?
    cat "$WI_DATA/release_$h.txt"
    if [[ $rc == 0 ]] && grep -q "^GPU_FREE " "$WI_DATA/release_$h.txt"; then echo "RELEASED $h: marker $REMOTE/GPU_FREE; report $WI_DATA/release_$h.txt ($(utc))"
    else echo "RELEASE INCOMPLETE on $h (rc=$rc): see the lines above ($(utc))"; exit 1; fi ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
