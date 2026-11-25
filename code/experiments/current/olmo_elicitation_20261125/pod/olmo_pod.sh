#!/usr/bin/env bash
# olmo_pod.sh: drive one RunPod H200 pod for the olmo elicitation curves. Run ON SERRANO against an ssh alias rp-<name>
# (~/.ssh/config, key ~/.ssh/id_runpod). Never creates, stops or deletes a pod; never reads or prints a key or token;
# no HF token is needed (models and Dan's bucket are public). Modelled on Dan's pod/olmo_ifcurve/ (setup.sh, job.sh,
# worker.sh, start.sh) and on two_authors_20261111/authors_pod.sh (ssh-alias driven, md5-checked copies).
#
#   olmo_pod.sh setup  <alias> [--no-wait]       push our folder + inputs to /workspace/oe (md5), run pod/oe_setup.sh
#                                                detached (Dan's versions + peft; ~10 min), wait for SETUP_DONE
#   olmo_pod.sh push   <alias>                   push again (code / inputs changed); md5 both sides
#   olmo_pod.sh run    <alias> <ckpt> [--stages edl,T,H,L,rep] [--edl-args "<flags>"] [--n-h 8|4] [--keep-weights]
#                      [--lora-check]            (stage L: stage_adapter.py checks runs/<ckpt>/edl and stages
#                                                best_order0 + source.json, pushed md5-checked; --lora-check runs
#                                                lora_check.py at this checkpoint too; at dpo it always runs)
#                      [--verify-sha]            (--stages download: fetch + verify only, weights kept)
#                                                push, then start pod/oe_chain.sh for ONE checkpoint under
#                                                setsid + nohup + timeout (download -> edl -> T -> H -> rep -> weights
#                                                deleted); default stages edl,T,H; default --edl-args "--max-tok 6144"
#   olmo_pod.sh status <alias> [<ckpt> ...]      one line per stage: state, progress, heartbeat age, GPU MiB, free GB
#   olmo_pod.sh fetch  <alias> <ckpt>            rsync /workspace/oe/runs/<ckpt>/ -> $OE_DATA/runs/<ckpt>/, md5 both
#                                                sides, one-line verdict (FETCH_OK / FETCH_PARTIAL / FETCH_MISMATCH)
#   olmo_pod.sh stop   <alias> [<ckpt>]          SIGTERM to the recorded chain's process group (never pkill -f)
# Global flag (any position): --dry-run   print the ssh / rsync commands instead of running them.
set -euo pipefail
DRY=0
args=()
for a in "$@"; do [[ $a == --dry-run ]] && DRY=1 || args+=("$a"); done
set -- "${args[@]}"
cmd=${1:?usage: olmo_pod.sh <setup|push|run|status|fetch|stop> <alias> ...}
h=${2:?usage: olmo_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
P=$(cd "$(dirname "$0")" && pwd)     # .../olmo_elicitation_20261125/pod
X=$(dirname "$P")                    # our experiment folder
OE_DATA=${OE_DATA:-/scratch/cowifornia/nd-rl/olmo_elicit}
DAN=$OE_DATA/ext/nd-takehome
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

push() {  # stage our folder (code), stage2 inputs and the EDL pool; rsync to /workspace/oe; md5 every file both sides
  local ST=$OE_DATA/pod_stage/$h MAN=$OE_DATA/pod_stage/$h.manifest.txt
  rm -rf "$ST"; mkdir -p "$ST/pod" "$ST/stage2"
  (cd "$X" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.json' -o -name '*.csv' \) \
     -exec cp -p {} "$ST/" \;)
  cp -p "$P"/*.sh "$P"/*.py "$ST/pod/"
  [[ -d $OE_DATA/stage2 ]] && cp -rp "$OE_DATA/stage2/." "$ST/stage2/"
  if [[ -d $OE_DATA/pool ]]; then mkdir -p "$ST/pool"; cp -rp "$OE_DATA/pool/." "$ST/pool/"; fi
  if [[ -n ${ADAPTER_CK:-} ]]; then  # stage L: this checkpoint's staged adapter (hard links, same filesystem)
    mkdir -p "$ST/adapters"; cp -al "$OE_DATA/adapters_stage/$ADAPTER_CK" "$ST/adapters/$ADAPTER_CK"
  fi
  find "$ST" -name __pycache__ -prune -exec rm -rf {} +
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$MAN")
  echo "staged $(wc -l < "$MAN") files, $(du -sh "$ST" | cut -f1) for $h"
  xs "mkdir -p /workspace/oe"
  x timeout 1800 "${RSYNC[@]}" "$ST/" "$h:/workspace/oe/"
  [[ $DRY == 1 ]] && { echo "+ (md5 check of $(wc -l < "$MAN") files)"; return 0; }
  (cd "$ST" && xargs -d '\n' md5sum < "$MAN") > "$MAN.lab.md5"
  "${SSH[@]}" "$h" 'cd /workspace/oe && xargs -d "\n" md5sum' < "$MAN" > "$MAN.pod.md5"
  if ! cmp -s "$MAN.lab.md5" "$MAN.pod.md5"; then
    diff "$MAN.lab.md5" "$MAN.pod.md5" | head -20; echo "PUSH MD5 MISMATCH: nothing started"; exit 1
  fi
  echo "push md5 ok: $(wc -l < "$MAN") files equal on serrano and $h ($(utc))"
}

case $cmd in
  setup)
    xs "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1"
    push
    xs "mkdir -p /workspace/oe/meta && cd /workspace/oe && (setsid nohup timeout 3000 bash pod/oe_setup.sh > meta/setup.log 2>&1 < /dev/null &) ; echo setup started"
    [[ ${1:-} == --no-wait || $DRY == 1 ]] && { echo "setup running detached; log /workspace/oe/meta/setup.log"; exit 0; }
    echo "setup start $(utc) (about 10 min)"
    for _ in $(seq 1 100); do
      sleep 30
      out=$("${SSH[@]}" "$h" 'tail -1 /workspace/oe/meta/setup.log; pgrep -f "pod/oe_setu[p].sh" > /dev/null && echo ALIVE || echo GONE' 2>/dev/null || echo "SSH_FAIL")
      if grep -q "SETUP_DONE" <<< "$out"; then
        "${SSH[@]}" "$h" 'grep -E "CODE_OK|VERSIONS|NLTK_OK|GRADER_OK|GPU_TEST|GPU_OK|GPU_BAD|SETUP_DONE" /workspace/oe/meta/setup.log'
        echo "setup done $(utc)"; exit 0
      fi
      if grep -q "^GONE" <<< "$out"; then
        "${SSH[@]}" "$h" 'tail -25 /workspace/oe/meta/setup.log'; echo "SETUP FAILED ($(utc)): see /workspace/oe/meta/setup.log"; exit 1
      fi
    done
    echo "SETUP TIMEOUT (50 min)"; exit 1 ;;
  push)
    push ;;
  run)
    ck=${1:?run <alias> <ckpt> [--stages ...]}; shift
    [[ " $CKPTS " == *" $ck "* ]] || { echo "unknown checkpoint $ck (one of: $CKPTS)"; exit 2; }
    stages=edl,T,H; edl_args="--max-tok 6144"; nh=8; keep=0; vsha=0; lcheck=0
    while [[ $# -gt 0 ]]; do
      case $1 in
        --stages) stages=$2; shift 2 ;;
        --edl-args) edl_args=$2; shift 2 ;;
        --n-h) nh=$2; shift 2 ;;
        --keep-weights) keep=1; shift ;;
        --verify-sha) vsha=1; shift ;;
        --lora-check) lcheck=1; shift ;;
        *) echo "unknown option $1"; exit 2 ;;
      esac
    done
    [[ $stages =~ ^(edl|T|H|L|rep)(,(edl|T|H|L|rep))*$ || $stages == download ]] || { echo "--stages: comma list of edl,T,H,L,rep, or download"; exit 2; }
    [[ $nh == 8 || $nh == 4 ]] || { echo "--n-h 8 or 4"; exit 2; }
    [[ $edl_args =~ ^[-A-Za-z0-9_.,=/\ ]*$ ]] || { echo "--edl-args: plain flags only"; exit 2; }
    # inputs present on serrano for the asked stages
    [[ ,$stages, == *,edl,* ]] && { [[ -d $OE_DATA/pool ]] || { echo "edl asked but no $OE_DATA/pool"; exit 2; }; }
    [[ ,$stages, == *,T,* ]] && { [[ -s $OE_DATA/stage2/T_$ck.jsonl.gz ]] || { echo "no $OE_DATA/stage2/T_$ck.jsonl.gz (make_stage2.py)"; exit 2; }; }
    [[ ,$stages, == *,H,* ]] && { [[ -s $OE_DATA/stage2/pids_H_n$nh.txt ]] || { echo "no H inputs (make_stage2.py)"; exit 2; }; }
    [[ ,$stages, == *,rep,* ]] && { [[ -s $OE_DATA/stage2/pids_rep.txt ]] || { echo "no rep inputs (make_stage2.py)"; exit 2; }; }
    if [[ ,$stages, == *,L,* ]]; then
      [[ -s $OE_DATA/stage2/pids_L.txt && -s $OE_DATA/stage2/seeds_L.csv ]] || { echo "no L inputs (make_stage2.py)"; exit 2; }
      python3 "$X/stage_adapter.py" "$ck" "$OE_DATA/adapters_stage/$ck" --runs "$OE_DATA/runs" || { echo "stage L refused for $ck"; exit 2; }
      ADAPTER_CK=$ck
    fi
    grep -q "^$ck," "$OE_DATA/stage2/seeds.csv" || { echo "no seeds for $ck in $OE_DATA/stage2/seeds.csv"; exit 2; }
    # one chain per pod: refuse if any chain is alive
    if [[ $DRY != 1 ]]; then
      alive=$("${SSH[@]}" "$h" 'python3 /workspace/oe/pod/oe_status.py 2>/dev/null | grep " chain alive=1" || true')
      [[ -z $alive ]] || { echo "a chain is running on $h: $alive"; exit 1; }
      "${SSH[@]}" "$h" 'test -f /workspace/oe/meta/setup.done' || { echo "no setup.done on $h: run setup first"; exit 1; }
    fi
    push
    TO_DOWNLOAD=${TO_DOWNLOAD:-2400}; TO_EDL=${TO_EDL:-14400}; TO_T=${TO_T:-3600}; TO_H=${TO_H:-7200}; TO_REP=${TO_REP:-7200}
    total=$((TO_DOWNLOAD + 600))
    [[ ,$stages, == *,edl,* ]] && total=$((total + TO_EDL))
    [[ ,$stages, == *,T,* ]] && total=$((total + TO_T + 1800))
    [[ ,$stages, == *,H,* ]] && total=$((total + TO_H + 1800))
    [[ ,$stages, == *,rep,* ]] && total=$((total + TO_REP + 1800))
    TO_L=${TO_L:-7200}
    [[ ,$stages, == *,L,* ]] && total=$((total + TO_L + 3600))
    envs="VERIFY_SHA=$vsha KEEP_WEIGHTS=$keep N_H=$nh TO_DOWNLOAD=$TO_DOWNLOAD TO_EDL=$TO_EDL TO_T=$TO_T TO_H=$TO_H TO_REP=$TO_REP TO_L=$TO_L L_CHECK=$lcheck"
    out=$(xs "mkdir -p /workspace/oe/runs/$ck/logs && cd /workspace/oe && (setsid nohup timeout --signal=TERM --kill-after=120 $total env $envs EDL_ARGS='$edl_args' bash pod/oe_chain.sh $ck $stages >> runs/$ck/logs/nohup.log 2>&1 < /dev/null &) ; sleep 3; tail -3 runs/$ck/logs/chain.log 2>/dev/null")
    echo "$out"
    if grep -q "LOCKED\|GPU_BAD" <<< "$(tail -1 <<< "$out")"; then echo "NOT STARTED: $ck on $h ($(utc))"; exit 1; fi
    echo "started $ck on $h: stages=$stages n_h=$nh edl_args='$edl_args' timeout=${total}s ($(utc))" ;;
  status)
    xs "python3 /workspace/oe/pod/oe_status.py $*" ;;
  fetch)
    ck=${1:?fetch <alias> <ckpt>}
    [[ " $CKPTS " == *" $ck "* ]] || { echo "unknown checkpoint $ck"; exit 2; }
    L=$OE_DATA/runs/$ck; F=$OE_DATA/pod_stage/$h.fetch.$ck
    mkdir -p "$L" "$(dirname "$F")"
    if [[ $DRY == 1 ]]; then
      echo "+ ssh $h 'cd /workspace/oe/runs/$ck && find . -type f | sort' > $F.list; md5sum both sides"
      echo "+ rsync --files-from=$F.list $h:/workspace/oe/runs/$ck/ $L/"; exit 0
    fi
    alive=$("${SSH[@]}" "$h" "python3 /workspace/oe/pod/oe_status.py $ck 2>/dev/null | grep -c ' chain alive=1' || true")
    "${SSH[@]}" "$h" "cd /workspace/oe/runs/$ck && find . -type f ! -name '*.lock' | sed 's|^\./||' | sort" > "$F.list" || true
    [[ -s $F.list ]] || { echo "FETCH_EMPTY $ck: nothing under /workspace/oe/runs/$ck on $h"; exit 1; }
    "${SSH[@]}" "$h" "cd /workspace/oe/runs/$ck && xargs -d '\n' md5sum" < "$F.list" > "$F.pod.md5"
    timeout 3600 "${RSYNC[@]}" --files-from="$F.list" "$h:/workspace/oe/runs/$ck/" "$L/"
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
    xs "cd /workspace/oe/runs && for d in ${ck:-*}; do f=\$d/state/chain.pid; [ -f \$f ] || continue; set -- \$(cat \$f); [ -f \$d/state/chain.DONE ] && continue; kill -0 \$1 2>/dev/null || continue; echo \"SIGTERM chain \$d pid=\$1 pgid=\$2\"; kill -TERM -- -\$2 2>/dev/null || kill -TERM \$1; done; sleep 20; python3 /workspace/oe/pod/oe_status.py ${ck} | grep ' chain '" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
