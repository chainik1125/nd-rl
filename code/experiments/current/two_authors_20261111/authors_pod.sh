#!/usr/bin/env bash
# authors_pod.sh: run the two-authors jobs on Runpod A40 pods, one job per pod. Run ON SERRANO, by the orchestrator,
# against a pod's ssh alias (`rp-*` in ~/.ssh/config, key ~/.ssh/id_runpod; add one with the knockout's
# `python3 knockout_danbase_20261104/register_pod.py ssh rp-<name> <ip> <port>`). This script never creates, stops or
# deletes a pod, never reads or prints an API key, and never copies a sealed set. Adapted from the knockout's
# podsetup.sh, runpod_copy_line.sh and podcheck.sh (knockout_danbase_20261104, 2026-11-05).
#
#   authors_pod.sh setup     <alias>                       install the env on a new pod (8-15 min; authors_pod_setup.sh)
#   authors_pod.sh stage     <alias> <job> <arg> [resume]  stage the job's files on serrano, list them; copy nothing
#   authors_pod.sh copy      <alias> <job> <arg> [resume]  stage, rsync to the pod (no --delete), md5 every file both sides
#   authors_pod.sh check-env <alias> [<base name>]         GPU, torch.cuda, Lean accepts one proof and rejects one, the
#                                                          CPU quota; with a base on the pod, a tiny measure on the GPU
#   authors_pod.sh start     <alias> <launcher args...>    authors_launch.py <args> --gpu 0 on the pod (detached job)
#   authors_pod.sh check     <alias>                       job status, newest log tail, GPU, disk, python processes
#   authors_pod.sh fetch     <alias> [--light]             pod TA/{bases,measure,raw,runs,logs} -> serrano TA, md5-checked
#                                                          (alarm checkpoints never; --light: no checkpoints at all)
#
# <job> <arg>:  pretrain <corpus>          {authors,control}_{half,p10}
#               measure  <base>            <corpus>_s<seed> (its base must be in serrano's TA/bases)
#               rl       <line> [resume]   rl_<corpus>[_b<b>]_s<seed>; `resume` also copies serrano's TA/runs/<line>/
#                                          (a later segment continues from one of its rolling checkpoints)
# What a job's copy holds (pod /workspace layout = serrano paths through authors_pod_setup.sh's links):
#   every job  exp/: two_authors_20261111/authors_{corpus,profile,sets,measure,gate,launch}.py (authors_corpus: the
#              corpus's detectors, which the profile calls); author_tags_20261110/{habits,
#              tag_core,tag_danread,tag_grpo,tag_launch,tag_line,tag_measure,tag_pretrain,tagcorpus}.py;
#              knockout_danbase_20261104/{common,gpu,merge,danread}.py
#              kd/dan_code_13e41bbd/: the 24 modules of Dan's tree these import, coe/{coe_assemble,coe_sigma}.py,
#              nd_verify/ (code only)
#              ta/sets/: the three measurement sets and their manifest (not test sets)
#   pretrain   ta/data/<corpus>.jsonl.gz (+ its manifest when present); Dan's data/p2/heldout.jsonl (validation loss)
#   measure    ta/bases/stage1_<base>.pt, .tag.json, .pt.args.json
#   rl         the base as for measure; kd/pools/dan_rl_targets.jsonl (Dan's 4,495) and canary_long.jsonl; kg/dan_axis.json
#              (Dan's ladders' x grid); the GRPO code subset of kd/snap/f35d3daf_danbase (as the knockout copied it);
#              Dan's data/coe/sigma_heldout_strict.txt and transfer_track832.jsonl TRIMMED TO THE 106 STRICT ROWS
#              (exactly as runpod_copy_line.sh does); with `resume`, serrano's ta/runs/<line>/ (alarm checkpoints left out)
# NOT copied: textbook72 / tb72, test_13_20, test_21_30 or any other sealed or evaluation set, the other 726 rows of
# transfer_track832, the generator pool, the other corpora and bases. The stage is checked for sealed names and the
# trimmed transfer file for exactly the 106 strict rows before anything leaves serrano.
set -euo pipefail
cmd=${1:?usage: authors_pod.sh <setup|stage|copy|check-env|start|check|fetch> <alias> ...}
h=${2:?usage: authors_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
W=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$W")
KD=/scratch/cowifornia/nd-rl/knockout_danbase
D=$KD/dan_code_13e41bbd
G=$KD/snap/f35d3daf_danbase
TA=${TA:-/scratch/cowifornia/nd-rl/two_authors}
KG=/scratch/cowifornia/nd-rl/knockout_grpo
ENVD=/scratch/cowifornia/nd-rl/think/nd-rl/code
PY=/workspace/venv/bin/python
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
DAN_MODULES="best_model eval_set expert_iter gen intuit lean_gate lean_judge lean_prefilter lean_tok minlen model
  nd2lean normalize patterns patterns2 prune record sample state_env state_ladder_ei state_sample state_train
  state_train_best tokenizer"
SEALED='textbook72|tb72|test_13_20|test_21_30'
utc() { date -u +%FT%TZ; }

stage() {  # stage <job> <arg> [resume] -> $ST (the pod's /workspace layout), $MAN (relative paths)
  local job=$1 arg=$2 resume=${3:-}
  ST=$TA/pod_stage/$h
  MAN=$TA/pod_stage/$h.manifest.txt
  rm -rf "$ST"
  mkdir -p "$ST"/exp/{two_authors_20261111,author_tags_20261110,knockout_danbase_20261104} \
    "$ST"/kd/dan_code_13e41bbd/{coe,nd_verify} "$ST"/ta/sets
  cp -p "$W"/authors_{corpus,profile,sets,measure,gate,launch}.py "$ST/exp/two_authors_20261111/"
  for f in habits tag_core tag_danread tag_grpo tag_launch tag_line tag_measure tag_pretrain tagcorpus; do
    cp -p "$EXP/author_tags_20261110/$f.py" "$ST/exp/author_tags_20261110/"; done
  for f in common gpu merge danread; do cp -p "$EXP/knockout_danbase_20261104/$f.py" "$ST/exp/knockout_danbase_20261104/"; done
  local T=$ST/kd/dan_code_13e41bbd
  for f in $DAN_MODULES; do cp -p "$D/$f.py" "$T/"; done
  cp -p "$D"/coe/{coe_assemble,coe_sigma}.py "$T/coe/"
  cp -p "$D"/nd_verify/{__init__,verify}.py "$T/nd_verify/"
  [[ -s $TA/sets/manifest.json ]] || { echo "no measurement sets in $TA/sets (authors_sets.py build)"; exit 2; }
  cp -p "$TA"/sets/{manifest.json,marker_plain.jsonl,route_unseen.jsonl,marker_heldout_class.jsonl} "$ST/ta/sets/"
  case $job in
    pretrain)
      [[ $arg =~ ^(authors|control)_(half|p10)$ ]] || { echo "pretrain <corpus>: {authors,control}_{half,p10}"; exit 2; }
      [[ -s $TA/data/$arg.jsonl.gz ]] || { echo "no corpus $TA/data/$arg.jsonl.gz"; exit 2; }
      mkdir -p "$ST/ta/data" "$T/data/p2"
      ln "$TA/data/$arg.jsonl.gz" "$ST/ta/data/" 2>/dev/null || cp -p "$TA/data/$arg.jsonl.gz" "$ST/ta/data/"
      [[ -s $TA/data/$arg.manifest.json ]] && cp -p "$TA/data/$arg.manifest.json" "$ST/ta/data/"
      cp -p "$D/data/p2/heldout.jsonl" "$T/data/p2/" ;;
    measure|rl)
      local base
      if [[ $job == measure ]]; then
        [[ $arg =~ ^(authors|control)_(half|p10)_s[0-9]+$ ]] || { echo "measure <base>: <corpus>_s<seed>"; exit 2; }
        base=$arg
      else
        [[ $arg =~ ^rl_((authors|control)_(half|p10))(_b([0-9]+))?_s([0-9]+)$ ]] || { echo "rl <line>: rl_<corpus>[_b<b>]_s<seed>"; exit 2; }
        base=${BASH_REMATCH[1]}_s${BASH_REMATCH[5]:-${BASH_REMATCH[6]}}
      fi
      for x in "stage1_$base.pt" "stage1_$base.tag.json"; do
        [[ -s $TA/bases/$x ]] || { echo "no $TA/bases/$x on serrano (fetch it from its pretraining pod first)"; exit 2; }; done
      mkdir -p "$ST/ta/bases"
      for x in "stage1_$base.pt" "stage1_$base.tag.json" "stage1_$base.pt.args.json"; do
        [[ -e $TA/bases/$x ]] && { ln "$TA/bases/$x" "$ST/ta/bases/" 2>/dev/null || cp -p "$TA/bases/$x" "$ST/ta/bases/"; }; done
      if [[ $job == rl ]]; then
        mkdir -p "$ST/kd/pools" "$ST/kg" "$T/data/coe" "$ST/kd/snap/f35d3daf_danbase/code/ndbench/core"
        local S=$ST/kd/snap/f35d3daf_danbase
        cp -p "$G/code/__init__.py" "$S/code/"
        cp -rp "$G/code/thinking" "$S/code/"
        cp -p "$G/code/ndbench/__init__.py" "$S/code/ndbench/"
        cp -p "$G"/code/ndbench/core/{__init__.py,keys.py,types.py} "$S/code/ndbench/core/"
        cp -p "$G"/COMMIT "$G"/COMMIT_PATCHED "$S/"
        for x in dan_rl_targets.jsonl canary_long.jsonl; do ln "$KD/pools/$x" "$ST/kd/pools/" 2>/dev/null || cp -p "$KD/pools/$x" "$ST/kd/pools/"; done
        cp -p "$KG/dan_axis.json" "$ST/kg/"
        cp -p "$D/data/coe/sigma_heldout_strict.txt" "$T/data/coe/"
        python3 - "$D" "$T" <<'EOF'
import json, os, sys
d, t = sys.argv[1], sys.argv[2]
strict = set(open(f"{d}/data/coe/sigma_heldout_strict.txt").read().split())
src = f"{d}/data/coe/transfer_track832.jsonl"
rows = [l for l in open(src) if json.loads(l)["name"] in strict]
assert len(rows) == 106, len(rows)
out = f"{t}/data/coe/transfer_track832.jsonl"
open(out, "w").writelines(rows)
m = os.stat(src).st_mtime
os.utime(out, (m, m))
EOF
        if [[ $resume == resume ]]; then
          [[ -d $TA/runs/$arg ]] || { echo "resume: no $TA/runs/$arg on serrano (fetch it first)"; exit 2; }
          mkdir -p "$ST/ta/runs"
          rsync -a --exclude '*_alarm.pt' "$TA/runs/$arg" "$ST/ta/runs/"
        fi
      fi ;;
    *) echo "job must be pretrain, measure or rl"; exit 2 ;;
  esac
  find "$ST" -name __pycache__ -prune -exec rm -rf {} +
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$MAN")
  if grep -i -E "$SEALED" "$MAN"; then echo "SEALED NAME IN THE STAGE: nothing copied"; exit 1; fi
  if [[ -e $T/data/coe/transfer_track832.jsonl ]]; then
    python3 - "$D/data/coe/sigma_heldout_strict.txt" "$T/data/coe/transfer_track832.jsonl" <<'EOF'
import json, sys
strict = set(open(sys.argv[1]).read().split())
names = [json.loads(l)["name"] for l in open(sys.argv[2])]
assert len(names) == 106 and set(names) == strict, "the staged transfer file is not exactly the 106 strict rows"
EOF
  fi
  echo "staged $(wc -l < "$MAN") files, $(du -shL "$ST" | cut -f1), for $job $arg on $h ($MAN)"
}

case $cmd in
  setup)
    "${SSH[@]}" "$h" 'mkdir -p /workspace/env && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
    "${RSYNC[@]}" "$W/authors_pod_setup.sh" "$h:/workspace/authors_pod_setup.sh"
    "${RSYNC[@]}" "$ENVD/pyproject.toml" "$ENVD/uv.lock" "$h:/workspace/env/"
    "${SSH[@]}" "$h" 'cd /workspace/env && md5sum pyproject.toml uv.lock' | sed 's/^/pod: /'
    (cd "$ENVD" && md5sum pyproject.toml uv.lock) | sed 's/^/lab: /'
    echo "setup start $(utc) (8 to 15 min; log /workspace/setup.log on the pod)"
    timeout 3900 "${SSH[@]}" "$h" 'bash /workspace/authors_pod_setup.sh; tail -2 /workspace/setup.log'
    "${SSH[@]}" "$h" 'grep -q "^setup done" /workspace/setup.log' || { echo "SETUP FAILED: see /workspace/setup.log on $h"; exit 1; }
    echo "setup done $(utc)" ;;
  stage)
    stage "$@" ;;
  copy)
    stage "$@"
    if [[ ${DRY_RUN:-} == 1 ]]; then echo "DRY_RUN=1: nothing copied"; exit 0; fi
    "${SSH[@]}" "$h" 'mkdir -p /workspace'
    timeout 3600 "${RSYNC[@]}" "$ST/" "$h:/workspace/"
    (cd "$ST" && xargs -d '\n' md5sum < "$MAN") > "$MAN.lab.md5"
    "${SSH[@]}" "$h" 'cd /workspace && xargs -d "\n" md5sum' < "$MAN" > "$MAN.pod.md5"
    if ! cmp -s "$MAN.lab.md5" "$MAN.pod.md5"; then
      diff "$MAN.lab.md5" "$MAN.pod.md5" | head -20
      echo "MD5 MISMATCH: do not start the job; rerun the copy"; exit 1
    fi
    echo "md5: all $(wc -l < "$MAN.lab.md5") files equal on serrano and $h ($(utc))"
    grep -E "ta/(data|bases)/|kd/pools/" "$MAN.lab.md5" | sed -E 's/^(.{8})[0-9a-f]+/\1…/' || true ;;
  check-env)
    base=${1:-}
    timeout 900 "${SSH[@]}" "$h" bash -s -- "$base" <<'EOF'
set -euo pipefail
base=${1:-}  # ssh drops an empty argument
PY=/workspace/venv/bin/python
mkdir -p /workspace/check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; x=torch.randn(512,512,device='cuda'); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), bool(torch.isfinite(x@x).all()))"
/root/.elan/bin/lean --version
cd /workspace/kd/dan_code_13e41bbd
LEAN_GATE_LOG=/workspace/check/lean_gate.jsonl $PY -c "
import lean_gate as g
print('cpu quota', g.cpu_quota())
v = g.lean_check([('theorem chk (p : Prop) (hp : p) : p :=', 'by exact hp'), ('theorem chk (p q : Prop) (hp : p) : q :=', 'by exact hp')])[0]
print('lean verdicts (want [True, False]):', v)
assert v == [True, False], v
"
cd /workspace/exp/two_authors_20261111
$PY -c "
import authors_profile as AP
p = AP.profile('THM ( ( ~ Q ) > F ) SEQ Q PRF', 'N1 ( ( ~ Q ) > F ) : PR ; N2 | ( ~ Q ) : AS ; N3 | F : IMPE N1 N2 ; N4 ( ~ ( ~ Q ) ) : NEGI N2 N3 ; N5 Q : DN N4 ; QED')
assert p['ok'] and p['route'] is True, p
print('profile ok (route detected)')
"
wc -l /workspace/ta/sets/*.jsonl | tail -4  # the full check needs the exclusion files, which stay on serrano; the copy is md5-checked
if [[ -d /workspace/kd/snap/f35d3daf_danbase/code/thinking ]]; then  # an RL job's copy: GRPO's code imports
  $PY -c "import sys; sys.path.insert(0, '/workspace/exp/author_tags_20261110'); import tag_core as T; T.install_thinking(); from code.thinking import grpo; print('grpo import ok')"
fi
if [[ -n $base ]]; then
  rm -rf /workspace/check/m /workspace/check/r
  ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=0 timeout 600 $PY authors_measure.py run \
    --ckpt /workspace/ta/bases/stage1_$base.pt --ckpt_id check --tags A,B,none --sets marker_plain --limit 4 --k 2 \
    --device cuda --out_dir /workspace/check/m --raw_dir /workspace/check/r > /workspace/check/measure.log 2>&1
  grep -E "^check " /workspace/check/measure.log
fi
echo PODCHECK_OK
EOF
    ;;
  start)
    [[ $# -ge 1 ]] || { echo "start <alias> <pretrain|measure|rl> <launcher args>"; exit 2; }
    args=$(printf '%q ' "$@")
    "${SSH[@]}" "$h" "cd /workspace/exp/two_authors_20261111 && $PY authors_launch.py $args --gpu 0" ;;
  check)
    "${SSH[@]}" "$h" bash -s <<'EOF'
PY=/workspace/venv/bin/python
echo "== $(date -u +%FT%TZ) $(hostname)"
cd /workspace/exp/two_authors_20261111 && $PY authors_launch.py status 2>&1 | tail -25
L=$(ls -t /workspace/ta/logs/*.log 2>/dev/null | grep -v watchdog | head -1)
[[ -n $L ]] && { echo "== newest log $L"; tail -8 "$L"; }
echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "== disk"; df -h /workspace | tail -1
echo "== python processes"; ps -eo pid,etime,args | grep -E "authors_|tag_|grpo|state_train|danread" | grep -v grep | cut -c1-160
EOF
    ;;
  fetch)
    light=${1:-}
    ex="! -name '*_alarm.pt'"
    [[ $light == --light ]] && ex="! -name '*.pt'"
    F=$TA/pod_stage/$h.fetch
    mkdir -p "$(dirname "$F")"
    "${SSH[@]}" "$h" "cd /workspace/ta && find bases measure raw runs logs -type f $ex 2>/dev/null | sort" > "$F.list" || true
    [[ -s $F.list ]] || { echo "nothing to fetch on $h"; exit 0; }
    "${SSH[@]}" "$h" 'cd /workspace/ta && xargs -d "\n" md5sum' < "$F.list" > "$F.pod.md5"
    # a base or a measurement already on serrano must be the same file (they are never overwritten)
    conflict=0
    while read -r sum path; do
      if [[ $path =~ ^(bases|measure)/ && -e $TA/$path ]]; then
        [[ $(md5sum < "$TA/$path" | cut -d' ' -f1) == "$sum" ]] || { echo "CONFLICT: $TA/$path differs from $h's"; conflict=1; }
      fi
    done < "$F.pod.md5"
    [[ $conflict == 0 ]] || { echo "nothing fetched (resolve the conflicts by hand)"; exit 1; }
    timeout 7200 "${RSYNC[@]}" --files-from="$F.list" "$h:/workspace/ta/" "$TA/"
    (cd "$TA" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5"
    if cmp -s "$F.pod.md5" "$F.lab.md5"; then
      echo "md5: all $(wc -l < "$F.list") fetched files equal on $h and serrano ($(utc))"
    else
      diff "$F.pod.md5" "$F.lab.md5" | grep '^[<>]' | head -20
      echo "MD5 DIFFERENCES (expected only for files a running job is still writing: logs, status, metrics); fetch again after the job ends"
      exit 1
    fi ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
