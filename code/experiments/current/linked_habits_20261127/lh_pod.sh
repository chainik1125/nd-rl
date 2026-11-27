#!/usr/bin/env bash
# lh_pod.sh: linked habits version 2 on Runpod, ONE POD PER CORPUS (SPEC2.md "Pods"). Run ON SERRANO by the
# orchestrator against a pod's ssh alias (`rp-*` in ~/.ssh/config, key ~/.ssh/id_runpod; add one with
# `python3 ../knockout_danbase_20261104/register_pod.py ssh rp-<name> <ip> <port>`). This script never creates, stops or
# deletes a pod, never reads or prints an API key or any credential, and never copies a sealed set. Adapted from
# ../two_authors_20261111/authors_pod.sh (commit 9618763): the same setup, staging, md5 checks, sealed-name grep and
# 106-row trim; one copy holds everything a corpus needs (both stages), and `start` runs the whole corpus with
# lh_driver.sh, which stops at the gate until `go` (or `nogo`).
#
#   lh_pod.sh setup     <alias>                 install the env on a new pod (8-15 min; lh_pod_setup.sh)
#   lh_pod.sh stage     <alias> <corpus>        stage the corpus's files on serrano and list them; copy nothing
#   lh_pod.sh copy      <alias> <corpus>        stage, rsync to the pod (no --delete), md5 every file on both sides
#                                               (DRY_RUN=1: stage and list only)
#   lh_pod.sh check-env <alias> [<base>]        GPU, torch.cuda, Lean accepts one proof and rejects one, the 24-habit
#                                               detector, a twin, the trainers' and GRPO's imports; with a base on the
#                                               pod (e.g. c24_s0), a 4-theorem measurement on the GPU
#   lh_pod.sh start     <alias> <corpus>        start lh_driver.sh <corpus> on the pod, detached, under `timeout`
#   lh_pod.sh rl2       <alias> c24 [--dry-run] PREREG Amendment 10: lh_rl2.sh (c24's GRPO lines, RL seed 1, first base)
#                                               detached under `timeout` (TOTAL_H 9); after c24.BASE2_DONE; c24.RL2_DONE
#                                               (TOTAL_H, default 48 h); on the c4 pod also the Nov 14 bases; prints PID
#   lh_pod.sh go|nogo   <alias> <corpus>        after the gate: let the driver run stage 2 (go) or stop it (nogo)
#   lh_pod.sh job       <alias> <launcher args> one lh_launch.py job on the pod, --gpu 0, ND_GUARD=off (e.g. a resumed
#                                               GRPO segment: rl --line L --seg 2 --resume ...); detached
#   lh_pod.sh check     <alias>                 heartbeat, markers, driver log tail, jobs, GPU, disk, processes
#   lh_pod.sh fetch     <alias> [--light|--keep] pod LH/{bases,measure,raw,runs,logs,pod,pairs,arms} and the anchor files
#                                               -> serrano LH, md5-checked (--light: no checkpoints; --keep: the bases,
#                                               each run's final.pt and rm.pt, the measured snapshots, every GRPO
#                                               snapshot; the other checkpoints are counted and listed, not fetched).
#                                               Any failed, cut-off or empty listing fails the fetch; the last line
#                                               is FETCH_OK <alias> mode=<m> files=<n> ckpt=<k> left_on_pod=<j> (<GB>)
#                                               only on full success
#
# What a corpus's copy holds (pod /workspace = serrano paths through lh_pod_setup.sh's links):
#   exp/linked_habits_20261127/: this runner (lh_core, lh_launch, lh_measure, lh_pairs, lh_rlvar, lh_stats, lh_gate,
#     lh_read, lh_driver.sh), the corpus agent's detector and twins (lh_profile, lh_habits, lh_twin, notes), the third
#     agent's trainers (lh_tune, lh_dpo, lh_sft, lh_rm)
#   exp/two_authors_20261111/authors_{corpus,profile,sets,measure,gate,launch}.py, rlvar.py
#   exp/author_tags_20261110/{habits,tag_core,tag_danread,tag_grpo,tag_launch,tag_line,tag_measure,tag_pretrain,tagcorpus}.py
#   exp/knockout_danbase_20261104/{common,gpu,merge,danread}.py
#   kd/dan_code_13e41bbd/: the modules of Dan's tree these import, coe/{coe_assemble,coe_sigma}.py, nd_verify/ (code),
#     data/p2/heldout.jsonl (validation loss), data/coe/sigma_heldout_strict.txt and transfer_track832.jsonl TRIMMED TO
#     THE 106 STRICT ROWS (as W2's pod copy)
#   kd/pools/{dan_rl_targets,canary_long}.jsonl; kd/snap/f35d3daf_danbase: the GRPO code subset; kg/dan_axis.json
#   lh/sets/{marker_plain,route_unseen}.jsonl + manifest.json (measurement sets, not test sets)
#   lh/data/<corpus>.jsonl.gz (+ manifest), note_table.json, habit_table.json, rewarded_class.json, rl_pool_premises.json,
#     pretrain_steps.json; on the c0 pod also c0n.jsonl.gz (+ manifest; its steps: `extra.c0n` in pretrain_steps.json); on the c4 pod also
#     lh/bases/stage1_{authors,control}_half_s0.{pt,tag.json,pt.args.json}
# NOT copied: textbook72 / tb72, test_13_20, test_21_30 or any other sealed or evaluation set, the other 726 rows of
# transfer_track832, the generator pool, other corpora, other bases, any credential. The stage is grep-checked for
# sealed names and the trimmed transfer file for exactly the 106 strict rows before anything leaves serrano.
set -euo pipefail
cmd=${1:?usage: lh_pod.sh <setup|stage|copy|check-env|start|job|go|nogo|check|fetch> <alias> ...}
h=${2:?usage: lh_pod.sh $cmd <alias> ...}
[[ $h =~ ^rp-[a-z0-9-]+$ ]] || { echo "alias must look like rp-<name> (an entry of ~/.ssh/config)"; exit 2; }
shift 2
W=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$W")
KD=/scratch/cowifornia/nd-rl/knockout_danbase
D=$KD/dan_code_13e41bbd
G=$KD/snap/f35d3daf_danbase
LH=${LH:-/scratch/cowifornia/nd-rl/linked_habits}
KG=/scratch/cowifornia/nd-rl/knockout_grpo
ENVD=/scratch/cowifornia/nd-rl/think/nd-rl/code
PODW=/workspace/exp/linked_habits_20261127
PY=/workspace/venv/bin/python
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60)
RSYNC=(rsync -rlpt -e "ssh -o BatchMode=yes -o ConnectTimeout=15")
DAN_MODULES="best_model eval_set expert_iter gen intuit lean_check lean_gate lean_judge lean_prefilter lean_tok minlen model
  nd2lean normalize patterns patterns2 prune record sample state_env state_ladder_ei state_sample state_train
  state_train_best tokenizer"
MINE="lh_core lh_launch lh_measure lh_pairs lh_rlvar lh_stats lh_gate lh_read"
THEIRS="lh_profile lh_habits lh_twin notes lh_tune lh_dpo lh_sft lh_rm"
CORPORA_RE=${CORPORA_RE:-^(c24|c9|c4|c0)$}
NOV14_POD=${NOV14_POD:-c4}
declare -A EXTRA_OF=([c0]=${EXTRA_C0-c0n})  # extra bases per pod (lh_core.EXTRA_BASES); EXTRA_C0= stages none
SEALED='textbook72|tb72|test_13_20|test_21_30'
utc() { date -u +%FT%TZ; }
corpus_arg() { [[ ${1:-} =~ $CORPORA_RE ]] || { echo "<corpus> must match $CORPORA_RE"; exit 2; }; }

stage() {  # stage <corpus> -> $ST (the pod's /workspace layout), $MAN (relative paths)
  local c=$1
  corpus_arg "$c"
  ST=$LH/pod_stage/$h
  MAN=$LH/pod_stage/$h.manifest.txt
  rm -rf "$ST"
  mkdir -p "$ST"/exp/{linked_habits_20261127,two_authors_20261111,author_tags_20261110,knockout_danbase_20261104} \
    "$ST"/kd/dan_code_13e41bbd/{coe,nd_verify,data/p2,data/coe} "$ST"/kd/pools "$ST"/kg "$ST"/lh/{sets,data} \
    "$ST"/kd/snap/f35d3daf_danbase/code/ndbench/core
  for f in $MINE $THEIRS; do
    [[ -s $W/$f.py ]] || { echo "no $W/$f.py"; exit 2; }
    cp -p "$W/$f.py" "$ST/exp/linked_habits_20261127/"; done
  cp -p "$W/lh_driver.sh" "$W/lh_rl2.sh" "$ST/exp/linked_habits_20261127/"
  cp -p "$EXP"/two_authors_20261111/{authors_corpus,authors_profile,authors_sets,authors_measure,authors_gate,authors_launch,rlvar}.py \
    "$ST/exp/two_authors_20261111/"
  for f in habits tag_core tag_danread tag_grpo tag_launch tag_line tag_measure tag_pretrain tagcorpus; do
    cp -p "$EXP/author_tags_20261110/$f.py" "$ST/exp/author_tags_20261110/"; done
  for f in common gpu merge danread; do cp -p "$EXP/knockout_danbase_20261104/$f.py" "$ST/exp/knockout_danbase_20261104/"; done
  local T=$ST/kd/dan_code_13e41bbd
  for f in $DAN_MODULES; do [[ -e $D/$f.py ]] && cp -p "$D/$f.py" "$T/"; done
  cp -p "$D"/coe/{coe_assemble,coe_sigma}.py "$T/coe/"
  cp -p "$D"/nd_verify/{__init__,verify}.py "$T/nd_verify/"
  cp -p "$D/data/p2/heldout.jsonl" "$T/data/p2/"
  # the measurement sets (lh_measure.py sets put W2's two sets here), the corpus and its tables
  [[ -s $LH/sets/manifest.json && -s $LH/sets/marker_plain.jsonl && -s $LH/sets/route_unseen.jsonl ]] \
    || { echo "no measurement sets in $LH/sets (lh_measure.py sets)"; exit 2; }
  cp -p "$LH"/sets/{manifest.json,marker_plain.jsonl,route_unseen.jsonl} "$ST/lh/sets/"
  local b
  for b in $c ${EXTRA_OF[$c]:-}; do  # the pod's corpora: its own, and on the c0 pod c0n (with its own step file)
    [[ -s $LH/data/$b.jsonl.gz ]] || { echo "no corpus $LH/data/$b.jsonl.gz"; exit 2; }
    ln "$LH/data/$b.jsonl.gz" "$ST/lh/data/" 2>/dev/null || cp -p "$LH/data/$b.jsonl.gz" "$ST/lh/data/"
    [[ -s $LH/data/$b.manifest.json ]] && cp -p "$LH/data/$b.manifest.json" "$ST/lh/data/"
    if [[ $b != "$c" ]]; then  # its step count: `extra.<b>` in pretrain_steps.json (staged below)
      grep -q "\"$b\": {" "$LH/data/pretrain_steps.json" 2>/dev/null \
        || { echo "no step count for $b in $LH/data/pretrain_steps.json (lh_launch.py steps --extra $b)"; exit 2; }
    fi
  done
  for x in note_table.json habit_table.json rewarded_class.json rl_pool_premises.json pretrain_steps.json; do
    [[ -s $LH/data/$x ]] || { echo "no $LH/data/$x (pretrain_steps: lh_launch.py steps once every corpus exists)"; exit 2; }
    cp -p "$LH/data/$x" "$ST/lh/data/"; done
  if [[ $c == "$NOV14_POD" ]]; then  # the performance control's bases (lh_launch.py nov14 put them in LH/bases)
    mkdir -p "$ST/lh/bases"
    for b in authors_half control_half; do for s in .pt .tag.json .pt.args.json; do
      [[ -s $LH/bases/stage1_${b}_s0$s ]] || { echo "no $LH/bases/stage1_${b}_s0$s (lh_launch.py nov14)"; exit 2; }
      ln "$LH/bases/stage1_${b}_s0$s" "$ST/lh/bases/" 2>/dev/null || cp -p "$LH/bases/stage1_${b}_s0$s" "$ST/lh/bases/"; done; done
  fi
  # GRPO: the code subset of the snapshot (as W2 and the knockout copied it), Dan's pool and canary, the x grid
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
  find "$ST" -name __pycache__ -prune -exec rm -rf {} +
  (cd "$ST" && find . -type f | sed 's|^\./||' | sort > "$MAN")
  if grep -i -E "$SEALED" "$MAN"; then echo "SEALED NAME IN THE STAGE: nothing copied"; exit 1; fi
  python3 - "$D/data/coe/sigma_heldout_strict.txt" "$T/data/coe/transfer_track832.jsonl" <<'EOF'
import json, sys
strict = set(open(sys.argv[1]).read().split())
names = [json.loads(l)["name"] for l in open(sys.argv[2])]
assert len(names) == 106 and set(names) == strict, "the staged transfer file is not exactly the 106 strict rows"
EOF
  if grep -E '(^|/)(\.env|.*credential.*|.*secret.*|id_[a-z0-9]+|.*\.pem)$' "$MAN"; then echo "A CREDENTIAL-LIKE FILE IN THE STAGE: nothing copied"; exit 1; fi
  echo "staged $(wc -l < "$MAN") files, $(du -shL "$ST" | cut -f1), for corpus $c on $h ($MAN)"
}

case $cmd in
  setup)
    "${SSH[@]}" "$h" 'mkdir -p /workspace/env && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
    "${RSYNC[@]}" "$W/lh_pod_setup.sh" "$h:/workspace/lh_pod_setup.sh"
    "${RSYNC[@]}" "$ENVD/pyproject.toml" "$ENVD/uv.lock" "$h:/workspace/env/"
    "${SSH[@]}" "$h" 'cd /workspace/env && md5sum pyproject.toml uv.lock' | sed 's/^/pod: /'
    (cd "$ENVD" && md5sum pyproject.toml uv.lock) | sed 's/^/lab: /'
    echo "setup start $(utc) (8 to 15 min; log /workspace/setup.log on the pod)"
    timeout 3900 "${SSH[@]}" "$h" 'bash /workspace/lh_pod_setup.sh; tail -2 /workspace/setup.log'
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
    grep -E "lh/(data|sets|bases)/|kd/pools/" "$MAN.lab.md5" | sed -E 's/^(.{8})[0-9a-f]+/\1…/' || true ;;
  check-env)
    base=${1:-}
    [[ -z $base || $base =~ ^[a-z0-9_]+_s[0-9]+$ ]] || { echo "base must look like c24_s0"; exit 2; }
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
cd /workspace/exp/linked_habits_20261127
LH=/scratch/cowifornia/nd-rl/linked_habits $PY -c "
import gzip, json, glob
import lh_core as C, lh_profile as P, lh_twin as TW, lh_rlvar, lh_rm, lh_dpo, lh_sft
tab = C.profile_tables()
f = sorted(glob.glob(str(C.LH / 'data' / 'c*.jsonl.gz')))[0]
recs = [json.loads(x) for _, x in zip(range(200), gzip.open(f, 'rt'))]
bad = [r['key'] for r in recs if not P.profile(r['prompt'], r['proof'], tab)['ok']]
assert not bad, bad[:3]
tw = [TW.twin(r['prompt'], r['proof'], 'reit', tab) for r in recs]
n = sum(t is not None for t in tw)
print('detector reads 200 corpus records; reit twins:', n, '; Lean on 5 twins:', TW.lean_check_twins([(r['prompt'], t) for r, t in zip(recs, tw) if t][:5], 4))
"
wc -l /workspace/lh/sets/*.jsonl | tail -3
$PY -c "import sys; sys.path.insert(0, '/workspace/exp/author_tags_20261110'); import tag_core as T; T.install_thinking(); from code.thinking import grpo, canary; print('grpo import ok')"
if [[ -n $base ]]; then
  rm -rf /workspace/check/m /workspace/check/r
  ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=8 CUDA_VISIBLE_DEVICES=0 timeout 600 $PY lh_measure.py run \
    --ckpt /workspace/lh/bases/stage1_$base.pt --ckpt_id check --tags A,B,none --sets marker_plain --limit 4 --k 2 \
    --device cuda --out_dir /workspace/check/m --raw_dir /workspace/check/r > /workspace/check/measure.log 2>&1
  grep -E "^check |pass@8" /workspace/check/measure.log
fi
echo PODCHECK_OK
EOF
    ;;
  rl2)  # PREREG Amendment 10: c24's GRPO lines with RL seed 1 from the first base (lh_rl2.sh), after <c>.BASE2_DONE
    c=${1:-}; corpus_arg "$c"
    if [[ ${2:-} == --dry-run ]]; then
      "${SSH[@]}" "$h" "cd $PODW && LH=/scratch/cowifornia/nd-rl/linked_habits PY=$PY bash $PODW/lh_rl2.sh $c --dry-run"
      exit $?
    fi
    total=$(( ${TOTAL_H:-9} * 3600 ))
    "${SSH[@]}" "$h" "mkdir -p /workspace/lh/pod && cd $PODW && [[ ! -d /workspace/lh/pod/$c.driver.lock ]] || { echo 'a driver for $c holds its lock'; exit 3; }; \
      LH=/scratch/cowifornia/nd-rl/linked_habits PY=$PY nohup timeout --kill-after=120 $total bash $PODW/lh_rl2.sh $c \
      >> /workspace/lh/pod/$c.rl2.log 2>&1 < /dev/null & echo \"rl2 $c PID \$! (timeout $total s), log /workspace/lh/pod/$c.rl2.log\""
    ;;
  start)
    c=${1:-}; corpus_arg "$c"
    total=$(( ${TOTAL_H:-48} * 3600 ))
    nov=$( [[ $c == "$NOV14_POD" ]] && echo 1 || echo 0 )
    "${SSH[@]}" "$h" "mkdir -p /workspace/lh/pod && cd $PODW && [[ ! -d /workspace/lh/pod/$c.driver.lock ]] || { echo 'a driver for $c holds its lock'; exit 3; }; \
      LH=/scratch/cowifornia/nd-rl/linked_habits PY=$PY NOV14=$nov nohup timeout --kill-after=120 $total bash $PODW/lh_driver.sh $c \
      >> /workspace/lh/pod/$c.driver.log 2>&1 < /dev/null & echo \"driver $c PID \$! (timeout $total s, Nov 14 bases: $nov), log /workspace/lh/pod/$c.driver.log\""
    ;;
  job)
    [[ $# -ge 1 ]] || { echo "job <alias> <pretrain|measure|prep2|stage2|rl|status> <launcher args>"; exit 2; }
    args=$(printf '%q ' "$@")
    "${SSH[@]}" "$h" "cd $PODW && ND_GUARD=off LH=/scratch/cowifornia/nd-rl/linked_habits $PY lh_launch.py $args --gpu 0" ;;
  go|nogo)
    c=${1:-}; corpus_arg "$c"
    f=$( [[ $cmd == go ]] && echo GO || echo NOGO )
    "${SSH[@]}" "$h" "[[ -e /workspace/lh/pod/$c.GATE_READY ]] || { echo 'no $c.GATE_READY on the pod: the base is not measured yet'; exit 3; }; date -u +%FT%TZ > /workspace/lh/pod/$c.$f && echo 'wrote /workspace/lh/pod/$c.$f'" ;;
  check)
    "${SSH[@]}" "$h" bash -s <<'EOF'
PY=/workspace/venv/bin/python
echo "== $(date -u +%FT%TZ) $(hostname)"
for f in /workspace/lh/pod/*.heartbeat.json; do [[ -e $f ]] && { echo "== heartbeat $f"; cat "$f"; echo; }; done
echo "== markers: $(cd /workspace/lh/pod 2>/dev/null && ls -1 | grep -E '\.(GATE_READY|GO|NOGO|DONE|BASE2_DONE|RL2_DONE|FAILED|NOT_RUN|STOPPED)$' | tr '\n' ' ')"
for f in /workspace/lh/pod/*.driver.log; do [[ -e $f ]] && { echo "== $f"; tail -8 "$f"; }; done
cd /workspace/exp/linked_habits_20261127 && LH=/scratch/cowifornia/nd-rl/linked_habits $PY lh_launch.py status 2>&1 | tail -25
L=$(ls -t /workspace/lh/logs/*.log 2>/dev/null | grep -v watchdog | head -1)
[[ -n $L ]] && { echo "== newest job log $L"; tail -6 "$L"; }
for f in $(ls -t /workspace/lh/logs/*.tasks.out 2>/dev/null | head -2); do echo "== $f"; tail -3 "$f"; done
echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "== disk"; df -h /workspace | tail -1
echo "== python processes"; ps -eo pid,etime,args | grep -E "lh_|authors_|tag_|grpo|state_train|rlvar" | grep -v grep | cut -c1-160
EOF
    ;;
  fetch)
    light=${1:-}
    [[ -z $light || $light == --light || $light == --keep ]] || { echo "fetch: unknown mode $light (none, --light or --keep)"; exit 2; }
    mode=${light#--}
    mode=${mode:-full}
    F=$LH/pod_stage/$h.fetch
    mkdir -p "$(dirname "$F")"
    rm -f -- "$F".*  # never a stale listing from an earlier fetch
    fail() { echo "FETCH FAILED $h mode=$mode: $*"; exit 1; }
    listing() {  # listing <out> <remote command> [stdin file]: the command's lines on the pod, then a sentinel the pod
      # appends (LISTING_END <count>); a non-zero ssh, a missing sentinel or another line count is a failure (audit W1)
      local out=$1 n got
      "${SSH[@]}" "$h" "cd /workspace/lh && { $2 ; } | awk '{print} END {print \"LISTING_END \" NR}'" < "${3:-/dev/null}" > "$out.raw" \
        || fail "ssh exited $? while listing $(basename "$out")"
      [[ $(tail -n 1 "$out.raw") =~ ^LISTING_END\ ([0-9]+)$ ]] || fail "the listing $(basename "$out") is cut off (no LISTING_END line)"
      n=${BASH_REMATCH[1]}
      head -n -1 "$out.raw" > "$out"
      got=$(wc -l < "$out")
      (( got == n )) || fail "the listing $(basename "$out") has $got lines; the pod sent $n"
    }
    # every file but the checkpoints; then the checkpoints (size, path) and the fully measured snapshot folders
    listing "$F.list" "{ find bases measure raw runs logs pod pairs arms -type f ! -name '*.pt' ! -name '*.tmp' 2>/dev/null; \
      find data -maxdepth 1 -type f \( -name 'anchor_*' -o -name 'klstates_*' \) 2>/dev/null; } | sort"
    [[ -s $F.list ]] || fail "the pod lists no file under /workspace/lh"
    listing "$F.pt" "find bases measure raw runs logs pod pairs arms -type f -name '*.pt' -printf '%s %p\n' 2>/dev/null; \
      find arms -mindepth 3 -maxdepth 3 -type d -path 'arms/*/measure/*' -printf 'DIR %p\n' 2>/dev/null"
    # which checkpoints: full = all but *_alarm.pt; --light = none; --keep = the bases, each run's final.pt and rm.pt,
    # the fully measured snapshots (arms/<run>/measure/step<k> exists), every GRPO snapshot (_it<N>_snap.pt). Full and --keep
    # refuse when the pod's own records (a base's tag.json, a run's done.json) name a checkpoint its listing lacks.
    python3 - "$mode" "$F.list" "$F.pt" "$F.keep" "$F.left" "$F.need" <<'CHOOSEEOF' || fail "the checkpoint choice refused (above)"
import re, sys
mode, list_f, pt_f, keep_f, left_f, need_f = sys.argv[1:7]
listed = [ln.rstrip("\n") for ln in open(list_f) if ln.strip()]
pts, dirs = {}, set()
for ln in open(pt_f):
    ln = ln.rstrip("\n")
    if ln.startswith("DIR "):
        dirs.add(ln[4:])
        continue
    size, _, path = ln.partition(" ")
    if not (size.isdigit() and path.endswith(".pt")):
        sys.exit(f"FETCH FAILED: an unreadable line in the checkpoint listing: {ln[:120]!r}")
    pts[path] = int(size)
keep, lines = {}, {}
for p in pts:
    if mode == "light" or p.endswith("_alarm.pt"):
        continue
    if mode == "full":
        keep[p] = "checkpoint"
    elif re.fullmatch(r"bases/[^/]+\.pt", p):
        keep[p] = "base"
    elif re.fullmatch(r"arms/[^/]+/final\.pt", p):
        keep[p] = "run final"
    elif re.fullmatch(r"arms/rm_[^/]+/rm\.pt", p):
        keep[p] = "reward model"
    elif (m := re.fullmatch(r"arms/([^/]+)/snap_step(\d+)\.pt", p)) and f"arms/{m[1]}/measure/step{int(m[2])}" in dirs:
        keep[p] = "measured snapshot"
    elif m := re.fullmatch(r"runs/([^/]+)/seg(\d+)/ckpts/\1_it(\d+)(_snap)?\.pt", p):
        if m[4]:  # every GRPO snapshot (audit W2: the primary read is at the one nearest +30)
            keep[p] = "GRPO snapshot"
            lines.setdefault(m[1], [])
        else:
            lines.setdefault(m[1], []).append((int(m[3]), int(m[2]), p))
for line, plain in lines.items():
    if plain and not any(v == "GRPO snapshot" and k.startswith(f"runs/{line}/") for k, v in keep.items()):
        keep[max(plain)[2]] = "GRPO last checkpoint (the line has no snapshot)"
need, errs = [], []
if mode != "light":
    need = sorted(p for p in pts if re.fullmatch(r"bases/[^/]+\.pt", p) or re.fullmatch(r"arms/[^/]+/final\.pt", p))
    for p in listed:
        if m := re.fullmatch(r"bases/(.+)\.tag\.json", p):
            if f"bases/{m[1]}.pt" not in pts:
                errs.append(f"the pod has {p} but its checkpoint listing has no bases/{m[1]}.pt")
        elif (m := re.fullmatch(r"arms/([^/]+)/done\.json", p)) and not m[1].startswith("rm_"):
            if f"arms/{m[1]}/final.pt" not in pts:
                errs.append(f"the pod has {p} but its checkpoint listing has no arms/{m[1]}/final.pt")
    errs += [f"a required checkpoint was not chosen: {p}" for p in need if p not in keep]
if errs:
    for e in errs[:20]:
        print(f"FETCH FAILED: {e}")
    sys.exit(1)
left = sorted(p for p in pts if p not in keep)
for f, rows in ((keep_f, sorted(keep)), (left_f, [f"{pts[p]} {p}" for p in left]), (need_f, need)):
    with open(f, "w") as fh:
        fh.writelines(f"{r}\n" for r in rows)
kinds = {k: sum(1 for v in keep.values() if v == k) for k in sorted(set(keep.values()))}
print(f"checkpoints fetched: {len(keep)} ({sum(pts[p] for p in keep) / 1e9:.2f} GB; {kinds}); LEFT ON THE POD (lost when "
      f"it is deleted, reproducible from the seeds): {len(left)} checkpoints, {sum(pts[p] for p in left) / 1e9:.2f} GB, listed in {left_f}")
CHOOSEEOF
    cat "$F.keep" >> "$F.list"
    sort -u -o "$F.list" "$F.list"
    listing "$F.pod.md5" "xargs -d '\n' md5sum" "$F.list"
    gone=$(cut -c35- "$F.pod.md5" | sort | comm -23 "$F.list" -) || true
    [[ -z $gone ]] || fail "listed files vanished on the pod before their md5 (a job still running? fetch again after it): $(printf '%s ' $gone | cut -c1-400)"
    # a result already on serrano must be the same file (never overwritten): bases, base measurements, pairs, anchor
    # files, checkpoints, and the rows of every measurement and probe; records a running task rewrites (lh_offline.json,
    # metrics.jsonl, full_*.json, logs) may change between two fetches
    conflict=0
    while read -r sum path; do
      if [[ ( $path =~ ^(bases|measure|pairs|data)/ || $path =~ ^arms/.*\.pt$ || $path =~ ^arms/.*/(measure|probe)/.*\.jsonl$ ) && -e $LH/$path ]]; then
        [[ $(md5sum < "$LH/$path" | cut -d' ' -f1) == "$sum" ]] || { echo "CONFLICT: $LH/$path differs from $h's"; conflict=1; }
      fi
    done < "$F.pod.md5"
    [[ $conflict == 0 ]] || fail "nothing fetched (resolve the conflicts by hand)"
    timeout 7200 "${RSYNC[@]}" --files-from="$F.list" "$h:/workspace/lh/" "$LH/" || fail "rsync exited $?"
    (cd "$LH" && xargs -d '\n' md5sum < "$F.list") > "$F.lab.md5" 2> "$F.lab.err" \
      || fail "serrano's md5 of the fetched files failed: $(head -c 300 "$F.lab.err")"
    # a second listing on the pod: a file whose md5 changed between the two listings is still being written (a running
    # job's log, metrics, status), and heartbeats always change; those are named, not compared (audit R11). Every other
    # listed file must be equal on both sides; both listings must cover every listed file (audit W1); the required
    # checkpoints (bases, runs' final.pt) must be fetched and equal. FETCH_OK is printed only then, as the last line.
    listing "$F.pod2.md5" "xargs -d '\n' md5sum" "$F.list"
    python3 - "$h" "$mode" "$F.list" "$F.pod.md5" "$F.pod2.md5" "$F.lab.md5" "$F.keep" "$F.left" "$F.need" <<'VERIFYEOF' || exit 1
import sys
import time
alias, mode, list_f, p1_f, p2_f, lab_f, keep_f, left_f, need_f = sys.argv[1:10]
def read(f):
    return {ln[34:].rstrip("\n"): ln[:32] for ln in open(f) if len(ln) > 34}
def rows(f):
    return [ln.rstrip("\n") for ln in open(f) if ln.strip()]
paths = rows(list_f)
p1, p2, lab = read(p1_f), read(p2_f), read(lab_f)
errs = []
for name, d in (("the pod's first", p1), ("the pod's second", p2), ("serrano's", lab)):
    miss = [k for k in paths if k not in d]
    if miss:
        errs.append(f"{name} md5 listing lacks {len(miss)} of {len(paths)} listed files ({', '.join(miss[:3])})")
moving = sorted(k for k in paths if k in p2 and (p1.get(k) != p2[k] or (k.startswith("pod/") and k.endswith(".heartbeat.json"))))
compared = [k for k in paths if k in p2 and k not in moving]
if len(compared) + len(moving) != len(paths):
    errs.append(f"compared {len(compared)} + still being written {len(moving)} is not the {len(paths)} listed files")
bad = sorted(k for k in compared if lab.get(k) != p2[k])
for k in moving[:20]:
    print(f"still being written on the pod (not compared): {k}")
for k in bad[:20]:
    print(f"MD5 DIFFERS: {k}")
print(f"md5: {len(compared) - len(bad)} stable files equal, {len(bad)} differ, {len(moving)} still being written, of {len(paths)} listed")
if bad:
    errs.append(f"{len(bad)} stable files differ between the pod and serrano: fetch again; if they persist, stop and look")
listed = set(paths)
for k in rows(need_f):
    if k not in listed:
        errs.append(f"a required checkpoint was not fetched: {k}")
    elif k in moving or k in bad:
        errs.append(f"a required checkpoint is not equal on both sides: {k}")
left = [r.split(" ", 1) for r in rows(left_f)]
if errs:
    for e in errs:
        print(f"FETCH FAILED {alias} mode={mode}: {e}")
    sys.exit(1)
print(f"md5: every stable fetched file equal on {alias} and serrano ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})")
print(f"FETCH_OK {alias} mode={mode} files={len(paths)} ckpt={len(rows(keep_f))} left_on_pod={len(left)} "
      f"({sum(int(s) for s, _ in left) / 1e9:.2f} GB)")
VERIFYEOF
    ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
