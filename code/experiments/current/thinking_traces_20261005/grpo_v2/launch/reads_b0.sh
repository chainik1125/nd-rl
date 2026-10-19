#!/usr/bin/env bash
# reads_b0.sh (2026-10-18, agent:claude for Leon): reads_b0_l3test.sh for any finished mini20 line's checkpoints:
# line.sh's l3test:8 MINI_READS read at budget 0 (no thinking), reads_b0_l3test.sh's command, environment and guards.
#   bash reads_b0.sh GPU SPEC...             the SPECs' reads, one after another on GPU
#   bash reads_b0.sh GPU -f FILE [SPEC...]   ... the SPECs in FILE too (whitespace-separated; '#' starts a comment)
#   ECHO=1 bash reads_b0.sh GPU ...          each read's status and expanded command only (no GPU check, no lock,
#                                            nothing run or written)
# SPEC, for a mini20 arm (grpo_think_vs_mini20[_kKNOB], line.sh's knob list; refused: _tpanel, _szpanel, _ckreads,
# the 10-iteration mini arms and every other arm):
#   ARM:SEED           L3-mini's last checkpoint as line.sh's ck L3 derives it, ${ARM}_sSEED_L3/ckpts/${ARM}_sSEED_L3_it20.pt
#                      (a mini20 stage: ITERS = MINI20_ITERS = 20, IT3 = ITERS; no mini knob changes IT3), read
#                      under ARM: out l3test_ARM_sSEED_plain_k8_B0
#   ARM:SEED:STAGE:IT  STAGE's (L2, L3) rolling checkpoint at IT (--save_rolling 2: it0, it2, ... it18),
#                      ${ARM}_sSEED_STAGE/ckpts/${ARM}_sSEED_STAGE_itIT_rolling.pt, read under line.sh's MINI_ROLL
#                      name ARM_STAGEitIT: out l3test_ARM_STAGEitIT_sSEED_plain_k8_B0. A fork (kl1 g64 advpos advneg)
#                      has no L2-mini of its own: its L2 specs are refused (the reference's L2 is grpo_think_vs_mini20).
# A SPEC is read only when its line is done, 'grpo_v2 line done (ARM seed SEED)' in runs/logs/v2_ARM_sSEED.log
# (save_ckpt is not atomic; a done line's checkpoints are never rewritten: line.sh's grpo_stage returns at once when
# a stage's last checkpoint exists), and its checkpoint is there (STAGE:IT: the stage's last, _it20.pt, too).
# A malformed or refused SPEC stops the script before any read; a line not done or a missing checkpoint skips that
# SPEC (said). reads_b0_l3test.sh's own mini20 reads (grpo_think_vs_mini20:0 ... :3, the same out dirs) are skipped:
# it may be running, and its skip-if-exists check is 5 s before its launch.
# The read is reads_b0_l3test.sh's: line.sh's read1 l3test $(rsets l3test) 8 plain 96 NAME CKPT at --budgets 0, the
# same python, worktree ($WT on $BR, cwd: code.thinking resolves there) and environment (lib.sh's exports,
# LEAN_GATE_WORKERS=8); only the checkpoint, the out dir and its eval log differ.
# Guards (reads_b0_l3test.sh's, the GPU gate stricter after an audit, 2026-10-18): a read is skipped when its output
# dir or eval log exists (never removed or overwritten; the log is created exclusively); GPU is one of ours, 0-2 (3 is
# not a target, 4 is another agent's); the run stops when GPU is busy, or all 3 of GPUs 0-2 are (the PSU limit: at most
# 3 GPU jobs of ours), checked before each read. Busy: >= 500 MiB, a compute process, or a live pull_queue.sh, line.sh,
# reads_b0_l3test.sh or roll_reads_mini20.sh process with that GPU as its first argument (pgrep -f; this script and its
# ancestors excluded), since a line.sh between two reads holds no GPU memory. Timeout 1800 s a read; one line per read
# in $LOG; a lock (reads_b0.lock: one reads_b0.sh at a time). Never reads textbook72 / tb72.
set -u
ECHO=${ECHO:-0}
U="usage: [ECHO=1] bash reads_b0.sh GPU [-f FILE] SPEC...   (SPEC: ARM:SEED or ARM:SEED:L2|L3:IT)"
g=${1:-}; [[ $g =~ ^[0-9]+$ ]] || { echo "$U"; exit 2; }; shift
case $g in 0|1|2) ;; *) echo "GPU $g: ours are GPUs 0-2 (3 is not a target, 4 is another agent's): refusing"; exit 2 ;; esac
R=/scratch/cowifornia/nd-rl/think/runs; L=$R/logs; EV=$R/eval/v2; C=$R/grpo_v2
WT=/scratch/cowifornia/nd-rl/think/nd-rl-v2; BR=leon/grpo-v2  # line.sh's
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python  # line.sh's
LOG=$L/reads_b0.log; TO=1800; IT=20  # IT: a mini20 stage's last iteration (line.sh's MINI20_ITERS)
SET=$R/pools/curriculum/L3/test_1k.jsonl  # rsets l3test
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp  # lib.sh's exports
export LEAN_GATE_WORKERS=8  # the PSU limit's (serrano)
SPECS=()
while [ $# -gt 0 ]; do
  if [ "$1" = -f ]; then
    [ -n "${2:-}" ] && [ -f "$2" ] || { echo "no spec file '${2:-}': refusing"; exit 2; }
    while IFS= read -r x || [ -n "$x" ]; do read -ra ys <<< "${x%%#*}"; SPECS+=("${ys[@]}"); done < "$2"; shift 2
  else SPECS+=("$1"); shift; fi
done
[ ${#SPECS[@]} -gt 0 ] || { echo "$U"; exit 2; }
bad=0
for x in "${SPECS[@]}"; do  # the SPEC's form and arm, before anything runs
  [[ $x =~ ^[a-z0-9_]+:[0-9]+(:L[23]:[0-9]+)?$ ]] || { echo "refusing $x: not ARM:SEED or ARM:SEED:L2|L3:IT"; bad=1; continue; }
  a=${x%%:*}
  case $a in *_tpanel*|*_szpanel*|*_ckreads*) echo "refusing $x: $a is a read-only job (no checkpoints of its own)"; bad=1; continue ;; esac
  k=${a#grpo_think_vs_mini20}  # line.sh's mini knob list (refused there: lp00, lpm00)
  case $a in grpo_think_vs_mini20|grpo_think_vs_mini20_k*) ;; *) k=-- ;; esac
  case $k in ""|_krr|_kt07|_kt085|_kt115|_kt13|_klp[0-9][0-9]|_klpm[0-9][0-9]|_kkl1|_kg64|_kg64b|_kpool250|_kpool500|_ksize|_klossact|_klossthink|_kadvpos|_kadvneg) ;;
    *) echo "refusing $x: $a is not a mini20 arm (grpo_think_vs_mini20[_kKNOB])"; bad=1; continue ;; esac
  case $k:$x in _kkl1:*:L2:*|_kg64:*:L2:*|_kadvpos:*:L2:*|_kadvneg:*:L2:*)
    echo "refusing $x: ${k#_k} is an L3-mini fork, with no L2-mini of its own (grpo_think_vs_mini20's L2)"; bad=1 ;; esac
done
[ $bad = 0 ] || { echo "refusing: nothing run"; exit 2; }
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; [ "$ECHO" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
PAT='pull_queue|line|reads_b0_l3test|roll_reads_mini20'  # job scripts that hold the GPU of their first argument
ANC=" "; p=$$  # this script and its ancestors (a 'bash -c' chain naming it): never counted as a job
while [ -n "$p" ] && [ "$p" -gt 1 ]; do ANC="$ANC$p "; p=$(ps -o ppid= -p $p 2> /dev/null | tr -d ' '); done
busy() {  # GPU -> 0, with WHY, when it is in use: >= 500 MiB, a compute process, or a live job script for it
  local i=$1 m a p x
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2> /dev/null) \
    || { WHY="GPU $i: nvidia-smi failed (no such GPU?)"; return 0; }
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i 2> /dev/null)
  p=""; for x in $(pgrep -f "(^|[ /])($PAT)\.sh $i([^0-9]|\$)"); do case $ANC in *" $x "*) ;; *) p="$p $x" ;; esac; done
  [ "$m" -lt 500 ] 2> /dev/null || { WHY="GPU $i uses $m MiB (>= 500)"; return 0; }
  [ -z "$a" ] || { WHY="GPU $i has compute processes ($(echo $a | tr '\n' ' '))"; return 0; }
  [ -z "$p" ] || { WHY="GPU $i has job script processes (pids$p)"; return 0; }
  return 1; }
gate() {  # 0 when GPU $g is free and at most 2 of GPUs 0-2 are busy (at most 3 GPU jobs of ours); else says why
  local i b=0 w=""
  busy $g && { say "$WHY: refusing"; return 1; }
  for i in 0 1 2; do busy $i && { b=$((b + 1)); w="$w; $WHY"; }; done
  [ $b -lt 3 ] || { say "$b of GPUs 0-2 busy (${w#; }): refusing"; return 1; }
  return 0; }
case $SET in *textbook*|*tb72*) echo "refusing: $SET is textbook72"; exit 1 ;; esac
cd $WT || exit 1
[ "$(git -C $WT rev-parse --abbrev-ref HEAD)" = $BR ] || { echo "$WT is not on $BR: refusing"; exit 1; }
"$PY" -c "import code.thinking, sys; sys.exit(0 if code.thinking.__file__.startswith('$WT/') else 1)" \
  || { echo "code.thinking does not resolve to $WT: refusing"; exit 1; }
[ -f $SET ] || { echo "no $SET: refusing"; exit 1; }
if [ "$ECHO" != 1 ]; then
  exec 9> $L/reads_b0.lock; flock -n 9 || { echo "another reads_b0.sh is running: refusing"; exit 1; }
  gate || exit 1
  say "start on GPU $g (${#SPECS[@]} specs): $WT HEAD $(git -C $WT rev-parse HEAD)$([ -n "$(git -C $WT status --porcelain --untracked-files=no)" ] && echo ' (dirty)')"
fi
n=0; ran=0; SEEN=" "
for x in "${SPECS[@]}"; do
  IFS=: read -r a s st i <<< "$x"; n=$((n + 1))
  if [ -z "$st" ]; then  # ARM:SEED: line.sh's ck L3 (a mini20 arm: IT3 = 20)
    cp=$C/${a}_s${s}_L3/ckpts/${a}_s${s}_L3_it$IT.pt; fin=$cp; nm=$a
  else  # ARM:SEED:STAGE:IT: line.sh's MINI_ROLL checkpoint and name
    cp=$C/${a}_s${s}_$st/ckpts/${a}_s${s}_${st}_it${i}_rolling.pt; fin=$C/${a}_s${s}_$st/ckpts/${a}_s${s}_${st}_it$IT.pt; nm=${a}_${st}it$i
  fi
  o=$EV/l3test_${nm}_s${s}_plain_k8_B0; el=$L/v2_eval_$(basename $o).log; jl=$L/v2_${a}_s$s.log
  cmd=(env CUDA_VISIBLE_DEVICES=$g timeout $TO "$PY" -m code.thinking.eval --ckpt $cp --sets $SET --out $o --k 8
    --budgets 0 --arms plain --batch 1024 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10)
  why=()  # each reason this read is not run (the first one is said)
  grep -qE "^[0-9]{2}:[0-9]{2}Z grpo_v2 line done \(${a} seed ${s}\)\$" $jl 2> /dev/null \
    || why+=("line not done (no 'grpo_v2 line done ($a seed $s)' in $jl)")
  [ -f $fin ] || why+=("no stage last checkpoint $fin")
  [ -f $cp ] || why+=("no checkpoint $cp")
  case $cp in *_alarm.pt) why+=("$cp is an alarm checkpoint") ;; esac
  case $o in $EV/l3test_grpo_think_vs_mini20_s[0-3]_plain_k8_B0) why+=("reads_b0_l3test.sh's read") ;; esac
  case $SEEN in *" $o "*) why+=("a repeat of an earlier spec") ;; esac; SEEN="$SEEN$o "
  [ -e $o ] && why+=("$o exists")
  [ -e $el ] && why+=("$el exists")
  if [ "$ECHO" = 1 ]; then
    stt=" [would run]"; [ ${#why[@]} -gt 0 ] && stt=$(printf ' [SKIP: %s]' "${why[@]}")
    echo "# $n $x -> $(basename $o)$stt"; echo "cd $WT && $(printf '%q ' "${cmd[@]}")> $el 2>&1"; continue
  fi
  [ ${#why[@]} -gt 0 ] && { say "$x $(basename $o): ${why[0]}: skipped"; continue; }
  [ $ran -gt 0 ] && { sleep 5; gate || { say "stopped before $(basename $o)"; exit 1; }; }
  ( set -C; : > $el ) 2> /dev/null || { say "$x $(basename $o): $el exists: skipped"; continue; }  # exclusive
  t=$(date +%s); ran=$((ran + 1))
  "${cmd[@]}" > $el 2>&1; rc=$?
  say "read $(basename $o) ($x) exit $rc $(( $(date +%s) - t ))s: $(grep -E '^\[' $el | tr '\n' ' ' | cut -c1-200)"
done
[ "$ECHO" = 1 ] || say "done on GPU $g ($ran reads run)"
