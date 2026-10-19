#!/usr/bin/env bash
# roll_reads_mini20.sh (2026-10-18, agent:claude for Leon): line.sh's own MINI_ROLL reads of the finished mini20
# reference lines at seeds 0 and 1: MINI_PANEL (l3test:8 t1320:8 t2130:4 charles173:16, B64) of the rolling checkpoints
# L2-mini it8 and L3-mini it0, it8, it16, under grpo_think_vs_mini20_L2it8, _L3it0, _L3it8, _L3it16 (16 reads a seed).
#   bash roll_reads_mini20.sh GPU          seed 0, then seed 1, on GPU
#   ECHO=1 bash roll_reads_mini20.sh GPU   each seed's status and command only (no GPU check, no lock, nothing run
#                                          or written)
# Each seed is the rerun 'MINI_ROLL="L2:8 L3:0 L3:8 L3:16" bash line.sh GPU SEED grpo_think_vs_mini20' of a finished
# line, appended to the line's own log (runs/logs/v2_grpo_think_vs_mini20_sSEED.log) after a '==' header, as
# pull_queue.sh runs a job. On a finished line (line.sh md5 c338f00f...) that rerun trains nothing and moves nothing:
# grpo_stage returns at once when a stage's last checkpoint exists (before its crashed-dir move, waitfree and the
# watchdog); the line's 7 reads are skipped (their rows_plain_B64.jsonl exist); only the 16 MINI_ROLL reads run
# (line.sh's read1: waitfree on GPU, timeout 9000 s a read). The rerun also repeats both preflights (CPU, about 10 s;
# preflight_grpo_think_vs_mini20_sSEED.log and _extra.log are rewritten: their md5 before and after are logged) and
# writes HEAD to grpo_think_vs_mini20_sSEED.reads_commit.txt (base_commit.txt is kept).
# Guards: a seed runs only when its line is done ('grpo_v2 line done (grpo_think_vs_mini20 seed SEED)' in its log) and
# both stages' last checkpoints (_it20.pt) exist, else line.sh would train: that seed is refused; line.sh must be the
# version checked (md5 c338f00f...); GPU is one of ours, 0-2 (3 is not a target, 4 is another agent's); reads_b0.sh's
# GPU gate before each seed (GPU not busy, and not all 3 of GPUs 0-2 busy; busy: >= 500 MiB, a compute process, or a
# live pull_queue.sh, line.sh, reads_b0_l3test.sh or reads_b0.sh process with that GPU as its first argument, pgrep -f,
# this script and its ancestors excluded); the environment that changes line.sh (SMOKE, SMOKE_WT, KNOB_FULL, POOL_L2,
# POOL_L3, REJ_PENALTY, XFRAC) unset; one roll_reads_mini20.sh at a time (roll_reads_mini20.lock); a start and an end
# line per seed (UTC) in runs/logs/roll_reads_mini20.log.
# Deadline: DEADLINE_UTC (env; epoch seconds or a 'date -u -d' string such as '2026-10-18 15:00'; default: the
# script's start + 4 h) is checked before each seed starts: past it, the script stops (said) before that seed. It never
# kills a running line.sh: a seed started before the deadline runs to its end (about 1.5 h). No outer timeout: killing
# line.sh would orphan read1's 'timeout 9000 python' (its own process group); each read stays bounded by that 9000 s,
# and no watchdog runs.
set -u
g=${1:-}; ECHO=${ECHO:-0}
[[ $g =~ ^[0-9]+$ ]] || { echo "usage: [ECHO=1] bash roll_reads_mini20.sh GPU"; exit 2; }
case $g in 0|1|2) ;; *) echo "GPU $g: ours are GPUs 0-2 (3 is not a target, 4 is another agent's): refusing"; exit 2 ;; esac
T0=$(date +%s); DL=${DEADLINE_UTC:-}  # the deadline for starting a seed, epoch seconds
if [ -z "$DL" ]; then DL=$((T0 + 4 * 3600))
elif ! [[ $DL =~ ^[0-9]+$ ]]; then
  DL=$(date -u -d "$DL" +%s 2> /dev/null) || { echo "DEADLINE_UTC='$DEADLINE_UTC': not epoch seconds or a date -d string: refusing"; exit 2; }
fi
DLS=$(date -u -d @$DL +%Y-%m-%dT%H:%M:%SZ)
R=/scratch/cowifornia/nd-rl/think/runs; L=$R/logs; C=$R/grpo_v2
LINE=$C/line.sh; MD5=c338f00f  # the line.sh read for this script (its rerun behaviour)
ARM=grpo_think_vs_mini20; SEEDS="0 1"; ROLL="L2:8 L3:0 L3:8 L3:16"; IT=20  # IT: a mini20 stage's last iteration
LOG=$L/roll_reads_mini20.log
export LEAN_GATE_WORKERS=8  # the PSU limit's (serrano)
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; [ "$ECHO" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
PAT='pull_queue|line|reads_b0_l3test|reads_b0'  # job scripts that hold the GPU of their first argument (reads_b0.sh's gate)
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
m=$(md5sum < $LINE | cut -c1-32)
[ "${m:0:8}" = $MD5 ] || { echo "$LINE has md5 $m, not the checked $MD5...: refusing"; exit 1; }
if [ "$ECHO" != 1 ]; then
  exec 9> $L/roll_reads_mini20.lock; flock -n 9 || { echo "another roll_reads_mini20.sh is running: refusing"; exit 1; }
  say "start on GPU $g: seeds $SEEDS, MINI_ROLL=$ROLL, no seed starts after $DLS"
else echo "# deadline for starting a seed: $DLS ($(( (DL - T0) / 60 )) min from now)"; fi
n=0
for s in $SEEDS; do
  jl=$L/v2_${ARM}_s$s.log; pf="$C/preflight_${ARM}_s$s.log $C/preflight_${ARM}_s${s}_extra.log"
  why=()  # each reason this seed must not run (line.sh would train, or the line is not finished)
  grep -qE "^[0-9]{2}:[0-9]{2}Z grpo_v2 line done \(${ARM} seed ${s}\)\$" $jl 2> /dev/null \
    || why+=("line not done (no 'grpo_v2 line done ($ARM seed $s)' in $jl)")
  for st in L2 L3; do f=$C/${ARM}_s${s}_$st/ckpts/${ARM}_s${s}_${st}_it$IT.pt; [ -f $f ] || why+=("no $f (line.sh would train)"); done
  todo=""; for c in $ROLL; do  # the reads still to do (line.sh skips a read whose rows_plain_B64.jsonl exists)
    st=${c%%:*}; i=${c#*:}; nm=${ARM}_${st}it$i
    [ -f $C/${ARM}_s${s}_$st/ckpts/${ARM}_s${s}_${st}_it${i}_rolling.pt ] || why+=("no $st it$i rolling checkpoint")
    for r in l3test:8 t1320:8 t2130:4 charles173:16; do  # line.sh's MINI_PANEL
      [ -f $R/eval/v2/${r%:*}_${nm}_s${s}_plain_k${r#*:}/rows_plain_B64.jsonl ] || todo="$todo ${r%:*}_${nm}"; done
  done
  cmd=(env -u SMOKE -u SMOKE_WT -u KNOB_FULL -u POOL_L2 -u POOL_L3 -u REJ_PENALTY -u XFRAC MINI_ROLL="$ROLL" bash $LINE $g $s $ARM)
  if [ "$ECHO" = 1 ]; then
    stt=" [would run: $(echo $todo | wc -w) of 16 reads to do]"; [ ${#why[@]} -gt 0 ] && stt=$(printf ' [REFUSED: %s]' "${why[@]}")
    echo "# seed $s$stt"; echo "echo \"== \$(date -u +%H:%MZ) roll_reads_mini20.sh GPU $g MINI_ROLL=$ROLL\" >> $jl"
    echo "$(printf '%q ' "${cmd[@]}")>> $jl 2>&1"; continue
  fi
  [ ${#why[@]} -gt 0 ] && { say "$ARM seed $s: ${why[0]}: refused"; continue; }
  [ -z "$todo" ] && { say "$ARM seed $s: its 16 MINI_ROLL reads exist: skipped"; continue; }
  [ $n -gt 0 ] && sleep 5
  [ $(date +%s) -lt $DL ] || { say "past the deadline $DLS: stopped before $ARM seed $s"; exit 1; }
  gate || { say "stopped before $ARM seed $s"; exit 1; }
  p0=$(md5sum $pf 2> /dev/null | cut -c1-8 | tr '\n' ' ')
  say "$ARM seed $s: start on GPU $g, MINI_ROLL=$ROLL ($(echo $todo | wc -w) reads to do; deadline $DLS)"
  echo "== $(date -u +%H:%MZ) roll_reads_mini20.sh GPU $g MINI_ROLL=$ROLL" >> $jl
  t=$(date +%s); n=$((n + 1))
  "${cmd[@]}" >> $jl 2>&1; rc=$?
  p1=$(md5sum $pf 2> /dev/null | cut -c1-8 | tr '\n' ' ')
  say "$ARM seed $s: end, exit $rc $(( $(date +%s) - t ))s (preflight logs md5 ${p0}-> $p1$([ "$p0" = "$p1" ] && echo '(unchanged)' || echo '(CHANGED)')): $(tail -1 $jl | cut -c1-160)"
done
[ "$ECHO" = 1 ] || say "done on GPU $g"
