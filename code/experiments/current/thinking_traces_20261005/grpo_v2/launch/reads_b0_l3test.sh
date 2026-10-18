#!/usr/bin/env bash
# reads_b0_l3test.sh (2026-10-18, agent:claude for Leon): line.sh's l3test:8 MINI_READS read at budget 0 (no thinking).
#   bash reads_b0_l3test.sh GPU          the 7 reads below, one after another on GPU
#   ECHO=1 bash reads_b0_l3test.sh GPU   print the 7 expanded commands only (no GPU check, nothing run or written)
# The read is line.sh's read1 l3test $(rsets l3test) 8 plain 96 NAME CKPT (T 0.8, max_rej 10, --seed 0, batch 1024,
# max_action 512), the same python, worktree ($WT on $BR, cwd: code.thinking resolves there) and environment (lib.sh's
# exports, LEAN_GATE_WORKERS=8 as the queue's), with two changes: --budgets 0 for line.sh's RB=64, and the output dir,
# line.sh's name plus _B0 (as its _T13 suffix), so the rows land in <dir>/rows_plain_B0.jsonl and no B64 read is touched.
# B0 for an effort (lean_thinke) checkpoint is the B0 of its GRPO budgets 0,64,256: sample.step_prompt prefills
# 'state <e0> <think> </think>' for both (grpo.py's rollouts and eval.py call the same think_generate); eval.py needs
# no other flag (effort is read from the tokenizer, --think auto is on for a think tokenizer, min_think min(0, B) = 0).
# Guards: a read is skipped when its output dir or eval log exists (never removed or overwritten); the run stops when
# GPU uses >= 500 MiB or has a compute process, or 3 or more GPUs use >= 500 MiB (the PSU limit), checked before each
# read; timeout 1800 s a read (line.sh's read1: 9000); one line per read in $LOG. Never reads textbook72 / tb72.
set -u
g=${1:-}; ECHO=${ECHO:-0}
[[ $g =~ ^[0-9]+$ ]] || { echo "usage: [ECHO=1] bash reads_b0_l3test.sh GPU"; exit 2; }
R=/scratch/cowifornia/nd-rl/think/runs; L=$R/logs; EV=$R/eval/v2; C=$R/grpo_v2
WT=/scratch/cowifornia/nd-rl/think/nd-rl-v2; BR=leon/grpo-v2  # line.sh's
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python  # line.sh's
LOG=$L/reads_b0_l3test.log; TO=1800
SET=$R/pools/curriculum/L3/test_1k.jsonl  # rsets l3test
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp  # lib.sh's exports
export LEAN_GATE_WORKERS=8  # the PSU limit's (serrano)
# NAME SEED CKPT: line.sh's read names (reads_of NAME: <set>_NAME_s<seed>_plain_k8; the base as s1g3_ckreads's 's1g3')
READS=(
  "grpo_think_vs_mini20 0 $C/grpo_think_vs_mini20_s0_L3/ckpts/grpo_think_vs_mini20_s0_L3_it20.pt"
  "grpo_think_vs_mini20 1 $C/grpo_think_vs_mini20_s1_L3/ckpts/grpo_think_vs_mini20_s1_L3_it20.pt"
  "grpo_think_vs_mini20 2 $C/grpo_think_vs_mini20_s2_L3/ckpts/grpo_think_vs_mini20_s2_L3_it20.pt"
  "grpo_think_vs_mini20 3 $C/grpo_think_vs_mini20_s3_L3/ckpts/grpo_think_vs_mini20_s3_L3_it20.pt"
  "s1g3 0 $R/stage1/S1g3_s0/model.pt"
  "grpo_think_vs_mb 0 $C/grpo_think_vs_mb_s0_L3/ckpts/grpo_think_vs_mb_s0_L3_it39.pt"
  "grpo_think_vs_mb_kkl0 0 $C/grpo_think_vs_mb_kkl0_s0_L3/ckpts/grpo_think_vs_mb_kkl0_s0_L3_it39.pt"
)
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; [ "$ECHO" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
gate() {  # 0 when GPU $g is free and fewer than 3 GPUs are busy; else says why
  local m b a
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2> /dev/null) \
    || { say "GPU $g: nvidia-smi failed (no such GPU?): refusing"; return 1; }
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2> /dev/null)
  b=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 != 4 && $2 + 0 >= 500' | wc -l)  # GPU 4: another agent's (Leon, 2026-10-18)
  [ "$m" -lt 500 ] || { say "GPU $g uses $m MiB (>= 500): refusing"; return 1; }
  [ -z "$a" ] || { say "GPU $g has compute processes ($(echo $a | tr '\n' ' ')): refusing"; return 1; }
  [ "$b" -lt 3 ] || { say "$b of GPUs 0-3 use >= 500 MiB (at most 3 GPU jobs of ours): refusing"; return 1; }
  return 0; }
case $SET in *textbook*|*tb72*) echo "refusing: $SET is textbook72"; exit 1 ;; esac
cd $WT || exit 1
[ "$(git -C $WT rev-parse --abbrev-ref HEAD)" = $BR ] || { echo "$WT is not on $BR: refusing"; exit 1; }
"$PY" -c "import code.thinking, sys; sys.exit(0 if code.thinking.__file__.startswith('$WT/') else 1)" \
  || { echo "code.thinking does not resolve to $WT: refusing"; exit 1; }
[ -f $SET ] || { echo "no $SET: refusing"; exit 1; }
if [ "$ECHO" != 1 ]; then
  exec 9> $L/reads_b0_l3test.lock; flock -n 9 || { echo "another reads_b0_l3test.sh is running: refusing"; exit 1; }
  gate || exit 1
  say "start on GPU $g: $WT HEAD $(git -C $WT rev-parse HEAD)$([ -n "$(git -C $WT status --porcelain --untracked-files=no)" ] && echo ' (dirty)')"
fi
n=0
for r in "${READS[@]}"; do
  read -r nm s cp <<< "$r"; n=$((n + 1))
  o=$EV/l3test_${nm}_s${s}_plain_k8_B0; el=$L/v2_eval_$(basename $o).log
  cmd=(env CUDA_VISIBLE_DEVICES=$g timeout $TO "$PY" -m code.thinking.eval --ckpt $cp --sets $SET --out $o --k 8
    --budgets 0 --arms plain --batch 1024 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10)
  if [ "$ECHO" = 1 ]; then
    st=""; [ -f $cp ] || st="$st [NO CKPT]"; [ -e $o ] && st="$st [OUT EXISTS: would skip]"; [ -e $el ] && st="$st [LOG EXISTS: would skip]"
    echo "# $n $(basename $o)$st"; echo "cd $WT && $(printf '%q ' "${cmd[@]}")> $el 2>&1"; continue
  fi
  case $cp in *_alarm.pt|*_rolling.pt) say "$(basename $o): $cp is not a final checkpoint: skipped"; continue ;; esac
  [ -f $cp ] || { say "$(basename $o): no checkpoint $cp: skipped"; continue; }
  [ -e $o ] && { say "$(basename $o): $o exists: skipped"; continue; }
  [ -e $el ] && { say "$(basename $o): $el exists: skipped"; continue; }
  [ $n -gt 1 ] && { sleep 5; gate || { say "stopped before $(basename $o)"; exit 1; }; }
  t=$(date +%s)
  "${cmd[@]}" > $el 2>&1; rc=$?
  say "read $(basename $o) exit $rc $(( $(date +%s) - t ))s: $(grep -E '^\[' $el | tr '\n' ' ' | cut -c1-200)"
done
[ "$ECHO" = 1 ] || say "done on GPU $g"
