#!/usr/bin/env bash
# One Lean-judged pass-rate measurement for hard12r with `code.thinking.eval` (plain arm, T 0.8,
# max_action 512, max_steps 96, max_rej 10, batch 1024: the settings of runs/midrun/gpu0_midrun.sh).
# usage: run_eval.sh GPU NAME CKPT SETS OUT K [BUDGETS=0] [SEED=1]
# GPU 0: waits (poll 30 s, at most WAIT_MIN = 240 min) until runs/gpu0.lock is absent AND GPU 0 has
#   < 500 MiB in use, then creates the lock atomically with its PID and removes it on exit.
# GPU 1: only once no python process of ours runs on it (the expert-iteration run has exited) and
#   it has < 500 MiB in use. Any other GPU is refused.
# Every job runs under `timeout 3600`; nvidia-smi is checked immediately before the launch.
# Skips a job whose OUT/summary.json already exists. Appends one line per job to OUT/../jobs.log.
set -u
G=$1 NAME=$2 CK=$3 SETS=$4 OUT=$5 K=$6 BUD=${7:-0} SEED=${8:-1}
R=/scratch/cowifornia/nd-rl/think/runs
LOCK=$R/gpu0.lock
LOGD=$R/pools/hard12r/work/logs
WAIT_MIN=${WAIT_MIN:-240}
mkdir -p "$LOGD" "$(dirname "$OUT")"
[ -f "$OUT/summary.json" ] && { echo "$NAME already done"; exit 0; }
case $G in 0 | 1) ;; *) echo "GPU $G not allowed"; exit 2 ;; esac
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G"; }
ours() {  # python processes of this user on GPU $G
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G")
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" |
    awk -F', ' '{print $2}' | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done |
    grep -c "^$(id -un)" || true
}
free_now() {
  if [ "$G" = 0 ]; then
    [ ! -e "$LOCK" ] && [ "$(mem)" -lt 500 ]
  else
    [ "$(ours)" -eq 0 ] && [ "$(mem)" -lt 500 ]
  fi
}
got=0
for _ in $(seq 1 $((WAIT_MIN * 2))); do
  if free_now; then
    if [ "$G" = 0 ]; then
      if (set -o noclobber; echo $$ > "$LOCK") 2>/dev/null; then got=1; break; fi
    else
      got=1; break
    fi
  fi
  sleep 30
done
[ $got = 1 ] || { echo "$(date -u +%FT%TZ) GPU $G not free after $WAIT_MIN min: $NAME not run"; exit 3; }
[ "$G" = 0 ] && trap 'rm -f "$LOCK"' EXIT
used=$(mem)
[ "$used" -lt 500 ] || { echo "$(date -u +%FT%TZ) GPU $G busy ($used MiB) at launch: $NAME not run"; exit 3; }
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp
t=$(date +%s)
echo "$(date -u +%FT%TZ) start $NAME GPU $G k $K B $BUD seed $SEED $SETS" >> "$LOGD/jobs.log"
CUDA_VISIBLE_DEVICES=$G timeout 3600 code/.venv/bin/python -m code.thinking.eval --ckpt "$CK" \
  --sets "$SETS" --out "$OUT" --k "$K" --budgets "$BUD" --arms plain --batch 1024 --seed "$SEED" \
  --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10 > "$LOGD/eval_$NAME.log" 2>&1
rc=$?
s=$(($(date +%s) - t))
echo "$(date -u +%FT%TZ) end $NAME GPU $G exit $rc ${s}s $(grep -E '^\[plain' "$LOGD/eval_$NAME.log" | tr '\n' ' ')" \
  >> "$LOGD/jobs.log"
exit $rc
