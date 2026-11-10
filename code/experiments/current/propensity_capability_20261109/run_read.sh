#!/usr/bin/env bash
# The propensity/capability read on ONE GPU of this host: checkpoints in priority order, then the CPU result rows.
#
#   run_read.sh GPU [CKPTS] [--hours H] [--budget SECS] [--k K] [--kf KF] [--batch B] [--dry-run]
#
#   GPU      nvidia-smi index (PCI order) on this host; the job sees it as cuda:0
#   CKPTS    comma list of ckpt ids and groups from sets/ckpts.json: `main` (the 19), `matched` (the 5 at matched
#            compute), `matched_early` (the 2 early 0.1 % reads), `all` (default: the 26 in priority order)
#   --hours  wall-clock cap for the GPU phase, enforced by `timeout` (default 3)
#   --budget GPU seconds over the whole read, summed in ROOT/gpu_ledger.jsonl across restarts (default 10800);
#            pc_run.py stops before the next arm once it is reached (exit 4)
#   --dry-run  print the plan (arms still to do per checkpoint, attempts, time estimate), the GPU's current
#            users and the exact commands; touches no GPU and writes nothing
#
# Gates, in pc_run.py: no other user or process on the GPU and under 500 MiB in use at launch, re-checked before every
# arm (exit 3); a mkdir lock ROOT/locks/<host>_gpu<N>.lock (exit 3 if held); the GPU-seconds ledger (exit 4). Arms are
# written atomically and skipped when present, so a rerun resumes. The rows phase (CPU, single process) runs after the
# GPU phase ends, for whatever finished. A watchdog (STALL / EXIT / OVERRUN) is attached when present.
# Progress: ROOT/logs/read_<UTC stamp>.log; PIDs in ROOT/logs/read.pid and watchdog.pid.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/scratch/cowifornia/nd-rl/propensity_capability
PY=${PY:-/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python}
WATCHDOG=/scratch/cowifornia/nd-rl/think/runs/watchdog.py

[ $# -ge 1 ] || { sed -n 2,20p "$0"; exit 2; }
GPU=$1; shift
case "$GPU" in ''|*[!0-9]*) echo "GPU must be an nvidia-smi index, got '$GPU'"; exit 2;; esac
CKPTS=all
if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then CKPTS=$1; shift; fi
HOURS=3; BUDGET=10800; K=32; KF=16; BATCH=1024; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --hours) HOURS=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --k) K=$2; shift 2;;
    --kf) KF=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done

export ND_OFFLINE=1 ND_REGISTRY=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export LEAN_GATE_WORKERS=8 LEAN_GATE_LOG=$ROOT/logs/lean_gate.jsonl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
cd "$HERE" || exit 1
for f in "$ROOT/sets/theorems.jsonl" "$ROOT/sets/ckpts.json" "$PY"; do
  [ -e "$f" ] || { echo "missing $f (run pc_sets.py first, on the host that holds the checkpoints)"; exit 1; }
done
ARGS=(--ckpts "$CKPTS" --root "$ROOT" --k "$K" --kf "$KF" --batch "$BATCH")
GPU_CMD=("$PY" pc_run.py gpu "${ARGS[@]}" --gpu "$GPU" --gpu_budget "$BUDGET")
ROWS_CMD=("$PY" pc_run.py rows --ckpts "$CKPTS" --root "$ROOT")
SECS=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 }')

if [ "$DRY" = 1 ]; then
  echo "== dry run on $(hostname -s), GPU $GPU; nothing is launched =="
  "$PY" pc_run.py plan "${ARGS[@]}" || exit 1
  USED=$("$PY" -c "import pc_run, pathlib; print(int(pc_run.ledger_sum(pathlib.Path('$ROOT/gpu_ledger.jsonl'))))")
  echo "GPU seconds already in the ledger: $USED of $BUDGET"
  echo "GPU users now (index: users):"; "$PY" "$HERE/../factor_read_20261106/gpu.py" "$(hostname -s)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.limit --format=csv,noheader -i "$GPU"
  ls -d "$ROOT"/locks/*.lock 2>/dev/null && echo "(a lock above is held)"
  echo "would run: timeout $SECS ${GPU_CMD[*]}"
  echo "then:      timeout 3600 ${ROWS_CMD[*]}"
  [ -f "$WATCHDOG" ] && echo "watchdog:  $PY $WATCHDOG pc_read:<PID>:$GPU:<LOG>:$HOURS"
  exit 0
fi

mkdir -p "$ROOT/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/read_$STAMP.log"
JOB="$ROOT/logs/read_$STAMP.sh"   # the exact commands, quoted, kept next to the log
{
  echo "#!/usr/bin/env bash"
  echo "cd $(printf %q "$HERE") || exit 1"
  printf 'timeout %q' "$SECS"; printf ' %q' "${GPU_CMD[@]}"; echo '; rc=$?'
  echo 'echo "$(date -u +%FT%TZ) gpu phase exit $rc (3 gate or lock, 4 budget, 124 timeout)"'
  printf 'timeout 3600'; printf ' %q' "${ROWS_CMD[@]}"; echo '; rc=$?'
  echo 'echo "$(date -u +%FT%TZ) rows phase exit $rc"'
} > "$JOB"
echo "$(date -u +%FT%TZ) launch on $(hostname -s) GPU $GPU, ckpts $CKPTS, k $K, kf $KF, batch $BATCH, cap ${HOURS} h, budget $BUDGET s; job $JOB" >> "$LOG"
nohup setsid bash "$JOB" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$ROOT/logs/read.pid"
if [ -f "$WATCHDOG" ]; then
  nohup setsid "$PY" "$WATCHDOG" "pc_read:$PID:$GPU:$LOG:$HOURS" >> "$ROOT/logs/watchdog.log" 2>&1 &
  echo "$!" > "$ROOT/logs/watchdog.pid"
fi
echo "read PID $PID (log $LOG)$( [ -f "$WATCHDOG" ] && echo "; watchdog PID $(cat "$ROOT/logs/watchdog.pid")")"
