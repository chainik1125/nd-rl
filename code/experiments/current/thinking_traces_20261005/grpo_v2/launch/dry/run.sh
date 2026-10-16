#!/bin/bash
# run.sh SCRIPT.dry MODE(normal|smoke) SEED ARM RESDIR [SETUP]: one dry run of line.sh in a fresh sandbox, in a clean
# environment, on GPU $DRY_GPU (default 3, as serrano's harness).
# SETUP (optional): a bash file run with SB=<the sandbox> before the line (stub checkpoints, logs).
# Saves exit, stdout.txt (the script's own output), rec.tsv (every stubbed call), pre_tree.txt (the sandbox before the
# line, after SETUP), tree.txt + md5.txt (after the line).
# Same as serrano's scratchpad/dry_b/run.sh (2026-10-11) except H and the GPU argument.
H=${DRY_HOME:-/scratch/cowifornia/nd-rl/think/dry}; SB=$H/sb
script=$1 mode=$2 seed=$3 arm=$4 res=$5 setup=${6:-}
/usr/bin/rm -rf "$SB" "$H/run"; mkdir -p "$H/run" "$res"; cp -a "$H/sb_template" "$SB"
if [ -n "$setup" ]; then SB=$SB /bin/bash -e "$setup" || { echo "setup $setup failed"; exit 1; }; fi
(cd "$SB" && /usr/bin/find . -mindepth 1 -printf '%y %p\n' | LC_ALL=C sort) > "$H/run/pre_tree.txt"
envs=(HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$H/stubs:/usr/local/bin:/usr/bin:/bin" LANG=C.UTF-8
      DRY_HARNESS="$H" DRY_REC="$H/run/rec.tsv" DRY_SB="$SB" DRY_LIB="$H/lib_dry.sh" DRY_PY="$H/pybin/python")
[ "$mode" = smoke ] && envs+=(SMOKE=1)
[ -n "${DRY_GPU_MEM:-}" ] && envs+=(DRY_GPU_MEM="$DRY_GPU_MEM")  # a busy GPU for waitfree
[ -n "${DRY_GPU_PID:-}" ] && envs+=(DRY_GPU_PID="$DRY_GPU_PID")
[ -n "${DRY_KNOB_FULL:-}" ] && envs+=(KNOB_FULL="$DRY_KNOB_FULL")  # a knob arm's full reads (2026-10-16)
: > "$H/run/rec.tsv"
if [ "$script" = queue ]; then  # run.sh queue MODE - - RESDIR [SETUP] -- QUEUE ARGS...: queue.sh.dry instead of a line
  shift 6; [ "${1:-}" = -- ] && shift
  env -i "${envs[@]}" /bin/bash "$H/queue.sh.dry" "$@" > "$H/run/stdout.txt" 2>&1; echo $? > "$H/run/exit"
else
  env -i "${envs[@]}" /bin/bash "$script" "${DRY_GPU:-3}" "$seed" "$arm" > "$H/run/stdout.txt" 2>&1; echo $? > "$H/run/exit"
fi
/usr/bin/sleep 0.3  # background stubs (the watchdog launches) finish recording
(cd "$SB" && /usr/bin/find . -mindepth 1 -printf '%y %p\n' | LC_ALL=C sort) > "$H/run/tree.txt"
(cd "$SB" && /usr/bin/find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r md5sum) > "$H/run/md5.txt"
[ -d "$SB/runs/logs" ] && { mkdir -p "$res/logs"; cp -a "$SB/runs/logs/." "$res/logs/"; }
cp "$H/run/"* "$res/"
