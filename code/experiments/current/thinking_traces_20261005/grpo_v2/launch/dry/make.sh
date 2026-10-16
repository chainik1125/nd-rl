#!/bin/bash
# Build the dry-run harness for the GRPO v2 launchers (nothing here runs python or touches a GPU).
#   bash make.sh [DIR]   DIR holds line.sh, lib.sh and queue.sh (default: the runtime install,
#                        runs/grpo_v2/{line,queue}.sh and runs/followup_20261007/lib.sh)
# Rebuilt on ynez 2026-10-13 after serrano's /scratch was lost; mirrors serrano's
# scratchpad/dry{,_b}/make.sh of 2026-10-11 (same stubs, same record columns), plus queue.sh and a
# column 8 with the environment the stubbed command saw.
#   stubs/        PATH stubs that record their argv (timeout, python3, nvidia-smi, sleep, rm, git) or fake output (date)
#   pybin/python  stands in for $PY: records argv; -c check -> 0; the preflight -> 22 "ok" lines; `-` (ARM_mb's
#                 averaging) -> its output file, holding the script it read
#   lib_dry.sh    lib.sh with only R= pointed at the sandbox ($DRY_SB/runs)
#   line.sh.dry   line.sh with only the `source lib.sh` line and the `PY=` line swapped
#   queue.sh.dry  queue.sh with only the `source lib.sh` line swapped (it runs $R/grpo_v2/line.sh = line.sh.dry)
#   sb_template   sandbox inputs the shell itself tests or heads: Stage-1 inits (empty) for seeds 0-2, 8-row pools
set -eu
H=${DRY_HOME:-/scratch/cowifornia/nd-rl/think/dry}
RR=/scratch/cowifornia/nd-rl/think/runs
if [ -n "${1:-}" ]; then LINE=$1/line.sh LIB=$1/lib.sh QUEUE=$1/queue.sh
else LINE=$RR/grpo_v2/line.sh LIB=$RR/followup_20261007/lib.sh QUEUE=$RR/grpo_v2/queue.sh; fi
mkdir -p "$H"
rm -rf "$H/stubs" "$H/pybin" "$H/sb_template"
mkdir -p "$H/stubs" "$H/pybin"
{ echo "line.sh  $(md5sum < "$LINE" | cut -c1-32)  $LINE"; echo "lib.sh   $(md5sum < "$LIB" | cut -c1-32)  $LIB"
  echo "queue.sh $(md5sum < "$QUEUE" | cut -c1-32)  $QUEUE"; } > "$H/sources.txt"

cat > "$H/rec.sh" <<'EOF'
# sourced by the dry-run stubs: append one TSV record of this call to $DRY_REC
dry_rec() {  # KIND ARGV...
  local kind=$1; shift
  local fd1 fd2 cvd envh argv envs v
  fd1=$(/usr/bin/readlink /proc/$$/fd/1 2>/dev/null || echo '?'); fd2=$(/usr/bin/readlink /proc/$$/fd/2 2>/dev/null || echo '?')
  case $fd1 in pipe:*) fd1=pipe ;; esac; case $fd2 in pipe:*) fd2=pipe ;; esac
  if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then cvd="CVD=[$CUDA_VISIBLE_DEVICES]"; else cvd="CVD=unset"; fi
  envh=$(/usr/bin/env -0 | /usr/bin/grep -zv '^_=' | LC_ALL=C /usr/bin/sort -z | /usr/bin/md5sum | /usr/bin/cut -c1-12)
  argv=$(printf '%q ' "$@")
  envs=""
  for v in ND_OFFLINE PYTHONDONTWRITEBYTECODE ND_REGISTRY_SYNC CUDA_DEVICE_ORDER PYTORCH_CUDA_ALLOC_CONF ND_LEAN TMPDIR SMOKE; do
    if [ -n "${!v+x}" ]; then envs+="$v=${!v} "; else envs+="$v- "; fi
  done
  printf '%s\t%s\t%s\t%s\t%s\tenv=%s\t%s\t%s\n' "$kind" "$PWD" "$cvd" "$fd1" "$fd2" "$envh" "${argv% }" "${envs% }" >> "$DRY_REC"
}
EOF

cat > "$H/stubs/timeout" <<'EOF'
#!/bin/bash
# records the wrapped training/eval command; fakes the checkpoint the line checks for, never runs it
. "$DRY_HARNESS/rec.sh"; dry_rec timeout "$@"
shift  # the time limit
mod="" out="" name="" iters="" rounds=""
a=("$@")
for ((i = 0; i < ${#a[@]}; i++)); do
  case ${a[i]} in
    -m) mod=${a[i+1]} ;; --out) out=${a[i+1]} ;; --name) name=${a[i+1]} ;;
    --iters) iters=${a[i+1]} ;; --rounds) rounds=${a[i+1]} ;;
  esac
done
case $out in "$DRY_SB"/*) ;; *) echo "DRY: --out outside the sandbox: '$out'" >&2; exit 97 ;; esac
case $mod in
  code.thinking.grpo) /usr/bin/mkdir -p "$out/ckpts" && : > "$out/ckpts/${name}_it$iters.pt"; echo "it $iters: dry" ;;
  code.thinking.ei) /usr/bin/mkdir -p "$out/ckpts" && : > "$out/ckpts/${name}_r$rounds.pt"; echo "ei dry" ;;
  code.thinking.eval) /usr/bin/mkdir -p "$out"; echo "[dry] eval" ;;
  *) echo "DRY: unexpected module '$mod'" >&2; exit 98 ;;
esac
EOF

cat > "$H/pybin/python" <<'EOF'
#!/bin/bash
# stands in for $PY in the direct (non-timeout) calls: the import check and the preflights
. "$DRY_HARNESS/rec.sh"; dry_rec py "$@"
case $1 in
  -c) exit 0 ;;
  -m) case $2 in *.preflight) for i in $(/usr/bin/seq 1 22); do echo "ok dry $i"; done; exit 0 ;; esac ;;
  -) case ${2:-} in  # ARM_mb's checkpoint averaging (a script on stdin): its output holds the script
       "$DRY_SB"/*) /usr/bin/mkdir -p "$(/usr/bin/dirname "$2")" && /usr/bin/cat > "$2"; echo "averaged dry"; exit 0 ;;
     esac
     echo "DRY: averaging outside the sandbox: '${2:-}'" >&2; exit 97 ;;
esac
echo "DRY: unexpected direct python call: $*" >&2; exit 98
EOF

for k in python3 sleep rm; do
  printf '#!/bin/bash\n. "$DRY_HARNESS/rec.sh"; dry_rec %s "$@"\nexit 0\n' $k > "$H/stubs/$k"
done
cat > "$H/stubs/nvidia-smi" <<'EOF'
#!/bin/bash
. "$DRY_HARNESS/rec.sh"; dry_rec nvidia-smi "$@"
case "$*" in *--query-gpu=memory.used*) echo "${DRY_GPU_MEM:-0}" ;; *--query-compute-apps*) [ -n "${DRY_GPU_PID:-}" ] && echo "$DRY_GPU_PID" ;; esac
exit 0
EOF
cat > "$H/stubs/git" <<'EOF'
#!/bin/bash
# read-only rev-parse goes to the real git; anything else is refused
. "$DRY_HARNESS/rec.sh"; dry_rec git "$@"
[ "$1" = -C ] && [ "$3" = rev-parse ] && exec /usr/bin/git "$@"
echo "DRY: unexpected git call: $*" >&2; exit 99
EOF
cat > "$H/stubs/date" <<'EOF'
#!/bin/bash
case "$*" in *+%s*) echo 1700000000 ;; *) echo 00:00 ;; esac
EOF
chmod +x "$H"/stubs/* "$H/pybin/python"

# lib.sh with R= moved into the sandbox (one line)
sed -e 's|^R=/scratch/cowifornia/nd-rl/think/runs;|R=$DRY_SB/runs;|' "$LIB" > "$H/lib_dry.sh"
[ "$(diff "$LIB" "$H/lib_dry.sh" | grep -c '^[<>]')" = 2 ] || { echo "lib transform did not change exactly 1 line"; exit 1; }

# line.sh with the two environment lines swapped
sed -e 's|^source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh$|source "$DRY_LIB"|' \
    -e 's|^PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python$|PY=$DRY_PY|' "$LINE" > "$H/line.sh.dry"
[ "$(diff "$LINE" "$H/line.sh.dry" | grep -c '^[<>]')" = 4 ] || { echo "line.sh transform did not change exactly 2 lines"; exit 1; }
# queue.sh with the source line swapped
sed -e 's|^source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh$|source "$DRY_LIB"|' "$QUEUE" > "$H/queue.sh.dry"
[ "$(diff "$QUEUE" "$H/queue.sh.dry" | grep -c '^[<>]')" = 2 ] || { echo "queue.sh transform did not change exactly 1 line"; exit 1; }

# sandbox inputs (synthetic 8-row pools: python is stubbed, so only the shell's head/cp/[ -s ] see them)
T=$H/sb_template/runs
mkdir -p "$T/logs" "$T/tmp" "$T/grpo_v2" "$T/pools/curriculum/L2" "$T/pools/curriculum/L3" "$T/pools/curriculum_3p"
for s in 0 1 2; do for m in S1g3 T0S; do mkdir -p "$T/stage1/${m}_s$s"; : > "$T/stage1/${m}_s$s/model.pt"; done; done
for l in 2 3; do
  for i in 1 2 3 4 5 6 7 8 9 10; do echo "{\"dry\": \"L$l row $i\"}"; done > "$T/pools/curriculum/L$l/pool.jsonl"
  for i in 1 2 3 4 5 6 7 8 9 10; do echo "{\"dry\": \"L${l}_3p row $i\"}"; done > "$T/pools/curriculum_3p/L${l}_3p.jsonl"
done
cp "$H/line.sh.dry" "$T/grpo_v2/line.sh"  # what queue.sh.dry runs ($R/grpo_v2/line.sh)
echo "harness ready in $H"; cat "$H/sources.txt"
