#!/usr/bin/env bash
# g0_driver.sh <line> <seed> <base_none 0|1>: runs ON THE POD (copied to /workspace/g0exp by g0_pod.sh prep).
# 1. tag_line.py --line <line> --gpu 0 with ND_GUARD=off, AT=/workspace/atg0 (own tree), detached by g0_pod.sh launch.
#    The launcher's one-job-per-GPU lock (tag_launch.run_job) is bypassed on purpose: a second job shares the pod's GPU.
#    At G0_T_END (default 3.3 h after the driver starts) a STOP file ends GRPO cleanly (tag_line then finishes its
#    pending snapshot reads); hard timeout 25 min later. A segment that dies of CUDA OOM (GPU shared with another job) is continued as the next
#    segment from its newest rolling checkpoint (tag_line's own continuation path), at most 4 times; an OOM before the
#    first rolling checkpoint of segment 1 restarts segment 1 (the dead folder is moved aside). Any other failure ends it.
# 2. Cross-tag reads of the newest snapshot under the two tags it was not read under in-run (+ the base under none
#    when base_none=1), the same read as in-run (tag_danread.py, strict 106, k 64, batch 128, round = iteration).
# 3. touch $AT/DONE_<line>.
set -uo pipefail
line=$1; seed=$2; bn=$3
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=/workspace/g0exp/author_tags_20261110
export AT=/workspace/atg0 ND_GUARD=off G0_LEAN_WORKERS=${G0_LEAN_WORKERS:-8} G0_READ_FRAC=${G0_READ_FRAC:-0.3}
utc() { date -u +%FT%TZ; }
LD=$AT/runs/$line
echo "$(utc) driver start $line seed $seed base_none $bn pid $$ host $(hostname) lean_workers $G0_LEAN_WORKERS read_frac $G0_READ_FRAC"
cd "$X" || exit 1
T_END=${G0_T_END:-$(( $(date +%s) + 11880 ))}  # G0_T_END: absolute stop time (epoch s); default 3.3 h from now
seg=${G0_SEG:-1}; resume=${G0_RESUME:-}; retries=0  # G0_SEG/G0_RESUME: continue a moved line (tag_line resume)
echo "$(utc) stop at $(date -u -d @$T_END +%FT%TZ), first seg $seg resume ${resume:-none}"
while :; do
  left=$(( T_END - $(date +%s) ))
  (( left > 300 )) || { echo "$(utc) no time left for seg $seg"; break; }
  args=(--line "$line" --gpu 0 --seg "$seg"); [[ -n $resume ]] && args+=(--resume "$resume")
  ( sleep "$left"; touch "$LD/seg$seg/STOP"; echo "$(utc) stop time reached: STOP written (seg $seg)" ) &
  sleeper=$!
  echo "$(utc) tag_line ${args[*]} (timeout $((left + 1500)) s)"
  CUDA_VISIBLE_DEVICES=0 LEAN_GATE_WORKERS="$G0_LEAN_WORKERS" timeout --kill-after=300 $((left + 1500)) "$PY" tag_line.py "${args[@]}"
  rc=$?
  kill "$sleeper" 2>/dev/null
  how=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['how'])" "$LD/seg$seg/end.json" 2>/dev/null || echo none)
  echo "$(utc) tag_line seg $seg end rc $rc how $how"
  if [[ $how == failed ]] && grep -q "OutOfMemoryError" "$AT/logs/${line}_seg$seg.log" && (( retries < 4 )); then
    retries=$((retries + 1))
    roll=$(ls "$LD"/seg*/ckpts/"${line}"_it*_rolling.pt 2>/dev/null | sed -E 's/.*_it([0-9]+)_rolling\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    if [[ -n $roll ]]; then
      resume=$roll; seg=$((seg + 1))
      echo "$(utc) OOM: continuing as seg $seg from $roll (retry $retries)"
    elif [[ $seg == 1 ]]; then
      mv "$LD" "$AT/oom_$line.$(date +%s)"; mkdir -p "$AT/oom_logs"; mv "$AT/logs/${line}_seg1.log" "$AT/logs/${line}.events.log" "$AT/oom_logs/" 2>/dev/null
      echo "$(utc) OOM before the first rolling checkpoint: restarting seg 1 (retry $retries)"
    else
      echo "$(utc) OOM and no rolling checkpoint: ending"; break
    fi
    sleep 30; continue
  fi
  break
done
tag=${line:1:1}
R=$AT/reads/cross_g0_p10_s$seed
mkdir -p "$R"
read_one() {  # <ckpt> <tag> <out> <round>
  [[ -s $3 ]] && { echo "$(utc) exists $3"; return; }
  echo "$(utc) read $1 tag $2 round $4 -> $3"
  env ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS="$G0_LEAN_WORKERS" OMP_NUM_THREADS=8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
    LEAN_GATE_DUMP="$R/lean_dump_$line.jsonl" ND_REGISTRY_DIR="$R/registry_$line" LEAN_GATE_LOG="$R/lean_gate_$line.jsonl" \
    DANREAD_MEM_FRAC="$G0_READ_FRAC" timeout 3600 "$PY" "$X/tag_danread.py" --tag "$2" -- --ckpt "$1" --seed "$seed" --round "$4" \
    --out "$3" --only106 --k 64 --batch 128 >> "$R/reads_$line.log" 2>&1
  local r=$?
  echo "$(utc) read rc $r"
  if (( r != 0 )) && grep -q OutOfMemoryError "$R/reads_$line.log"; then  # shared GPU: one retry after a pause
    sleep 120; echo "$(utc) retry after OOM"
    env ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS="$G0_LEAN_WORKERS" OMP_NUM_THREADS=8 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
      LEAN_GATE_DUMP="$R/lean_dump_$line.jsonl" ND_REGISTRY_DIR="$R/registry_$line" LEAN_GATE_LOG="$R/lean_gate_$line.jsonl" \
      DANREAD_MEM_FRAC="$G0_READ_FRAC" timeout 3600 "$PY" "$X/tag_danread.py" --tag "$2" -- --ckpt "$1" --seed "$seed" --round "$4" \
      --out "$3" --only106 --k 64 --batch 128 >> "$R/reads_$line.log" 2>&1
    echo "$(utc) read rc $?"
  fi
}
snap=$(ls "$LD"/seg*/ckpts/"${line}"_it*_snap.pt 2>/dev/null | sed -E 's/.*_it([0-9]+)_snap\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
if [[ -n $snap ]]; then
  it=$(basename "$snap" | sed -E 's/.*_it([0-9]+)_snap\.pt$/\1/')
  echo "$(utc) final snapshot $snap (it $it)"
  echo "{\"line\": \"$line\", \"snap\": \"$snap\", \"it\": $it, \"utc\": \"$(utc)\"}" > "$R/final_$line.json"
  for t in A B none; do
    [[ $t == "$tag" ]] && continue
    read_one "$snap" "$t" "$R/t${tag}${it}_${t}.json" "$it"
  done
else
  echo "$(utc) no snapshot: no cross reads of the trained model"
fi
[[ $bn == 1 ]] && read_one "$AT/bases/stage1_tagAB_p10_s$seed.pt" none "$R/base_none.json" 0
touch "$AT/DONE_$line"
echo "$(utc) driver done"
