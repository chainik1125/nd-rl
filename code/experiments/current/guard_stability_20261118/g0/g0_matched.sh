#!/usr/bin/env bash
# g0_matched.sh <line> <seed> <N> <base_none 0|1>: ON THE POD, beside the running line (coordinator, 2026-11-18 5:00 pm:
# cross-tag table at a matched iteration N for all six lines, as well as at each line's end).
#   1. base_none=1: the base under no tag (once per seed; the driver skips it later if the file exists)
#   2. waits for the line's iteration-N snapshot and for the line's own in-run read of it to finish, then reads that
#      snapshot under the two tags it was not trained under: same tag_danread command as g0_driver.sh, --round N.
# Never more than one extra read at a time; a read is never started while the line's in-run read is running.
# Out: AT/reads/cross_g0_p10_s<seed>/matched_N<N>/t<T><N>_<tag>.json; DONE_<line> there at the end.
set -uo pipefail
line=$1; seed=$2; N=$3; bn=$4
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=/workspace/g0exp/author_tags_20261110
export AT=/workspace/atg0
LW=${G0_LEAN_WORKERS:-8}; RF=${G0_READ_FRAC:-0.3}
utc() { date -u +%FT%TZ; }
tag=${line:1:1}
LD=$AT/runs/$line
R0=$AT/reads/cross_g0_p10_s$seed
R=$R0/matched_N$N
mkdir -p "$R"
echo "$(utc) matched-N reads: $line seed $seed N $N base_none $bn lean_workers $LW read_frac $RF"
wait_inrun() { while ps -eo args | grep -q "[t]ag_danread.py.*runs/$line/seg"; do sleep 15; done; }
read_one() {  # <ckpt> <tag> <out> <round> <logdir>
  [[ -s $3 ]] && { echo "$(utc) exists $3"; return; }
  wait_inrun
  echo "$(utc) read $1 tag $2 round $4 -> $3"
  env ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS="$LW" OMP_NUM_THREADS=8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
    LEAN_GATE_DUMP="$5/lean_dump_$line.jsonl" ND_REGISTRY_DIR="$5/registry_$line" LEAN_GATE_LOG="$5/lean_gate_$line.jsonl" \
    DANREAD_MEM_FRAC="$RF" timeout 3600 "$PY" "$X/tag_danread.py" --tag "$2" -- --ckpt "$1" --seed "$seed" --round "$4" \
    --out "$3" --only106 --k 64 --batch 128 >> "$5/reads_$line.log" 2>&1
  local r=$?
  echo "$(utc) read rc $r"
  if (( r != 0 )); then sleep 60; wait_inrun; echo "$(utc) retry"
    env ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS="$LW" OMP_NUM_THREADS=8 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
      LEAN_GATE_DUMP="$5/lean_dump_$line.jsonl" ND_REGISTRY_DIR="$5/registry_$line" LEAN_GATE_LOG="$5/lean_gate_$line.jsonl" \
      DANREAD_MEM_FRAC="$RF" timeout 3600 "$PY" "$X/tag_danread.py" --tag "$2" -- --ckpt "$1" --seed "$seed" --round "$4" \
      --out "$3" --only106 --k 64 --batch 128 >> "$5/reads_$line.log" 2>&1
    echo "$(utc) read rc $?"
  fi
}
[[ $bn == 1 ]] && read_one "$AT/bases/stage1_tagAB_p10_s$seed.pt" none "$R0/base_none.json" 0 "$R0"
snap=""
until [[ -n $snap ]]; do
  snap=$(ls "$LD"/seg[0-9]*/ckpts/"${line}_it${N}_snap.pt" 2>/dev/null | tail -1)
  [[ -n $snap ]] || { [[ -e $AT/DONE_$line ]] && { echo "$(utc) line done without an it $N snapshot"; exit 0; }; sleep 30; }
done
segd=$(dirname "$(dirname "$snap")")
itf=$(printf "it%04d" "$N")
echo "$(utc) snapshot $snap; waiting for the in-run read $segd/reads/$itf.json"
until [[ -s $segd/reads/$itf.json || -e $segd/reads/$itf.failed ]]; do sleep 20; done
echo "{\"line\": \"$line\", \"snap\": \"$snap\", \"it\": $N, \"own_read\": \"$segd/reads/$itf.json\", \"utc\": \"$(utc)\"}" > "$R/snap_$line.json"
for t in A B none; do
  [[ $t == "$tag" ]] && continue
  read_one "$snap" "$t" "$R/t${tag}${N}_${t}.json" "$N" "$R"
done
touch "$R/DONE_$line"
echo "$(utc) matched-N reads done"
