#!/usr/bin/env bash
# g0_reads.sh <line> <seed> <snap> <it>: runs ON THE POD, started by g0_monitor.py for a COLLAPSED line after its driver is
# done, when the driver's cross reads used a snapshot at or after the collapse. Reads <snap> (the last snapshot before
# the collapse, PREREG amendment) under the two tags the line was not trained under, exactly as g0_driver.sh's
# read_one; then touch $AT/DONE2_<line>.
set -uo pipefail
line=$1; seed=$2; snap=$3; it=$4
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
X=/workspace/g0exp/author_tags_20261110
export AT=/workspace/atg0
LW=${G0_LEAN_WORKERS:-8}; RF=${G0_READ_FRAC:-0.3}
utc() { date -u +%FT%TZ; }
tag=${line:1:1}
R=$AT/reads/cross_g0_p10_s$seed
mkdir -p "$R"
echo "{\"line\": \"$line\", \"snap\": \"$snap\", \"it\": $it, \"why\": \"last snapshot before the collapse\", \"utc\": \"$(utc)\"}" > "$R/precollapse_$line.json"
for t in A B none; do
  [[ $t == "$tag" ]] && continue
  out="$R/t${tag}${it}_${t}.json"
  [[ -s $out ]] && continue
  echo "$(utc) read $snap tag $t round $it -> $out"
  env ND_OFFLINE=1 ND_REGISTRY_SYNC=0 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS="$LW" OMP_NUM_THREADS=8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
    LEAN_GATE_DUMP="$R/lean_dump_$line.jsonl" ND_REGISTRY_DIR="$R/registry_$line" LEAN_GATE_LOG="$R/lean_gate_$line.jsonl" \
    DANREAD_MEM_FRAC="$RF" timeout 3600 "$PY" "$X/tag_danread.py" --tag "$t" -- --ckpt "$snap" --seed "$seed" --round "$it" \
    --out "$out" --only106 --k 64 --batch 128 >> "$R/reads_$line.log" 2>&1
  echo "$(utc) read rc $?"
done
touch "$AT/DONE2_$line"
