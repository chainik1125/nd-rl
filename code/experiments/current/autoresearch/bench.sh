#!/usr/bin/env bash
# On the pod: run N harness processes concurrently (seeds 0..N-1), optionally under CUDA MPS.
# Usage: bash bench.sh <out-root> <n> <mps 0|1>
set -uo pipefail
export ND_FORK=/workspace/nd-takehome
OUT=$1; N=$2; MPS=$3
mkdir -p "$OUT"
[ -f "$ND_FORK/data/train.jsonl" ] || bash /workspace/autoresearch/remote.sh /tmp/warm --help >/dev/null 2>&1
if [ "$MPS" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
  nvidia-cuda-mps-control -d && sleep 2
  echo "mps servers: $(echo get_server_list | nvidia-cuda-mps-control)" | tee "$OUT/mps.txt"
fi
export ND_TOTAL_SECS=3000
cd /workspace/autoresearch
for i in $(seq 0 $((N - 1))); do
  python3 harness.py --out "$OUT/seed$i" --seed "$i" > "$OUT/seed$i.log" 2>&1 &
done
wait
if [ "$MPS" = 1 ]; then echo "mps servers after: $(echo get_server_list | nvidia-cuda-mps-control)" >> "$OUT/mps.txt"; echo quit | nvidia-cuda-mps-control; fi
grep -h "SUMMARY\|pretrain returned\|samples/s" "$OUT"/seed*.log
