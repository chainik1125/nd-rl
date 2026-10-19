#!/usr/bin/env bash
# Pass rates under T0_s0 (k 16, T 0.8, plain) for one or more candidate sets, on GPU 0 only.
# usage: run_passrate.sh NAME:SETS.jsonl [NAME:SETS.jsonl ...]   (out: runs/pools/passrate/NAME/)
# Waits (poll every 60 s, at most 60 min) until GPU 0 has < 500 MiB in use, then runs each set in
# turn; every passrate call is wrapped in `timeout` (90 min) and logs to runs/pools/logs/NAME.log.
set -u
R=/scratch/cowifornia/nd-rl/think/runs
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp \
  CUDA_VISIBLE_DEVICES=0
mkdir -p $R/pools/logs $R/pools/passrate
for spec in "$@"; do
  name=${spec%%:*} sets=${spec#*:}
  for i in $(seq 0 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$used" -lt 500 ] && break
    [ "$i" -eq 60 ] && { echo "$(date -u +%FT%TZ) GPU 0 still busy ($used MiB) after 60 min: giving up on $name"; exit 1; }
    echo "$(date -u +%FT%TZ) GPU 0 busy ($used MiB), waiting"
    sleep 60
  done
  echo "$(date -u +%FT%TZ) start $name ($sets)"
  timeout 5400 code/.venv/bin/python -m code.experiments.current.thinking_traces_20261005.pools.passrate \
    --ckpt $R/stage1/T0_s0/model.pt --sets "$sets" --out $R/pools/passrate/$name --k 16 \
    --temperature 0.8 --batch 1024 --chunk 400 >> $R/pools/logs/$name.log 2>&1
  echo "$(date -u +%FT%TZ) end $name: exit $?"
done
