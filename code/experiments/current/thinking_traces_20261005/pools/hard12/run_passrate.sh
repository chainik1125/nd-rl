#!/usr/bin/env bash
# One pass-rate shard on one GPU: k attempts per theorem (plain arm, T 0.8, line forcing, max_steps 96, Lean-judged)
# with the previous agent's pools/passrate.py.
# usage: run_passrate.sh GPU NAME SETS.jsonl [MODEL=T0|T1v1] [K=32] [SEED=7]   -> runs/pools/hard12_work/passrate/NAME/
# env: BUDGET (think budget; default 0 for T0, 256 for T1v1: the Stage-1 thinking arm), BATCH, CHUNK.
# Refuses to start unless the GPU has < 500 MiB in use; never touches GPU 0. Wrapped in `timeout` (90 min).
set -u
G=$1 NAME=$2 SETS=$3 MODEL=${4:-T0} K=${5:-32} SEED=${6:-7}
[ "$G" = 0 ] && { echo "GPU 0 is reserved"; exit 2; }
R=/scratch/cowifornia/nd-rl/think/runs
W=$R/pools/hard12_work
case $MODEL in T0) CK=$R/stage1/T0_s0/model.pt; B=${BUDGET:-0} ;; T1v1) CK=$R/stage1/T1v1_s0/model.pt; B=${BUDGET:-256} ;; *) CK=$MODEL; B=${BUDGET:-0} ;; esac
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G")
[ "$used" -lt 500 ] || { echo "$(date -u +%FT%TZ) GPU $G busy ($used MiB): not starting $NAME"; exit 3; }
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ND_REGISTRY_SYNC=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ND_LEAN=$HOME/.elan/bin/lean TMPDIR=$R/tmp \
  CUDA_VISIBLE_DEVICES=$G LEAN_GATE_WORKERS=${LEAN_GATE_WORKERS:-8}
mkdir -p $W/logs $W/passrate
echo "$(date -u +%FT%TZ) start $NAME on GPU $G ($SETS, $MODEL, k $K, seed $SEED, budget $B)" >> $W/logs/$NAME.log
timeout 5400 code/.venv/bin/python -m code.experiments.current.thinking_traces_20261005.pools.passrate \
  --ckpt $CK --sets "$SETS" --out $W/passrate/$NAME --k $K --budget $B --temperature 0.8 --max_steps 96 \
  --batch ${BATCH:-1024} --chunk ${CHUNK:-250} --seed $SEED >> $W/logs/$NAME.log 2>&1
echo "$(date -u +%FT%TZ) end $NAME: exit $?" >> $W/logs/$NAME.log
