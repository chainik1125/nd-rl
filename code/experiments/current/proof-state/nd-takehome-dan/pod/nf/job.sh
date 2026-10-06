#!/usr/bin/env bash
# One job. Usage: bash pod/nf/job.sh <jobname> '<command line>'   Log artifacts/nf/logs/<jobname>.log ; markers artifacts/nf/<jobname>.done|.failed
cd /workspace/nd-takehome; export CUDA_MEM_FRACTION=${CUDA_MEM_FRACTION:-0.21} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LEAN_GATE_WORKERS=${LEAN_GATE_WORKERS:-3} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ND_SAMPLE_COMPACT=0 PATH=$HOME/.elan/bin:$PATH
J=$1; shift; mkdir -p artifacts/nf/logs
export LEAN_GATE_LOG=artifacts/nf/gate_$J.jsonl
echo "START $(date -u +%FT%TZ) $*" > artifacts/nf/logs/$J.log
if bash -c "$*" >> artifacts/nf/logs/$J.log 2>&1; then echo "END $(date -u +%FT%TZ)" >> artifacts/nf/logs/$J.log; touch artifacts/nf/$J.done; else echo "FAILED $(date -u +%FT%TZ)" >> artifacts/nf/logs/$J.log; touch artifacts/nf/$J.failed; fi
