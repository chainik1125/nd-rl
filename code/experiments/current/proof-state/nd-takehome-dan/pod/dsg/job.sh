#!/usr/bin/env bash
# One job. Usage: bash pod/dsg/job.sh <jobname> '<command line>'   Log artifacts/dsg/logs/<jobname>.log ; markers artifacts/dsg/<jobname>.done|.failed
cd /workspace/nd-takehome; export CUDA_MEM_FRACTION=${CUDA_MEM_FRACTION:-0.21} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LEAN_GATE_WORKERS=${LEAN_GATE_WORKERS:-12} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PATH=$HOME/.elan/bin:$PATH
J=$1; shift; mkdir -p artifacts/dsg/logs
export LEAN_GATE_LOG=artifacts/dsg/gate_$J.jsonl
echo "START $(date -u +%FT%TZ) $*" > artifacts/dsg/logs/$J.log
if bash -c "$*" >> artifacts/dsg/logs/$J.log 2>&1; then echo "END $(date -u +%FT%TZ)" >> artifacts/dsg/logs/$J.log; touch artifacts/dsg/$J.done; else echo "FAILED $(date -u +%FT%TZ)" >> artifacts/dsg/logs/$J.log; touch artifacts/dsg/$J.failed; fi
