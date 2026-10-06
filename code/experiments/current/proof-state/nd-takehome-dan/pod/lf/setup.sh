#!/usr/bin/env bash
# Pod setup: Lean 4.34.0 (core) via elan, unpack the training set, self-test of the Lean gate. Log: artifacts/lf/logs/setup.log
cd /workspace/nd-takehome; mkdir -p artifacts/lf/logs ckpts/lf ckpts/ladder
{
  [ -x ~/.elan/bin/lean ] || { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0; }
  ~/.elan/bin/lean --version
  [ -f data/train.jsonl ] || gunzip -k data/train.jsonl.gz
  nproc; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  LEAN_GATE_LOG=artifacts/lf/gate_selftest.jsonl python3 pod/lf/gate_selftest.py
  echo SETUP_DONE
} > artifacts/lf/logs/setup.log 2>&1
