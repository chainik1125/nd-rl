#!/usr/bin/env bash
# Pod setup: Lean 4.34.0 (core) via elan, self-test of the Lean gate, CPU quota. Log: artifacts/nf/logs/setup.log
cd /workspace/nd-takehome; mkdir -p artifacts/nf/logs ckpts/nf data/nf ckpts/lf ckpts/dsg ckpts/dsc data/dsc
{
  [ -x ~/.elan/bin/lean ] || { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0; }
  ~/.elan/bin/lean --version
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  PATH=$HOME/.elan/bin:$PATH LEAN_GATE_WORKERS=6 LEAN_GATE_LOG=artifacts/nf/gate_selftest.jsonl python3 pod/lf/gate_selftest.py
  echo SETUP_DONE
} > artifacts/nf/logs/setup.log 2>&1
