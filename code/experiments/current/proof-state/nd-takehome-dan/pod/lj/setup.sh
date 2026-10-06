#!/usr/bin/env bash
# Pod setup for run lean-judge: Lean 4.34.0 (core) via elan, then the Lean-only judge's own test suite.
set -x; mkdir -p artifacts/lj
{
  [ -x ~/.elan/bin/lean ] || { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0; }
  ~/.elan/bin/lean --version
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "no cpu.max"
  nproc
  PATH=$HOME/.elan/bin:$PATH LEAN_GATE_WORKERS=6 python3 tests/test_lean_only_judge.py
  PATH=$HOME/.elan/bin:$PATH LEAN_CHECK_WORKERS=6 python3 lean_check.py --selftest
} 2>&1 | tee artifacts/lj/setup.log
