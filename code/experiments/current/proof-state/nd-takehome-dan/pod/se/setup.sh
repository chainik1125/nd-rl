#!/usr/bin/env bash
# Pod setup for run state-env: Lean 4.34.0 (core) via elan, the run's data from the bucket, gate self-tests.
cd /workspace/nd-takehome; mkdir -p artifacts/se ckpts/se data/p2 data/ladder
{
  [ -x ~/.elan/bin/lean ] || { curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.34.0; }
  ~/.elan/bin/lean --version
  nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  pip -q install --break-system-packages -U huggingface_hub 2>&1 | tail -1
  B=hf://buckets/dan-pandori/nd-rl/lean-format/data
  hf buckets cp $B/p2/train_depth3_f0_a1.jsonl data/p2/train_depth3_f0_a1.jsonl
  hf buckets cp $B/p2/heldout.jsonl data/p2/heldout.jsonl
  hf buckets cp $B/ladder/rl_targets.jsonl data/ladder/rl_targets.jsonl
  hf buckets cp $B/ladder/transfer.jsonl data/ladder/transfer.jsonl
  wc -l data/p2/*.jsonl data/ladder/*.jsonl
  echo SETUP_DONE
} > artifacts/se/setup.log 2>&1
