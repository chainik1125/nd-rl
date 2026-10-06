---
written_on: 2026-10-06
written_by: agent:codex
status: frozen_collection_plan_not_launch_authorization
---

# Independent frozen-generator test pools

Collect all 400 theorems in `artifacts/learned_compute_allocation_20261006/data/main_v1/test.jsonl` in its immutable stored order, with eight attempts per theorem. Use a single combined-family process, seed 91006, batch 256 active attempts, temperature 0.8, maximum action 256 tokens and maximum 48 actions. This preserves MAIN DEV's active batch size, sampler, precision and caps, but combines the families rather than launching two separate family processes. No outcome-dependent filtering or replacement is permitted. The sampler dynamically refills active attempts in theorem/candidate order; the last batch may be smaller. Constructed references in input metadata never enter model prefixes or chooser inputs.

Use original r16 checkpoint SHA256 `ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24`. Collector SHA256 is `f7af455bd52acb178856aea9ebe91f5c5a9577e701e138c3f29a3eb1871455be`; renderer SHA256 is `a7fc12ebf184461a405bac1b48a1072da3a38302d0aac0da33e344b14b0ae980`. Both source files must be copied and hash-verified into the launch directory before invocation. Upstream commit is `4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c`; bind behavior-file hashes in the launch manifest. Sampling is BF16 autocast model, FP32 Gumbel-max logits at T0.8; diagnostics are BF16 full-forward replay and FP32 log-softmax/entropy (T0.8 behavioral likelihood and T1 base likelihood/entropy). Diagnostics are not bit-identical to cached decode.

Root owns two-phase evaluation authorization: first immutable decisions, models and signatures before generation; then a metadata-only seal of raw and transition public-pool hashes before any TEST inference or label inspection. Collector progress and summary include private counts, so redirect all collector output to an unread private log. Do not open its private JSONL, summary JSON or log during sealing. Renderer reads only public raw actions. Sealing reads only input IDs, public pools and source/config bytes; it neither consumes labels nor computes accuracy. Root appends the resulting metadata to the final registry with its parent freeze hash.

Commands below are intended, not executed. Replace GPU only after explicit coordinated release and a fresh occupancy check. Run from repository root, with `ND_OFFLINE=1` and local pure-output artifacts.

```bash
# Copy/hash-verification and PID/UTC/monotonic launch manifest precede this call.
ND_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 /data/users/dmitry/venvs/proof-state/bin/python \
  code/experiments/current/learned_compute_allocation_20261006/selection/collect.py \
  --data artifacts/learned_compute_allocation_20261006/data/main_v1/test.jsonl \
  --out artifacts/learned_compute_allocation_20261006/pools/main_test \
  --seed 91006 --k 8 --temperature 0.8 --batch 256 --max-action 256 --max-steps 48 \
  > artifacts/learned_compute_allocation_20261006/pools/main_test.private_generation.log 2>&1
ND_OFFLINE=1 /data/users/dmitry/venvs/proof-state/bin/python \
  code/experiments/current/learned_compute_allocation_20261006/selection/render_transitions.py \
  --input artifacts/learned_compute_allocation_20261006/pools/main_test/public.jsonl \
  --output artifacts/learned_compute_allocation_20261006/pools/main_test/public_transitions.jsonl
/data/users/dmitry/venvs/proof-state/bin/python \
  code/experiments/current/learned_compute_allocation_20261006/selection/seal_public_pool.py \
  --raw artifacts/learned_compute_allocation_20261006/pools/main_test/public.jsonl \
  --transitions artifacts/learned_compute_allocation_20261006/pools/main_test/public_transitions.jsonl \
  --expected-data artifacts/learned_compute_allocation_20261006/data/main_v1/test.jsonl \
  --expected-theorems 400 --expected-candidates 8 \
  --out artifacts/learned_compute_allocation_20261006/pools/main_test/public_seal.json
```

The launch manifest must record exact commands, seed, actual GPU, original checkpoint/input/collector/renderer/upstream hashes and monotonic process duration. Collector snapshots itself; renderer snapshots itself and records tokenizer/environment hashes. The planned sealer additionally checks exact theorem order, 400 rows, candidate IDs 0–7 and raw-action identity across public views. Root must authorize launch after registering this plan; this note alone does not authorize TEST scoring.
