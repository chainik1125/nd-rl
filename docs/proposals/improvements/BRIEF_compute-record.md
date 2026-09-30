# Run brief: compute-record — the scripts record their own compute

Run id `compute-record`. Repository `~/work/compute-record` (branch `dan_compute-record` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Budget **$2**, ceiling **4
pod-hours**: one short GPU job to check the GPU timing; everything else on CPU. Authorised by Dan
2026-09-29.

## Why

`AGENT_POLICY.md` (2026-09-29) asks every arm to record its compute: GPU-seconds and GPU type, generated
tokens (attempts, actions), training steps and tokens, and Lean / environment checks. They go into
results-registry rows with the metric names `gpu_seconds`, `gen_tokens`, `train_steps`, `train_tokens`,
`lean_checks`, labelled with `arm`, `seed` and `labels.round`. Until now agents derive these from job logs
by hand. Make the scripts write them, so every run gets them for free and comparisons can be checked for
compute afterwards (Dan: "I mostly just want to make sure we're recording compute used").

## What to do

1. **One helper in `record.py`,** for example `with record.compute(arm=…, seed=…, round=…) as c:`.
   - It times the block: wall-clock × the number of GPUs in use gives `gpu_seconds`. It records the GPU
     type (`torch.cuda.get_device_name`) in `labels.gpu`, and `labels.device = cpu` on CPU.
   - It exposes counters (`c.gen_tokens += …`) and writes one row per metric at exit, through the
     existing `record()` path. It follows the existing environment switches (`ND_OFFLINE`,
     `ND_REGISTRY`).
   - Overhead must be negligible. Measure it.
2. **Instrument the scripts that do the work.** Count only real work: non-pad tokens, samples actually
   decoded, and texts actually sent to Lean, not cache hits. At least:
   - Stage-1 training (`train.py`, `fast_train.py`, `state_train.py`): steps, training tokens,
     GPU-seconds.
   - Sampling (`sample.py`, `state_sample.py`): generated tokens, attempts or actions, GPU-seconds.
   - The RL drivers (`ladder_ei.py`, `state_ladder_ei.py`, `expert_iter.py`, `grpo.py`): per round, the
     sampling, fine-tune and judging totals.
   - The judge (`lean_gate.py` / `lean_judge.py`): Lean checks, and Lean process-seconds as
     `labels.lean_s`.
3. **`registry_merge.py`:** a per-arm compute table (sums by run, arm, seed and round).
4. **Tests:** extend the CPU smoke test (`tests/test_smoke_train_sample.py`, or a new test that CI runs) to
   assert that a tiny train + sample + judge writes the compute rows with the right labels, and that the
   counters match an independent count (e.g. tokens in the sampled file).
5. **Check on a GPU once:** one short job (≤ 15 min) on the smallest GPU pod. Its `gpu_seconds` must agree
   with the job's own wall-clock within 5 %.
6. **Documentation:** `REGISTRY.md` gets the metric names, what each counts, and what it does not (e.g.
   Lean-cache hits). The same in one paragraph of `CONTRIBUTING.md`.

**Coordination.** `frontier-supply` and `search-expert` start after `long-pool-2`. If this run has merged
by then, they get the rows for free; otherwise they fall back to job logs. Merge as soon as review passes.
Keep the changes to the shared scripts small, to limit conflicts with those runs.

## Pre-registration (before the first pod)

- Expected overhead < 1 % of step time.
- The counters match independent counts exactly (tokens, checks) or within 5 % (GPU-seconds vs
  wall-clock).
- CI stays green.

## Deliverables

- Merged into the fork's `dan` after review.
- `preregistration/compute-record.md`, `run_compute_record.md` (≤ 250 words), `log.md`, and `STATUS.md`
  line `COMPUTE-RECORD DONE <UTC>`.
- Pods deleted; `touch ~/runs/compute-record/executor.done`.

Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
