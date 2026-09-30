---
written_on: 2026-09-30
written_by: agent:claude
---

# ndbench benchmarking run: runbook

Companion to `PLAN.md` (the full plan, §4 for the GPU schedule table and §A for the environment/
safety rules) and `schedule.yaml` (the concrete commands). This note only covers *how to launch*
each job safely and *when to stop*; see `README.md` for what the folder holds and where outputs go.

## Prerequisites

- Run everything from the worktree root: `cd /scratch/cowifornia/nd-rl/bench/nd-rl`.
- `code/.venv` is a symlink to the main worktree's venv (plan §A1) -- do **not** run
  `make install-dev`/`uv sync` from here.
- Every suite a job's `run`/`edl` command names must already be built (`python -m code.ndbench
  build --suite <name> --version v1 --config configs/<name>.yaml`); building is CPU-only and
  subject to the CPU rules below, not the GPU ones.
- `configs/models.yaml` holds the `base`/`rl`/`armC` model labels (WP8 integration fix list item
  6); `--model base`/`--model rl`/`--model armC` resolve through it, with a sha256 check against
  the checkpoint on disk.

## The only supported way to launch a GPU job

**Set `CUDA_VISIBLE_DEVICES=4` for the whole process before it starts**, then pass `--device cuda`:

```bash
CUDA_VISIBLE_DEVICES=4 python -m code.ndbench run --suite parts_pool --version v1 \
    --model base --k 32 --device cuda --out /scratch/cowifornia/nd-rl/bench/out
```

This is not optional or a convenience -- as of WP8 integration fix list item 8, both `cli.py`'s
`--device cuda` guard and `runner.gpu.lease` itself *refuse* (a friendly `SystemExit`/`RuntimeError`,
not a crash) unless `CUDA_VISIBLE_DEVICES` already equals `"4"` in the environment when they run.
Setting it any later -- after `import torch`, or inside `lease()` itself -- is unreliable (a CUDA
context, once initialised, ignores later changes to the variable) and is exactly the bug this fix
closed. Do not work around the refusal by setting the variable after the process starts.

**GPUs 0-3 belong to another user and must never be touched.** Only GPU 4 may ever be leased; every
`run`/`edl`/`throughput` invocation above must include `--device cuda` and the env var, or it will
default to CPU (fine for a dry run, useless for the real schedule).

Only one job at a time -- no concurrent GPU use, per plan §4.

## The stop rule (plan §4)

Before *every* job (not just once at the start), check whether the ledger still has room:

```bash
python -m code.ndbench budget --out /scratch/cowifornia/nd-rl/bench/out --cap-h 9.5 \
    --est-h <next job's est_min / 60>
```

`would_fit` is `spent_h + 1.5 * est_h <= cap_h`. If `false`, **skip that job and try the next one
in `schedule.yaml`'s order that does fit** (plan §4: "skip to the next job that fits (in order)").
Continue this way until either every job has been tried or the cap is reached; never round-robin
back to a skipped job in the same pass.

To see the whole schedule's projected stop-rule table at once (WP8 integration fix list item 10;
CPU-only, never touches the GPU or `gpulock`):

```bash
python -m code.ndbench budget --out /scratch/cowifornia/nd-rl/bench/out \
    --schedule code/experiments/current/benchmarking_20260930/schedule.yaml
```

This walks `schedule.yaml`'s `jobs` list top to bottom, using the store's *real* ledgered
`spent_h` as the starting point and each job's own `est_min` for the projection -- rerun it after
every job actually finishes (its real `gpu_s` differs from the estimate) to keep the projection
honest, not just once at the start.

**Mid-job stop:** if a running job passes 2x its own `est_min` without finishing, stop it gracefully
(let it finish its current chunk/checkpoint and mark the run `partial` rather than killing it
mid-write) rather than letting it run unbounded (plan §4).

**Hard stop:** whatever is in flight when the ledger hits 9.5 GPU-h total wall time, stop
everything -- there is no "just this last job" exception.

## The ledger

`runner.gpu.lease` appends one row to `out/store/ledger.jsonl` per lease (start/end UTC, `gpu_s`
wall time held, `status`), regardless of how the job exits (success, error, or `partial`).
`Store.ledger_total_s()` sums every row's `gpu_s`; `budget`'s `spent_h` is that sum in hours. This
is wall time the lock was *held*, not GPU-busy time -- a job that idles while acquiring a busy lock
still counts from when it actually acquired GPU 4, not from when it started waiting (`gpu.lease`'s
`wait_s` only affects how long it waits to acquire, not what gets ledgered once it does).

## CPU rules for build steps (plan §A2)

Shared 48-CPU host: at most **8 minlen worker processes** at a time (`MinlenCertifier(procs=...)`,
already capped internally at 8), any single CPU job under **~15 minutes** (split a suite build by
`families:`/`schemas:` subsets in its config if a full build risks exceeding that), and cache every
minlen label under `/scratch/cowifornia/nd-rl/bench/out/cache/`. No GPU use for a build step, ever.

## If something looks wrong

- `run`/`edl` failing immediately with a `sha256 ... does not start with the configured prefix`
  error means the checkpoint at that `configs/models.yaml` path changed (or the path is wrong) --
  stop and investigate before overriding the check; never edit the check away to make it pass.
- A `CUDA_VISIBLE_DEVICES` refusal (from either `cli.py` or `gpu.lease`) means the process was
  launched without the env var set from the start; relaunch it correctly rather than exporting the
  variable into the already-running shell and retrying (see "only supported way" above).
