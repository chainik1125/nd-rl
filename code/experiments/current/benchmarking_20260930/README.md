---
written_on: 2026-09-30
written_by: agent:claude
---

# ndbench benchmarking experiment (2026-09-30)

Working folder for the `ndbench` capability-benchmark suite requested by Leon on 2026-09-30:
benchmarks B1-B8 (glue/compositionality, restricted families, skill profile, law x substitution
size, free lines/padding, prefix completion, excess description length, and a Yue et al.
base-vs-RL pass@k replication), plus an abstract model-adapter interface and a dashboard. Branch
`leon/benchmarking`, worktree `/scratch/cowifornia/nd-rl/bench/nd-rl`. See `experiment-summaries/`
(after results land) for the human-readable write-up this folder feeds; this README only covers
the working folder itself.

## What this folder holds

- `PLAN.md` -- the full plan: §0 verified facts/corrections, §1 architecture and interfaces, §2 the
  eight benchmarks' exact definitions, §3 the work-package table, §4 the GPU schedule, §5
  risks/decisions, and §A the main-agent amendments (environment, CPU/GPU safety, interp hooks,
  git rules) that override everything above them.
- `INTERP_SPEC.md` -- the follow-on interpretability spec (ordering-marginalised likelihood,
  RL-base steering vectors, weight interpolation, linear probes), owned by a separate interp coder
  and scheduled as G11+ once its own `configs/interp_*.yaml` exist.
- `configs/` -- one YAML per suite build (`textbook_dev.yaml`, `textbook_law.yaml`, `gen_ctrl.yaml`,
  `skill_pool.yaml`, `parts_pool.yaml`, `families.yaml`, `laws.yaml`, `padding.yaml`, `glue.yaml`,
  `glue_length_sweep.yaml`, `prefix.yaml`), `edl.yaml` for the excess-description-length protocol
  (`edl_dan12.yaml` for Dan's cap-12 models: premise cap 3, Dan's exclusion tiers, his own
  held-out null control),
  and `models.yaml` (the `base`/`rl`/`armC` model-label registry: adapter, checkpoint path, sha256
  prefix). Each config's header comment names the exact `python -m code.ndbench build/run/edl`
  invocation it goes with and any real-run prerequisites (e.g. `glue.yaml`/`padding.yaml` need a
  real G1 mastery run id filled in first).
- `schedule.yaml` -- plan §4's G0-G10 (+ reserve) schedule as concrete, ready-to-run
  `python -m code.ndbench ...` commands with their suite/version/model/k/adaptive rule and a
  per-job time estimate; a placeholder section notes where G11+ (interp) jobs slot in once they
  exist. Estimates are plan §4's ~300-samples/s planning figures -- G0 (throughput) measures the
  real number and should be used to rescale the rest before committing to the schedule.
- `runbook.md` -- how to actually launch a job (the `CUDA_VISIBLE_DEVICES=4`-first rule, no other
  way is supported), the plan §4 stop rule and how to check it (`budget --schedule`), the GPU
  ledger, and the CPU rules for build steps.

## Where outputs live

Everything this experiment produces goes under `/scratch/cowifornia/nd-rl/bench/out/`:

```
out/suites/<suite>/<version>/...     built Problem sets + SuiteManifest (python -m code.ndbench build)
out/store/runs.jsonl                 one row per run (python -m code.ndbench run)
out/store/items/<run_id>.jsonl       per-item, per-chunk ItemResult rows
out/store/samples/<run_id>.jsonl.gz  raw sampled attempts + verdicts (if keep_samples)
out/store/edl/<run_id>.jsonl         excess-description-length rows (python -m code.ndbench edl)
out/store/ledger.jsonl               GPU-hour ledger (runner.gpu.lease; see runbook.md)
out/cache/                           cached minlen certification labels
out/dash/                            rendered dashboard (python -m code.ndbench dash)
```

None of this lives in the worktree or gets committed; `code/experiments/current/
benchmarking_20260930/` only holds the plan/configs/schedule that produce it, plus (once results
exist) the summary this README points to.

## How to rerun

To run only some benchmarks, or only the ones that need no GPU, use the cost tiers in
[`code/ndbench/README.md`](../../../ndbench/README.md) (`run_schedule.py --schedule
code/ndbench/panel.yaml --model <label> --max-tier cheap`). The steps below are the original
full schedule.

1. From the worktree root, build whichever suites the next job needs, e.g.:
   ```bash
   python -m code.ndbench build --suite parts_pool --version v1 \
       --config code/experiments/current/benchmarking_20260930/configs/parts_pool.yaml \
       --out /scratch/cowifornia/nd-rl/bench/out
   ```
2. Check the schedule's stop-rule table before spending any GPU time:
   ```bash
   python -m code.ndbench budget --out /scratch/cowifornia/nd-rl/bench/out \
       --schedule code/experiments/current/benchmarking_20260930/schedule.yaml
   ```
3. Launch the next fitting job from `schedule.yaml`, exactly as printed there, with
   `CUDA_VISIBLE_DEVICES=4` set for the whole process (see `runbook.md` -- this is enforced, not
   just advised).
4. Once results exist, `python -m code.ndbench dash --out /scratch/cowifornia/nd-rl/bench/out
   --dash-dir /scratch/cowifornia/nd-rl/bench/out/dash` renders the analysis views.

See `AGENTS.md` (repo root) for the general research-workspace conventions (note dates/history,
where reusable code and tests live, literature citation) this experiment folder follows.
