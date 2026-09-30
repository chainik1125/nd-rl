---
written_on: 2026-09-30
written_by: agent:claude
---

# compute-record: the scripts record their own compute as registry rows (infrastructure)

Status: completed and reviewed 2026-09-30. No hard-constraint violation, nothing quarantined. Verdict
**stands**, with six rewordings (applied below; the fork files keep the executor's text). Infrastructure
only: no proof is counted and no accuracy is claimed, so no checker, seed count or noise floor applies. The
code is merged into the fork's `dan` (`3fc57e41`).

Question: can the scripts write the compute that `AGENT_POLICY.md` (2026-09-29) asks for per arm and per
round (`gpu_seconds` with the GPU type, `gen_tokens`, `attempts` / `actions`, `train_steps`, `train_tokens`,
`lean_checks`, labelled `arm`, `seed`, `labels.round`) as results-registry rows, exactly and at negligible
cost, so that later runs need not derive compute from job logs?

## Configuration

- **Brief:** [`BRIEF_compute-record.md`](../../docs/proposals/improvements/BRIEF_compute-record.md)
  (Dan, 2026-09-29). Budget $2 / 4 pod-hours.
- **Fork:** worktree `~/work/compute-record`, branch `dan_compute-record` of `dan-pandori/nd-takehome`, from
  `6b7b2f5e`. Main code commit `4c34eb85`; review fixes `0f0a4103`; executor tip `3fc57e41`, merged into the
  fork's `dan` (fast-forward). The fork's `dan` has since moved on (`d3abe474`, long-pool-2).
- **Pre-registration:** fork `preregistration/compute-record.md`, commit `34693ad2` (00:10:29 UTC), before
  the first pod (cr1, 00:19:26).
- **What was built.** `record.save_config()` opens a process-level compute block; `record.compute(...)`,
  `record.phase(...)` and `record.child(...)` open nested blocks. Each block writes, at exit, `gpu_seconds`
  (wall-clock × GPUs held, clock from process start; `labels.gpu`, or `labels.device = cpu`) and each
  non-zero counter through the existing `record()` path (`ND_OFFLINE`, `ND_REGISTRY` respected;
  `ND_COMPUTE=0` turns it off). Counters are filled by `sample.py` / `state_sample.py` (decoded tokens,
  attempts), `train.py` / `fast_train.py` / `state_train.py` (steps, non-pad tokens), and the Lean gate /
  judge (`lean_checks`, `labels.lean_s`). The RL drivers (`ladder_ei.py`, `state_ladder_ei.py`,
  `expert_iter.py`, `grpo.py`) split each round into `sample` / `eval` / `finetune` (GRPO: `update`) rows.
  Time is exclusive, so sums never double count. `registry_merge.py --compute` prints the per-arm table.
  Definitions are in the fork's `REGISTRY.md` and `CONTRIBUTING.md`.
- **GPU check:** one NVIDIA A40 (secure, $0.49/h billed), pods cr1 (00:19–00:41, parts 1–2) and cr2
  (00:54–00:57, part 3, after the review fixes); jobs `pod/cr/gpu_job{,2,3}.sh`, analysis `pod/cr/analyze.py`
  → `artifacts/compute-record/gpu/analysis.json`. Model behind every GPU number: `ckpts/cr/fast_s0.pt`,
  3,214,336 parameters, `lean_seq`, from scratch in this run, 6,000 steps at batch 500 on
  `data/lj/train_retain.jsonl` (3,000 records, seed 0). It is a work generator only; no accuracy is
  claimed. CPU checks: GitHub Actions `ubuntu-latest` (torch 2.8.0+cpu) and the 2-vCPU VPS.
- **Do not reuse** the throwaway GPU-check checkpoints `ckpts/ladder/cr_ladder2_r*.pt` and
  `ckpts/cr_grpo_r*.pt`: their EI / GRPO targets were `data/lj/heldout200.jsonl`, so they have seen
  held-out theorems (reviewer).
- **Run write-up:** `run_compute_record.md`, `numbers.md` § compute-record, `log.md`, `STATUS.md` on the fork
  branch. Per-arm table of this run: `artifacts/compute-record/compute_table.tsv`.
- **Review:** [`review_compute-record.md`](https://github.com/dan-pandori/nd-takehome/blob/53896f60/review_compute-record.md),
  §Recount in [`22025128`](https://github.com/dan-pandori/nd-takehome/commit/22025128), §Compare and §Verdict
  in [`53896f60`](https://github.com/dan-pandori/nd-takehome/commit/53896f60) (on `dan_compute-record`).
  Reviewer code: fork `review/compute-record-recount/`; the reviewer's own CPU checks ran in Actions run
  36654170471. The reviewer discloses that the executor's commit subjects were visible before phase 1.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/compute-record/` (`ckpts/`, `artifacts/`); registry rows under
  `registry/compute-record/`.
- **Compute:** cr1 0.36 h + cr2 0.05 h = 0.42 pod-hours (`podbudget`), A40 at $0.49/h: **$0.21** of $2.
  The GPU job took 16 min against the brief's ≤ 15 min (disclosed).

## Observations

All four pre-registered expectations are met; every value below was reproduced by the reviewer unless
marked otherwise.

- **Counters equal independent counts exactly, wherever they can be re-derived.** The executor reports
  20 / 20 checks equal. The reviewer re-derived independently:
  - `train_tokens` of seven training jobs, by replaying the batch draw: 375,609,000 (fast, 6,000 steps);
    37,560,900 each (600-step legacy × 2, fast, fast per-step); 4,540,890 and 4,563,097 (the two ladder2
    child fine-tunes, which the executor had checked only by step count). `train_steps` 6,000 / 600 / 300.
  - `attempts`: 6,400, 1,850, 1,536 per GRPO round, 450, 4,000. GRPO `train_steps` 3 and 1 (steps with
    no variance make no update).
  - `lean_checks` against the gate logs: 189 (ladder), 4,766 + 4,871 = 9,637 (ladder2).
  - `gen_tokens` in the reviewer's CPU test: exact on base, fast and fast-without-compaction decoding,
    including cut-off rows (2,285 ×3 at `max_new` 40; 4,644 at 200).
  - Not derivable from stored files: the pod's `gen_tokens` (the executor's `sample_check` 513,325 is its
    own count), GRPO `train_tokens` (4,778; 808) and GRPO `lean_checks` (199; 12).
- **`gpu_seconds` vs the job's own wall-clock:** the pre-registered 5 % criterion is **met on the one job
  ≥ 3 min** (6,000-step fast train, 372.5 / 376.8 s = 0.989), which ran **before** the clock was changed to
  start at process exec. The final clock can only raise the ratio, but it was not run on a ≥ 3-min job. The
  2-min legacy jobs give 0.975 / 0.974. Shorter jobs miss by a fixed 2.2–4.9 s: 0.961 (1 min), 0.955
  (1 min, per-step), 0.939 (80 s, 3 processes), 0.920 (27 s), 0.831 (17 s). The cause of that fixed gap is
  asserted by the executor (the row upload at exit), not measured.
- **Overhead < 1 %, as pre-registered.** The per-step counter in `train.py` costs 49.7 µs (executor, A40
  host) or 103 µs (reviewer, VPS CPU) against a ≈ 202 ms legacy step: 0.025–0.05 %. The legacy on/off A/B
  (n = 2 each, one host) also shows **+1.14 s per process** (0.89 % of a 2-min job): a fixed cost, not
  per step, which the write-up left out. The per-step counter on the fast (CUDA-graph) path, added after
  review, was **not** A/B'd on one host; the only comparison is across two pods (−2.7 % tok/s).
- **Per-arm table.** `registry_merge.compute_table` equals the reviewer's own sums on every (arm, seed,
  round) group. Two artefacts of this run's table: a **spurious `bench` row** (arm None, seed None, 200
  steps, 12,496,799 tokens) written by part 1's benchmark, which is in `compute_table.tsv` and the bucket
  and is **not work**; and Stage-1 `train.py` jobs carry no arm unless given one, so the 6,000-step job and
  the two legacy A/B jobs of seed 0 merge into one row (7,200 steps).
- **CI green:** GitHub Actions on `dan` at `3fc57e41` (runs 36653017895, and 36651698262 on
  `ci-compute-record` with all compute checks); `tests/test_registry.py` passes on the VPS.
- **Hard constraints clean:** `nd_verify` tree `9437bb72` unchanged and not used as a judge;
  `artifacts/TEST_RUN_DONE` blob unchanged; no evaluation file read by training code (the GPU-check jobs
  used `heldout200` as targets for throwaway checkpoints only, see Configuration).

### Rewordings applied here (reviewer's items 1–6)

1. The fast path does **not** count "once per run": since part 3 `fast_train.py` counts per step, and that
   overhead is unmeasured. `REGISTRY.md` repeats the stale sentence, and also says start-up before
   `save_config` is not counted, which the process-age clock made false.
2. The `gpu_seconds` criterion: "met on the one job ≥ 3 min (6 min, 0.989), run before the clock change".
3. Signals: **SIGTERM** (a plain `kill`, `timeout`, a pod deleted by the budget watch) loses the open blocks:
   the current round's phase row and the process row, i.e. **everything** for a single-process `train.py`.
   SIGINT and uncaught exceptions keep all rows; exceptions leave phase rows marked `status: ok`.
   (`run_compute_record.md` says "a process killed by a signal writes no rows".)
4. `log.md`'s `git_sha` sentence is wrong: all part-1/2 rows say `4c34eb85`. Part 1 is labelled correctly;
   part 2 ran the process-age code (`9f970a9f`) but is **mislabelled** `4c34eb85`, because the pod's
   `.git_sha` file went stale when the code was re-synced without `.git`. Part 3's rows say `7fd4ddb1`.
5. The `bench` row in `compute_table.tsv` / `numbers.md` is not work (above).
6. `lean_check.check` counts duplicate texts (`REGISTRY.md` says "distinct"); under
   `ND_SAMPLE_EARLY=exact|goal`, `gen_tokens` excludes the injected tokens (3,881 counted vs 3,898 in `raw`).
   The default (`eos`) is exact.

## Limitations and next step

- **`gpu_seconds` is occupancy, not billed GPU time or utilisation:** with ≈ 5 CPU-bound jobs per card,
  Σ `gpu_seconds` is ≈ 5× the billed GPU time. A comparison between arms that shared cards differently
  needs billed GPU-hours as well. It includes CPU-side Lean time while the process holds CUDA.
- **Not instrumented:** `run_vllm.py` (no `gen_tokens` / `attempts`; `n_gpu` stays 1 under tensor
  parallelism unless `ND_N_GPU` is set; no row at all if the main process never initialises CUDA), and
  analysis scripts that sample without `save_config` (`peek_samples.py`, `lp_b.py`, `dsg_regress.py`).
  Validation and prefill tokens are not counted.
- **Before later runs rely on the rows** (reviewer): add a SIGTERM handler in `job_compute` (turn SIGTERM
  into `SystemExit` so `atexit` runs), since the pod runners stop jobs with `kill` and budget-watch deletions
  would drop the current round's compute. Until then, runs that may be killed should cross-check against job
  logs.
- **Next measurement** (reviewer; one pod, ≈ 15 min): (a) `fast_train` 600 steps with `ND_COMPUTE=0` vs on,
  alternated on one host; (b) one ≥ 3-min job on the final clock; (c) `kill -TERM` of `ladder_ei.py`
  mid-round, before and after a SIGTERM handler, counting rows; (d) GRPO with completions stored, to recount
  `train_tokens` and `lean_checks`.
