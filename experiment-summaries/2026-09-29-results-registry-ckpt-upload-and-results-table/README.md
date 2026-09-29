---
written_on: 2026-09-29
written_by: agent:claude
---

# results-registry: one results table, and checkpoints in the bucket before the script moves on

Status: **completed 2026-09-29 00:20–00:56 UTC (executor) and independently reviewed 01:00–01:12 UTC.
Verdict: stands, with five items to fix or reword before the registry is used as a source of truth.**
Infrastructure run; it makes no model-quality claim. No hard-constraint violation, nothing quarantined.
Where the reviewer's wording differs from the executor's, this summary uses the reviewer's.

Question (proposal 15 §5): can one structured registry replace "numbers in `numbers.md` prose, with checkpoints
uploaded by hand at the end"? Concretely: (i) does every checkpoint a script saves reach the bucket before the
script moves on, with an md5 that the downloaded copy matches? (ii) Can a reader answer "every held-out accuracy
of the control checkpoints, by run" from one merged table, without reading prose? (iii) Does a backfill from the
reviewed runs' summary files reproduce their headline numbers exactly? The motivation is that checkpoints were
lost or never uploaded in three earlier runs, and results were spread in prose across 27 branches.

The run was on the fork (`dan-pandori/nd-takehome`), branch `dan_results-registry`. **Unlike most runs, its code
is merged into the fork's `dan`** (`origin/dan` = `3bfdec15`), as its brief authorised, so it is in force for every
run that branches from the fork's `dan` from now on.

- Review: [`review_results-registry.md`](https://github.com/dan-pandori/nd-takehome/blob/9a38b9a4/review_results-registry.md)
  ([`293f6d00`](https://github.com/dan-pandori/nd-takehome/commit/293f6d00) §Recount, blind;
  [`9a38b9a4`](https://github.com/dan-pandori/nd-takehome/commit/9a38b9a4) §Compare/§Verdict; reviewer scripts
  `review_rr_*.py`).
- Executor write-up: [`run_results_registry.md`](https://github.com/dan-pandori/nd-takehome/blob/3bfdec15/run_results_registry.md),
  `numbers.md` § results-registry, [`REGISTRY.md`](https://github.com/dan-pandori/nd-takehome/blob/3bfdec15/REGISTRY.md)
  (usage and schema), `log.md`, `artifacts/results-registry/acceptance.json`. Results commit `05fcf75e`.
- Pre-registration: `preregistration/results-registry.md`, commit `454528da` at 00:23:07 UTC, before any code
  and before the pod (`rr-1`, created 00:34:32 UTC). Its text says "written ≈ 00:45"; the commit time is the
  one that counts (see Limitations on the log's clock times).

## Configuration

- **What was built.**
  - `model.save_ckpt` uploads to `hf://buckets/dan-pandori/nd-rl/<ND_RUN_ID>/ckpts/…` before it returns. It checks
    the remote size, writes an md5 sidecar `<ckpt>.upload.json` and a `ckpt_saved` row, and **raises** if
    `ND_RUN_ID` is unset or the upload fails. The opt-out is `ND_OFFLINE=1` (the pre-registration named it
    `ND_CKPT_OFFLINE`).
  - `record.preflight()` stops `train.py`, `state_train.py`, `grpo.py` and the EI drivers before the first step
    if `ND_RUN_ID` or the `hf` CLI is missing.
  - `record.py` writes one row per headline number: run, arm, seed, role, git SHA, config, data file + md5,
    checkpoint md5 + URI, metric, value, n, source file. Rows go to one file per process under
    `artifacts/<run>/registry/`, synced to `hf://…/registry/<run>/`. This is a deviation from the brief's single
    file, because a pull from a second pod overwrites a single file.
  - Instrumented: `train`, `fast_train`, `eval_set`, `sd_eval`, `coverage`, `expert_iter`, `ladder_ei`,
    `state_ladder_ei`, `grpo`, `state_eval`, `state_train`.
  - `registry_merge.py` merges and queries the rows; `registry_backfill.py` backfills from earlier runs'
    summary and round files.
- **Smoke models (throwaways; no model-quality claim).**
  - `ckpts/rr/smoke_s0.pt`: 3,214,336 params, `lean_seq`, from scratch, 300 steps on
    `data/p2/train_depth3_f0_a1.jsonl`, seed 0.
  - `ckpts/rr/ei_fast1_s0_r{1,2}.pt`: two 30-step EI fine-tunes of fast-stage1's `ckpts/fs/fast1_s0.pt`
    (3.2 M, `lean_seq`, from scratch, 6,000 steps, same training set). This third EI run was not in the
    pre-registration. The two pre-registered EI smokes from `smoke_s0` accepted 0 proofs and saved nothing.
- **Judge.** Lean alone for anything counted. `nd_verify` is unmodified and used only as a checker *label* in the
  backfill.
- **Pod.** `rr-1`, NVIDIA GeForce RTX 3090, $0.50/h billed, 0.21 pod-hours, **$0.10** (`podhours.log`), against a
  budget of $1 / 2 pod-hours.
- **Bucket.**
  - `hf://buckets/dan-pandori/nd-rl/registry/<run>/` holds the rows, and `…/registry_merged/` the merged table
    (`.jsonl.gz` and `.csv.gz`).
  - `…/results-registry/ckpts/rr/` holds the 5 smoke checkpoints, and `…/results-registry/artifacts/{rr,results-registry}/`
    the smoke and acceptance artifacts.
- **Acceptance.** Run `python3 registry_acceptance.py`; it writes `artifacts/results-registry/acceptance.json`.

## Observations (reviewed)

Each pre-registered check was re-derived with the reviewer's own code.

| check (pre-registered) | expected | measured (reviewer) |
|---|---|---|
| E1 checkpoint URI downloads to the recorded md5 | 100 %, ≥ 3 | **5 / 5** (final, step 150, state 150, EI r1, EI r2) |
| E2 negative controls must fail: flipped byte; no `ND_RUN_ID`; missing bucket | 3 / 3 fail | **3 / 3 fail**. The missing bucket raises after 3 tries and writes no sidecar |
| E3 held-out greedy of control checkpoints "by run", one filter | ≥ 8 runs | **25 runs / 3,091 rows** with the implemented roles `{stage1, init, frozen}`. With the pre-registered `{stage1, control, frozen}` it is 21 runs / 1,964 rows. The pass stands either way, **but the content is partly wrong** (below) |
| E4 three headline numbers string-equal from the backfill | 3 / 3 | **3 / 3**: lean-format `token_full` 0.948; ds-generator frozen C0 158 / 114 of 2,285; state-env S s0 T1 1,348 (a substitute named before the check) |
| E5 cost | < 5 ms/row; < 30 s/ckpt | **0.17 ms/row** median on the VPS (the executor measured 0.058). Uploads took 3.4–7.4 s per 12.9 MB checkpoint and 6.0 s for the 38.6 MB state file (from the sidecars, on the pod) |

- **Rows.** 299,006 = 63 live + 298,943 backfilled, with 0 exact duplicates, in 39 files. The bucket alone gives
  the same table. The backfill fetched 32 runs, **31 of which have rows**; podjob has no result files. The
  executor said "32 reviewed runs".
- **lean-prefilter's 0 false rejects in 1,310,119** is recoverable after all. It is the sum of three rows,
  1,214,162 + 76,281 + 19,676, and every `false_rej` row is 0. The executor's substitute was allowed but was not
  needed.
- **Lean re-check.** The smoke jobs produced 22 counted proofs: 4 held-out greedy proofs from `smoke_s0` (4 / 500)
  and 18 found in the EI runs. The reviewer's Lean harness accepts all 22 and rejects a negative control. These
  are ND renderings, because `eval_set.py` / `expert_iter.py` store no literal `lean_seq` text; that gap
  pre-dates this run.
- **Splits.** Training and every smoke evaluation subset share 0 renaming classes.

**Defects in the table's content** (reviewer; to be fixed before the table is trusted):

1. **Every GRPO row in `run4-grpo` is labelled as the control checkpoint, but it measures the GRPO-trained
   model.** 2,080 rows over 41 arms carry `role=init` and `ckpt=` the Stage-1 init. 260 of them fall in the E3
   answer (8.4 % of its rows), with values 0.573–0.917. The cause is that old `grpo.py` wrote the init path
   into every `round_<r>.json`. The same checkpoint measured by the EI arm's round 1 gives 0.8806; the GRPO rows
   run 0.6926 … 0.6008 over rounds 1–8. Live `grpo.py` is correct, so only the backfill is wrong.
2. **Checker labels follow the digest date, not the measurement date.** All 49,160 `run4-grpo` rows are
   labelled `lean`, but they were measured on 2026-09-17/18. Runs from before `lean_gate.py` (2026-09-21) were
   judged in the loop by `nd_verify`, so `lean+nd_verify` is also not literally how they counted.
3. **Inherited and superseded arms are counted under several runs.** lean-format's arms appear under
   ds-composition, ds-generator and lean-seed2. novelty-campaign's appear under ds-composition. ds-composition's
   `dsc_pre_cleanup/` and `dsc/` both yield `dsc/ei_a1_s0`, which gives 75 groups with two different values.
   A "by run" answer therefore double-counts.
4. **The E3 query mixes held-out sets.** It pools `data/heldout.jsonl`, `data/p2/heldout.jsonl`, 240 rows with
   `data=None`, and the smoke's 200-theorem subset under one metric name. For the smoke rows this is a
   code-version artifact: code at HEAD would name the subset's metric separately.
5. **Model labels.** Eval rows carry `ckpt` + `ckpt_md5` but no `n_params` or format, so a reader must join on
   `ckpt_md5`. For EI / GRPO rows, `seed` is the driver's seed, not necessarily the model's training seed.
   `REGISTRY.md` says otherwise.

## Interpretation

The checkpoint half is sound and already in force. `save_ckpt` uploads, verifies and raises, and all three
negative controls fail as they should. That closes the lost-checkpoint failure seen three times before.

The results-table half works mechanically: the query is one filter, the backfill reproduces headlines exactly,
and it is cheap. It is **not yet a clean source of truth**. Until items 1–3 are fixed, a query for control
accuracies returns some RL-trained numbers, and by-run counts double-count.

## Limitations and next step

- **Clock times.** The times in the executor's `log.md` / STATUS are wrong by 20–50 min and in places out of
  order. The actual window is 00:20–00:56 UTC. The ordering that matters holds on commit times.
- Backfilled rows have no checkpoint md5. `summary:` rows are only as well labelled as each run's summary file.
- The `save_ckpt` torch path was exercised on the pod, but the reviewer's VPS has no torch, so the reviewer
  skipped those test cases. The reviewer's next step: run `tests/test_registry.py --online` on a pod.
- **Runs that branched from the fork's `dan` before this merge** keep the old `save_ckpt` until they merge `dan`
  (examples: state-frontier, support-state, long-pool). After they merge, a pod job without `ND_RUN_ID` stops at
  start-up with an explicit message.
- **Next.** Fix items 1–3 in `registry_backfill.py`:
  - GRPO round rows get `role=rl` and no `ckpt`;
  - checkers are labelled by measurement date;
  - rows get `labels.inherited_from` and a superseded tag.

  Then document item 4 and item 5 in `REGISTRY.md`, and re-run the E3 query. It should contain no arm under two
  run ids, and no value that differs across rounds for a fixed (arm, ckpt, data) unless `role=frozen`.
  `review_rr_findings.py` prints both counts; today they are 269 rows and 9 run pairs.
- **Policy.** The executor proposed `AGENT_POLICY.md` text in the fork's `QUESTIONS.md` (01:40 UTC by its log
  clock), under which librarians query the registry instead of reading prose. The reviewer says that use needs the
  fixes above first. The default if Dan does not answer: no change to `AGENT_POLICY.md`.
