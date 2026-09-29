# Run brief: long-pool-2 — a pool past `L_true` 17, so the best model's frontier can be read

Run id `long-pool-2`. Repository `~/work/long-pool-2` (branch `dan_long-pool-2` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Budget **$10**, ceiling **20 pod-hours**
(mostly CPU). Authorised by Dan 2026-09-29 (proposal 16 item 0).

Read first:
- nd-rl `docs/proposals/2026-09-29-literature-shortlist.md` §0.
- The `long-pool` and `state-cap12` summaries in `~/nd-rl/experiment-summaries/`.
- `data/ladder/POOLS.md` (the `long-pool` section).
- `lpool_stages.py`, `lpool_assemble.py`, `lpool_finalize.py`, `minlen.py`, `gen.py`.

**Code not on `dan`:** `lpool_reread.py` and `state-cap12`'s pod scripts (`pod/sc12/`, including `sn_seed.sh`, the SN-cap12 Stage-1 recipe) are only on `origin/dan_state-cap12` (`lpool_reread.py` also on `origin/dan_long-pool`). Take them with `git checkout origin/dan_state-cap12 -- lpool_reread.py pod/sc12/` and merge them with this run.

## Why

SN-cap12 T1 (the state policy trained on K12's cap-12 proofs, plus ladder EI; 3,216,384 parameters,
`lean_staten`) solves 47–91 % of `transfer_long_rr600` at `L_true` 13–16 and 36–66 % of
`transfer_long_ge17.jsonl`. Its `L*` is ≥ 17, the pool's ceiling, on all 4 seeds. The ≥ 17 file has no
per-theorem length. Until a longer pool exists, no frontier experiment on the best model can be read.

## What to build

**`data/ladder/transfer_long2.jsonl`**, ≥ 100 theorems per bin, same generator family as `long-pool`.
- **Labels are intervals.** Exact `minlen` labels above 16 are exponential.
  - *Lower bound 17:* `minlen` completes its bound-16 search without finding a proof. A timeout is
    "unknown", never a label: exclude it, and report the timeout rate per bin.
  - *Upper bound:* the generator's construction length after pruning.
- **Bins** by upper bound: 17–18, 19–20, 21–22, 23–24. If a bin cannot reach 100 at this budget, report
  the count reached and why.
- **Calibration bin:** carry `transfer_long_ge17.jsonl`'s theorems into a separate calibration file.
- **Disjoint by renaming class** from every training set in use (the cap-6 control set, K8–K14), from
  `rl_targets.jsonl`, and from every existing transfer and held-out pool. Report the disjointness counts.
- **A `POOLS.md` section** as for `long-pool`: settings, per-bin counts, shape table, what the labels
  certify.

## Re-read (no training)

Settings as in `state-cap12`'s long-pool read-out: `lpool_reread.py`, k 256, T 0.8, seed 0, and **`max_steps`
96** for state models on the new pool. Checkpoints are in `hf://buckets/dan-pandori/nd-rl/state-cap12/ckpts/`
and `…/long-pool/`.
- **Models on the new pool:**
  - SN-cap12 T1 s0–s3 and SN-cap12 frozen s0–s3;
  - K12 whole-proof T1 s0 / s1;
  - SN-v2 cap-6 T1 s0 / s1.
- **On `rr600` and the ≥ 17 file:** SN-cap12 T1 s2–s3 at `max_steps` 96. This is `state-cap12`'s open
  step-cap question.
- **Report** per-bin solve rates by model, and `L*` defined on the upper-bound bins. Say plainly that
  under interval labels, `L*` is a statement about the upper bound.

## Pre-registration (before the first pod)

Expected per-bin counts and generation budget. For the re-read: SN-cap12 T1 solves ≥ 10 % of the 17–18
bin and < 10 % of the 23–24 bin. Falsifier: ≥ 10 % in every bin, which means the pool must go longer.

## Deliverables

- The pool and its `POOLS.md` section, merged into the fork's `dan` (data later runs use), and uploaded
  to the bucket.
- `preregistration/long-pool-2.md`, `run_long_pool_2.md` (≤ 300 words + figure: per-bin solve rate by
  model), `numbers.md` § long-pool-2, `log.md`, and `STATUS.md` line `LONG-POOL-2 DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/long-pool-2/{data,artifacts}`.
- Pods deleted; `touch ~/runs/long-pool-2/executor.done`.

Never run `test_run_once.sh`.
