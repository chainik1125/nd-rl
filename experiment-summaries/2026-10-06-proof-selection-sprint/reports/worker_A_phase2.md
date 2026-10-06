---
written_on: 2026-10-06
written_by: agent:claude (Worker A)
status: complete
---

# Worker A phase 2: does more hard-negative data make the learned head rank finished proofs?

Pre-registration: `logs/worker_A.md`, 03:38 UTC (amended 04:19, before any training: the early-stopping dev set is
the original + new dev pools for every rung). Phase-2 freeze: `results/phase2_test_freeze_sha256.json`
(frozen_utc 2026-10-06T04:50:09Z). Test evaluated once after it. Code: `code/train_verifier2.py`, `code/phase2_report.py`,
`code/phase2_within_ci.py`, `code/merge_parts.py`.

## Bottom line

- **More hard negatives did not close the gap to likelihood.** The phase-2 ranker was trained only on finished
  attempts, with a within-theorem pairwise loss and 3.3x the hard negatives (10,676 finished-invalid training
  attempts against 3,195). Its within-theorem AUROC among finished attempts is still **below the model's own
  likelihood** (`loglik_mean_masked`):
  - holdout250: 0.677 vs 0.788, a gap of **−0.111 [−0.177, −0.040]**
  - textbook72: 0.638 vs 0.687, a gap of **−0.049 [−0.196, +0.089]** (34 theorems; unresolved)
  - dev: −0.111 [−0.161, −0.060]
- **Primary selection result, k=8.** `completion_verifier2_ens` vs `completion_loglik`:
  - holdout250: 0.913 vs 0.918, paired difference completion_loglik − verifier2 = **+0.005 [−0.006, +0.016]** (unresolved)
  - textbook72: 0.514 vs 0.540, paired difference **+0.026 [+0.006, +0.049]**, so likelihood is better
  - The learned head does not beat likelihood on either set.
- **What the ladder shows.**
  - From 799 to 10,676 finished-invalid examples, test within-theorem AUROC rises a little: holdout250 0.629 → 0.677,
    textbook72 0.545 → 0.638. Dev stays flat (0.642 → 0.665).
  - Hard-negative scarcity explains at most part of the phase-1 gap. In the range tested, the curve is far below
    likelihood and is not converging toward it at any visible rate.
  - Phase-2 ranking did improve on the phase-1 BCE head on dev (+0.078 [+0.024, +0.129]); on test that improvement is
    not resolved (h250 +0.031 [−0.025, +0.096]).
- **Undertraining is unlikely in this recipe.**
  - The top-rung dev curve plateaus at 0.63–0.68 from step 100 to step 1300 (about 14 passes over the 1,492 pair
    theorems), with its best checkpoints at steps 400–500 (`figures/A2_top_rung_dev_curve.png`).
  - I did not test other learning rates or a larger model, so "the recipe cannot learn it" is not excluded.
- **Giving the head likelihood as an input feature (`verifier2_ll`) makes it worse than likelihood alone.**
  - Within-theorem AUROC: dev 0.682 vs 0.776; holdout250 0.718 vs 0.788 (gap −0.070 [−0.133, −0.010]);
    textbook72 0.698 vs 0.687 (+0.011 [−0.102, +0.125]).
  - The learned part adds noise that outweighs any signal it contributes on top of likelihood.

## Ladder: within-theorem finished-only AUROC vs finished-invalid training examples

| model | finished-invalid train examples (in pair theorems) | pair theorems | best step | within-thm finished AUROC: dev (orig pools) | within-thm finished AUROC: holdout250 | within-thm finished AUROC: textbook72 |
|---|---|---|---|---|---|---|
| verifier2_r25 | 799 (703) | 455 | 100 | 0.642 | 0.629 | 0.545 |
| verifier2_r50 | 1598 (1423) | 756 | 100 | 0.633 | 0.635 | 0.523 |
| verifier2_r100 | 3195 (2862) | 1227 | 300 | 0.657 | 0.664 | 0.568 |
| verifier2_rall_s0 | 10676 (9499) | 1492 | 400 | 0.627 | 0.674 | 0.662 |
| verifier2_rall_s1 | 10676 (9499) | 1492 | 500 | 0.662 | 0.683 | 0.642 |
| verifier2_rall_s2 | 10676 (9499) | 1492 | 400 | 0.657 | 0.654 | 0.596 |
| verifier2_ens (mean of rall s0-s2) | 10676 (9499) | 1492 |  | 0.665 | 0.677 | 0.638 |
| verifier2_ll (rall + loglik feature) | 10676 (9499) | 1492 | 1400 | 0.682 | 0.718 | 0.698 |
| *loglik_mean_masked* (reference) | | | | 0.776 | 0.788 | 0.687 |
| *verifier_ens* (reference) | | | | 0.587 | 0.646 | 0.596 |
| *verifier_hist_ens* (reference) | | | | 0.622 | 0.647 | 0.622 |

n theorems with both classes among finished attempts: dev (orig pools) 185, holdout250 116, textbook72 34

All rungs use one seed except the top rung, which has 3 seeds.

- **Data per rung.**
  - Rungs r25 / r50 / r100 keep a nested random subsample of the original pools' finished-invalid train attempts and
    all finished-valid ones.
  - The top rung adds the new pools:
    - rr600 k=96b: 600 theorems, seeds 1–2
    - rl_targets k=48b: the 751 train theorems with per-attempt validity < 0.9, seed 3
    - dev1108 k=48b: 419 low-validity theorems, seed 4
  - Train/dev membership is unchanged from `splits.json`.
- **Train-split numbers corrected per the midpoint audit.** The original training data is 97,776 attempts, 86,217
  valid (88.2%). It contains 3,195 finished-invalid attempts; 1,227 theorems have both a finished-valid and a
  finished-invalid attempt.

Figure `figures/A2_auroc_vs_data.png`: x is finished-invalid training attempts (log scale). y is within-theorem
AUROC on dev and on each test set. The dashed lines are the `loglik_mean_masked` baseline for each set, and the
crosses are individual top-rung seeds.

## Paired within-theorem AUROC differences (theorem bootstrap, B=1000; `results/phase2_within_ci.json`)

| comparison | dev (185 thm) | holdout250 (116 thm) | textbook72 (34 thm) |
|---|---|---|---|
| verifier2_ens − loglik_mean_masked (**primary a**) | −0.111 [−0.161, −0.060] | −0.111 [−0.177, −0.040] | −0.049 [−0.196, +0.089] |
| verifier2_rall_s0 / s1 / s2 − loglik | −0.149 / −0.114 / −0.118 | −0.114 / −0.105 / −0.134 | −0.025 / −0.045 / −0.092 |
| verifier2_ll − loglik_mean_masked | −0.094 [−0.139, −0.043] | −0.070 [−0.133, −0.010] | +0.011 [−0.102, +0.125] |
| verifier2_ens − verifier_ens (phase 1) | +0.078 [+0.024, +0.129] | +0.031 [−0.025, +0.096] | +0.042 [−0.083, +0.137] |

Only theorems with both valid and invalid finished attempts count.

## Selected-proof accuracy (all theorems; `results/selection_{holdout250,textbook72}_p2.json`)

| scorer | holdout250 k=8 | holdout250 k=64 | textbook72 k=8 | textbook72 k=64 |
|---|---|---|---|---|
| random pick (expectation) | 0.836 | 0.836 | 0.405 | 0.407 |
| completion | 0.896 | 0.904 | 0.519 | 0.556 |
| completion_loglik | **0.918** [0.886, 0.946] | **0.936** | **0.540** [0.441, 0.643] | **0.625** |
| shallow_lr | 0.913 | 0.924 | 0.531 | 0.583 |
| completion_verifier_ens (phase 1) | 0.906 | 0.924 | 0.514 | 0.542 |
| **completion_verifier2_ens** (primary b) | 0.913 [0.881, 0.939] | 0.924 | 0.514 [0.417, 0.615] | 0.542 |
| completion_verifier2_ll | 0.918 [0.888, 0.946] | 0.932 | 0.531 [0.432, 0.634] | 0.597 |
| oracle pass@m | 0.940 | 0.960 | 0.594 | 0.694 |

Paired differences at k=8:

| comparison | holdout250 | textbook72 |
|---|---|---|
| completion_loglik − completion_verifier2_ens | +0.0049 [−0.0055, +0.0155] | +0.0256 [+0.0055, +0.0488] |
| completion_loglik − completion_verifier2_ll | −0.0007 [−0.0116, +0.0096] | +0.0092 [−0.0099, +0.0274] |

The raw `verifier2_*` scores are not usable as selectors on their own (holdout250 k=8 0.785). They were trained only
on finished attempts and rank unfinished ones arbitrarily. Selection uses the pre-registered composite (finished
first, then verifier2 score).

## Checks

- **Pools.** All three extra pools replay byte-for-byte (0 mismatches in 113,760 attempts). Worker B's unmodified
  `score_loglik.py` on the same pools: 0 replay, sha or nd mismatches. `pools_v2/MANIFEST.json` was updated with a
  phase field.
- **Label-permutation tests.** Labels and reasons were shuffled and the scorers re-run in score-only mode:
  - final phase-1 checkpoints (verifier_s0–s2, verifier_hist_s0–s2, both ensembles; dev1108 and textbook72): 16/16
    score files byte-identical
  - all 7 verifier2 models (dev1108): 7/7 byte-identical
  - The textbook72 test printed only identity flags; no metric was computed.
- **Exposure.**
  - No phase-2 choice used anything from the test pools: low-validity theorem selection, ladder, loss, input format,
    early stopping, the likelihood feature.
  - Test LABELS were never used for a choice in either phase.
  - Phase 1's 02:09 input-format decision did use textbook72's label-free clipping rate.
  - Test-set likelihood numbers were already known from the phase-1 read.

## Limitations

1. **One seed per lower rung, and the ladder is non-monotone on dev.** Seed spread at the top rung is about ±0.02–0.04,
   about the size of the ladder steps, so the slope is uncertain.
2. **The new data is a different theorem mix.** It is concentrated on low-validity and long-proof (rr600) theorems,
   so "more data" is confounded with "harder theorems".
3. **textbook72 has only 34 theorems** with both finished classes; its AUROC intervals are about ±0.12.
4. **One training recipe** (lr 1e-4 / 1e-3, 16 theorems × ≤8+8 attempts per step, r16 body fine-tuned). Other
   learning rates, a frozen body, or a larger model were not tried. "Undertraining" is excluded only for this recipe.
5. **OOM reruns.** r50, r100 and ll first crashed with CUDA OOM: a process I do not own (pid 2868009 / 2879135, likely
   the sibling worker) took 86–127 GB on GPUs 5/6. I reran them with identical settings; the crashed attempts had
   written no checkpoint.
6. The likelihood baseline and the `verifier2_ll` feature both come from Worker B's masked teacher-forced likelihood.
   They are not independent of B's masking choices.
