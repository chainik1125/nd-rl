---
written_on: 2026-10-06
written_by: agent:claude (Worker A)
status: complete
---

# Worker A phase 4: a trained judge that keeps the per-token likelihood

Pre-registration: `logs/worker_A.md`, 05:28 UTC. Phase-4 freeze: `results/phase4_test_freeze_sha256.json`
(07:01:29 UTC, 308 files). The test set was read once after the freeze; all files were re-hashed afterwards and none
had changed.

Code: `code/train_judge.py` (training), `code/phase4_report.py` (scoring with Worker B's unmodified
`score_loglik.py --ckpt`, publishing, analysis).

## Bottom line

Training the sibling generator s1 as a likelihood judge on s0's training-split pools beats both plain s1 likelihood
and the sibling-likelihood mean. The measure is within-theorem AUROC among finished attempts on s0's test pools.
**This is the first learned scorer in the sprint that beats likelihood.**

**Pairwise judge (`judge_pair`, 2-seed mean)**, gain on each s0 pool:

| pool | vs plain s1 likelihood | vs sibling mean |
|---|---|---|
| holdout250 | +0.065 [+0.036, +0.094] | +0.064 [+0.033, +0.095] |
| textbook72 | +0.159 [+0.088, +0.236] | +0.159 [+0.084, +0.237] |
| second s0 draw, holdout250 | +0.075 [+0.048, +0.103] | +0.059 [+0.035, +0.085] |
| second s0 draw, textbook72 | +0.094 [+0.037, +0.158] | +0.121 [+0.058, +0.189] |

Absolute values: holdout250 0.893 vs 0.828, textbook72 0.942 vs 0.783.

**SFT judge (`judge_sft`)** is smaller but positive:
- holdout250 +0.037 [+0.012, +0.061], textbook72 +0.081 [+0.030, +0.132] over plain s1.
- On the second draw: holdout250 +0.069 [+0.045, +0.095]; textbook72 +0.019 [−0.028, +0.065], unresolved.
- Pairwise beats SFT on textbook72 (+0.078 [+0.018, +0.145]); on holdout250 the gap is +0.028 [−0.004, +0.061],
  unresolved.

**Transfer to other generators' proofs.** A judge that had only learned s0's error habits should not improve on
s1/s2 proofs. It does improve, so it is not just s0-specific:
- Pairwise judge vs sibling-likelihood mean, holdout250 / textbook72:
  - s1-generated pools: +0.081 [+0.044, +0.122] / +0.071 [+0.008, +0.133]
  - s2-generated pools: +0.026 [+0.002, +0.056] / +0.083 [+0.045, +0.125]
- The SFT judge does not beat the sibling-likelihood mean on s1/s2 pools (−0.014 to +0.032, all unresolved).
- Caution: s1-generated pools are the judge's *own* init generator, so the comparison with plain s1 there
  (+0.130 / +0.165) is a self-judge comparison.

**Selection at k=8 (completion first).** The gains are small because plain s1 likelihood already sits close to the
oracle.
- Pairwise judge vs `completion_loglik_s1`, s0 test pools:
  - holdout250: 0.933 vs 0.931, +0.002 [−0.002, +0.006]
  - textbook72: 0.582 vs 0.568, +0.014 [+0.001, +0.031]
  - Oracle 0.940 / 0.594, random 0.836 / 0.405.
- Gap recovered at k=8: about 93% (holdout250) and 94% (textbook72). For comparison: completion_loglik (s0 self)
  79% / 71%; phase-1 verifier_ens 68% / 58%.
- Against `completion_loglik_siblings`: +0.005 [−0.001, +0.012] on holdout250; +0.015 [−0.001, +0.033] on textbook72.

## Within-theorem finished-only AUROC (`results/phase4_within.json`, figure `figures/A4_trained_judge.png`)

| test pool (generator) | n thm | plain s1 (start) | sibling mean | self judge | judge_sft (s0 / s1 seeds) | judge_pair (s0 / s1 seeds) |
|---|---|---|---|---|---|---|
| holdout250 (s0) | 116 | 0.828 | 0.829 | 0.788 | **0.865** (0.871 / 0.860) | **0.893** (0.890 / 0.893) |
| textbook72 (s0) | 34 | 0.783 | 0.782 | 0.687 | **0.864** (0.863 / 0.859) | **0.942** (0.938 / 0.941) |
| holdout250 (s0, 2nd draw k64b) | 134 | 0.832 | 0.848 | 0.813 | **0.901** | **0.907** |
| textbook72 (s0, 2nd draw k64b) | 32 | 0.843 | 0.817 | 0.764 | 0.863 | **0.938** |
| holdout250 (s1) | 135 | 0.775 (self) | 0.824 | 0.775 | 0.815 | **0.905** |
| textbook72 (s1) | 41 | 0.758 (self) | 0.852 | 0.758 | 0.838 | **0.923** |
| holdout250 (s2) | 132 | 0.842 | 0.856 | 0.807 | 0.856 | **0.883** |
| textbook72 (s2) | 42 | 0.831 | 0.841 | 0.771 | 0.873 | **0.924** |

Notes on the table:
- "Sibling mean" is the mean of the two likelihood judges that did not generate the pool (Worker B's definition;
  `loglik_mean_masked_siblings` on s0 k64).
- Ensemble values are the mean of the two seeds' masked mean log p.
- 95% theorem-bootstrap intervals for every paired difference are in `phase4_within.json`.

## k=8 selected-proof accuracy (all theorems; `results/selection_<set>_<gen>_k<k>_p4.json`)

| pool | random | oracle | completion_loglik (self) | completion_loglik_s1 | sibling mean | completion_judge_sft | completion_judge_pair |
|---|---|---|---|---|---|---|---|
| holdout250 s0 | 0.836 | 0.940 | 0.918 | 0.931 | 0.928 | 0.933 | 0.933 |
| textbook72 s0 | 0.405 | 0.594 | 0.540 | 0.568 | 0.567 | 0.570 | **0.582** |
| holdout250 s0 k64b | 0.831 | 0.938 | 0.915 | 0.927 | 0.927 | 0.932 | 0.929 |
| textbook72 s0 k64b | 0.404 | 0.589 | 0.544 | 0.561 | 0.556 | 0.565 | 0.576 |
| holdout250 s1 | 0.831 | 0.955 | 0.921 (=s1) | 0.921 | 0.934 | 0.937 | 0.941 |
| textbook72 s1 | 0.434 | 0.579 | 0.544 (=s1) | 0.544 | 0.562 | 0.562 | 0.562 |
| holdout250 s2 | 0.818 | 0.935 | 0.909 | 0.916 | 0.923 | 0.921 | 0.925 |
| textbook72 s2 | 0.439 | 0.648 | 0.584 | 0.606 | 0.612 | 0.619 | 0.634 |

Paired differences at k=8, completion_judge_pair minus each likelihood baseline:

| pool | − completion_loglik_s1 | − sibling mean |
|---|---|---|
| holdout250 s0 | +0.002 [−0.002, +0.006] | +0.005 [−0.001, +0.012] |
| textbook72 s0 | +0.014 [+0.001, +0.031] | +0.015 [−0.001, +0.033] |
| holdout250 s1 | +0.020 [+0.009, +0.034] | +0.006 [−0.003, +0.017] |
| holdout250 s2 | +0.010 [+0.003, +0.019] | +0.002 [−0.001, +0.006] |
| textbook72 s2 | +0.027 [+0.013, +0.045] | +0.022 [+0.006, +0.043] |

All other pairs are in the per-pool json files.

## Training, anchor strength and drift

| run | best dev within-thm AUROC (s1 start 0.826, 259 dev theorems) | best step | dev masked log p valid / invalid (start −0.050 / −0.090) | token KL(s1 ‖ judge) |
|---|---|---|---|---|
| sft_s0 | 0.887 | 3000 (= last step; may be undertrained) | −0.026 / −0.061 | n/a |
| sft_s1 | 0.886 | 1800 | −0.026 / −0.060 | n/a |
| pair_s0 | 0.892 | 800 | −0.086 / −0.307 | ~0.08 |
| pair_s1 | 0.888 | 1800 | −0.095 / −0.332 | ~0.07 |

Training setup:
- **SFT**: 147,128 Lean-valid finished train attempts; Dan's action-token CE on effective (canonical) actions;
  lr 1e-4; 128 proofs per step.
- **Pair**: 1,492 theorems with both classes; 9,499 finished-invalid attempts; DPO-style logistic on the difference
  of masked mean log p relative to frozen s1, with β = 10 and λ_KL = 1.0 (token KL(s1 ‖ policy) at all scored action
  positions); lr 5e-5; 16 theorems × up to 4 + 4 attempts per step.

**Drift.** The anchor did *not* keep likelihoods at s1's level:
- The pair judges lower the masked log p of invalid proofs about 3.5× (−0.09 → −0.31) and of valid proofs about 1.8×
  (−0.05 → −0.09). The token KL to s1 is about 0.07–0.08 nats per token.
- The pair judge is therefore a re-calibrated model, not a faithful generator likelihood. As a *generator* it would
  likely be worse; as a ranking judge it is better.
- SFT raises the likelihood of all finished proofs, valid and invalid.

## Checks and exposure

- **Label-permutation test**: Worker B's `score_loglik.py` with each of the 4 final checkpoints on dev1108 under
  shuffled labels gives 4/4 byte-identical scores.
- **Data and early stopping**: training and early stopping used only s0 train/dev-split pools and labels. No test pool
  (s0 k64, k64b, s1, s2) was scored before training ended or read before the freeze.
- **Checkpoint format**: checkpoints are saved in Dan's format (`state`, `tok_mode`, `cfg`) and load with `load_ckpt`.
- **Prior exposure**: test-pool *likelihood* baselines were known from earlier phases and Worker B's work.
- **OOM restarts**: two of my own processes OOM'd on their first launch; I switched them to length-bucketed batches and
  restarted with the same seeds. No foreign-process OOMs this phase.

## Limitations

1. **Few runs.** Two seeds per variant. Seed agreement is close (within about 0.01–0.03 within-theorem on test), but
   only one hyperparameter setting each (β, λ, lr) was run.
2. **Small textbook72 pools.** They have 32–42 theorems with both finished classes, so the large textbook72 gains have
   wide intervals (±0.07).
3. **Recalibrated judge.** The pair judge's absolute likelihoods drifted a lot, so it should not be used as a sampler
   or as a calibrated probability. Only its within-theorem ranking was evaluated.
4. **Possible undertraining.** sft_s0's best checkpoint is its final step.
5. **Selection headroom is small.** Plain s1 likelihood is already near the oracle on holdout250, so most of the AUROC
   gain does not turn into k=8 accuracy.

## Published scores (`~/runs/proof-selection/scores/`)

- Raw judge scores: `<stem>__judge_{sft,pair}_s{0,1}`, `judge_sft`, `judge_pair`. These are seed means of B's
  `loglik_mean_masked_judge_*`, which are also kept.
- Completion-first composites: `completion_judge_{sft,pair}[_s{0,1}]`.
- Sibling-likelihood mean: `loglik_sibmean` and `completion_loglik_sibmean`.
- Pools covered: holdout250 and textbook72 for s0 k64, s0 k64b, s1 k64 and s2 k64, plus judge scores on dev1108 k16.
- Checkpoints: `~/runs/proof-selection/judge/{sft,pair}_s{0,1}/best.pt`.
