---
written_on: 2026-10-06
written_by: agent:claude (Worker A)
status: complete
---

# Worker A phase 3: does the learned head inherit the generator's blind spot through its initialisation?

**Setup.**
- Pre-registered in `logs/worker_A.md` at 04:57 UTC; freeze at 05:07:39 UTC (`results/phase3_test_freeze_sha256.json`,
  120 files, re-hashed after the test read with 0 changed).
- Recipe: the phase-2 top rung (within-theorem pairwise ranker on finished attempts, history input, all data), with
  the body initialised from the sibling generators s1 or s2. Two training seeds per init.
- The s0-init comparator is phase 2's `verifier2_rall_s*`: either the seed-matched 2-run mean
  (`verifier3_init_s0`) or the 3-run `verifier2_ens`.
- Every s0 pool and Worker B's s1/s2-generated test pools are scored.
- Code: `code/train_verifier2.py` (`--init`, `--scorer`, pool spec `set:k:gen`), `code/phase3_report.py`.

## Bottom line: the hypothesis is not supported

- **P1 (sibling-initialised heads beat the s0-initialised head on s0's test pools): not supported.**
  - holdout250: `verifier3_sibinit_ens` 0.698 vs `verifier2_ens` 0.677, difference +0.021 [−0.009, +0.051].
  - textbook72: 0.609 vs 0.638, difference −0.029 [−0.094, +0.045].
  - Neither difference is resolved, and they point in opposite directions.
  - The s1-initialised head is significantly *worse* than the s0-initialised head on textbook72 (−0.125
    [−0.193, −0.059], 34 theorems).
- **P2 (sibling-initialised heads reach sibling likelihood): clearly rejected.**
  - holdout250: 0.698 vs `loglik_mean_masked_siblings` 0.829, difference −0.131 [−0.184, −0.074].
  - textbook72: 0.609 vs 0.782, difference −0.174 [−0.300, −0.041].
  - The heads stay below even the *self* likelihood judge: −0.090 [−0.149, −0.024] on holdout250, −0.078
    [−0.211, +0.049] on textbook72.
- **Secondary (self penalty for heads): no clear diagonal.**
  - On s1 pools the s1-initialised head is the lowest of the three on both sets, which matches the prediction.
  - Every paired difference is unresolved (holdout250 −0.019 / −0.022; textbook72 −0.047 / −0.019).
  - On s2 pools the s2-initialised head is not the worst (holdout250 +0.010 / −0.016; textbook72 +0.010 / +0.025).
  - B's likelihood judges show a consistent self penalty on the diagonal (bottom row of the figure). The learned heads
    do not: they are uniformly poor, 0.53–0.75.
- **Interpretation (hypothesis, not tested).** Whatever limits the fine-tuned head, it is not mainly the generator's
  self-blind-spot inherited through initialisation. Fine-tuning a scalar head discards the per-token likelihood
  signal; a sibling *likelihood judge* keeps it.

## Within-theorem finished-only AUROC: init × generator matrix (`figures/A3_init_matrix.png`)

| test set | pool generator | head init s0 | head init s1 | head init s2 | judge s0 | judge s1 | judge s2 | theorems |
|---|---|---|---|---|---|---|---|---|
| holdout250 | s0 | **0.678** | 0.656 | 0.695 | **0.788** | 0.828 | 0.786 | 116 |
| holdout250 | s1 | 0.748 | **0.728** | 0.750 | 0.822 | **0.775** | 0.813 | 135 |
| holdout250 | s2 | 0.679 | 0.705 | **0.690** | 0.824 | 0.842 | **0.807** | 132 |
| textbook72 | s0 | **0.654** | 0.529 | 0.635 | **0.687** | 0.783 | 0.737 | 34 |
| textbook72 | s1 | 0.630 | **0.583** | 0.602 | 0.829 | **0.758** | 0.833 | 41 |
| textbook72 | s2 | 0.581 | 0.565 | **0.591** | 0.825 | 0.831 | **0.771** | 42 |

Notes on the table:
- Bold marks self cells, where the head or judge comes from the pool's own generator.
- Each head cell is the mean of 2 training runs of one init; the s0 column uses verifier2_rall_s0 and s1.
- Judge cells are Worker B's `loglik_mean_masked[_sX]`.
- Only theorems with both valid and invalid finished attempts count.

## Primary paired comparisons on s0 test pools (theorem bootstrap, B=1000)

| comparison | holdout250 (116) | textbook72 (34) |
|---|---|---|
| P1 sibinit_ens − verifier2_ens (s0, 3 runs) | +0.021 [−0.009, +0.051] | −0.029 [−0.094, +0.045] |
| P1 sibinit_ens − s0 init (seed-matched, 2 runs) | +0.021 [−0.010, +0.052] | −0.045 [−0.100, +0.017] |
| init_s1 − init_s0 | −0.022 [−0.062, +0.020] | −0.125 [−0.193, −0.059] |
| init_s2 − init_s0 | +0.017 [−0.021, +0.055] | −0.019 [−0.086, +0.056] |
| P2 sibinit_ens − loglik_mean_masked_siblings | −0.131 [−0.184, −0.074] | −0.174 [−0.300, −0.041] |
| sibinit_ens − loglik_mean_masked (self judge) | −0.090 [−0.149, −0.024] | −0.078 [−0.211, +0.049] |

## Selection at k=8 on s0 test pools (all theorems; `results/selection_{holdout250,textbook72}_p3.json`)

| scorer | holdout250 | textbook72 |
|---|---|---|
| completion_loglik_siblings (B) | **0.928** [0.897, 0.954] | **0.567** [0.467, 0.675] |
| completion_loglik | 0.918 | 0.540 |
| completion_verifier2_ens (s0 init) | 0.913 | 0.514 |
| completion_verifier3_init_s0 / _s1 / _s2 | 0.916 / 0.907 / 0.915 | 0.526 / 0.507 / 0.520 |
| completion_verifier3_sibinit_ens | 0.914 [0.883, 0.941] | 0.513 [0.416, 0.617] |

Paired differences at k=8:

| comparison | holdout250 | textbook72 |
|---|---|---|
| completion_loglik_siblings − completion_verifier3_sibinit_ens | +0.013 [+0.006, +0.023] | +0.054 [+0.026, +0.087] |
| completion_verifier2_ens − completion_verifier3_sibinit_ens | −0.002 [−0.007, +0.003] | +0.001 [−0.012, +0.014] |

## Runs and checks

- **Dev results.** Best dev within-theorem AUROC (all dev pools):
  - init_s1: 0.673 / 0.687
  - init_s2: 0.676 / 0.668
  - s0 init (phase 2): 0.676 / 0.683 / 0.680
  - There was no dev advantage before the test read either.
- **Checkpoint load.** All three generator checkpoints load with tok_mode lean_staten, vocab 111 and cfg 6×384;
  `train_verifier2.py` asserts this.
- **Pools.** Worker B's s1/s2 test pools replay byte for byte (0 mismatches). Their rows say `gen: s0_r16`, a
  collector constant; the `.gen.json` sidecar gives the true generator.
- **Label permutation.** 4/4 phase-3 models are byte-identical under shuffled labels (score-only, dev1108).
- **Published scores** (in `~/runs/proof-selection/scores/`):
  - `verifier3_init_s{1,2}_seed{0,1}`
  - `verifier3_init{s1,s2}`, which equal `verifier3_init_s{1,2}` (2-run means)
  - `verifier3_init_s0` (s0 init, seed-matched)
  - `verifier3_sibinit_ens`
  - `completion_verifier3_*`
  - All are written for every s0 pool and every s1/s2 test pool. `verifier2_*` is also written for the s1/s2 pools.

## Limitations

- **Small runs.** Two training runs per init. Single runs vary by about 0.03–0.04, which is the size of most cell
  differences.
- **Small textbook72 rows.** They have 34–42 theorems; the one significant cell difference (init_s1 vs init_s0 on s0
  textbook72) could be a seed effect.
- **Only s0 data in training.** Every head is trained on s0-generated attempts only. A head trained on sibling-generated
  pools was not tested.
- **Same input format.** All heads see identical replayed inputs, and only the body initialisation differs.

## Correction to phase 1 (coordinator query)

Phase-1 limitation 8 ("14 of 72 textbook72 theorems were seen by the generator during RL") should be withdrawn.
- **Evidence it rested on.** Only the field `role: "rl_train"` on 14 rows of `nd-takehome/data/bs/textbook72.jsonl`;
  the other 58 rows say `textbook_dev`.
- **What the label means.** Dan's `numbers.md` calls this the "dev58 + train14" split (Robbie's naming), and I found
  nothing saying the r16 generator trained on these problems.
- **Overlap.** I did not check rl_targets / K12 membership myself. The midpoint audit found no rl_targets overlap.
- **The 14 names:** textbook_0c463f021f70dc4f6b35, _0dd5ea3d0aa761fea3d9, _0ed8057166e48df4ff93, _54e1d81ad54a80f1f835,
  _58b532c9c8cc1819378b, _6997656e16fb567550a6, _92b4192856ec54a3e1ee, _936ceade34df08c26377, _9fa52f90369a7b6286f7,
  _bcb4f30a22574210fc12, _cea8801c51d3f9d213d6, _da42d4814cf3a388b9fc, _f8e226f6c5b45234819d, _f918caefa752f2beda53.
