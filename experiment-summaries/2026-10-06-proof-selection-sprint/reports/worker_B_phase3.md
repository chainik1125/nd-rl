---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 3: is a generator a worse judge of its own proofs? (generator × judge matrix)

Branch `dmitry/selection-sprint` (uncommitted). Pre-registered in `../logs/worker_B.md` ("Phase 3", 04:16 UTC), before
any phase-3 pool was generated. The test read happened once (`code/phase3.py`, 04:46 UTC). Nothing in phase 3 was
tuned: all parameters were fixed in the pre-registration or carried over from phase 2.

## Summary

- **Yes: a generator is a measurably worse judge of its own proofs.** We added the s1 and s2 r16 generators on both
  test sets, giving all 3 × 3 (generator, judge) cells.
  - **Within-theorem AUROC (finished attempts).** The self-judging cell is the lowest in its row in 5 of 6 rows; the
    exception is holdout250 generator s0, where judge s2 ties it (0.786 vs 0.788). The two-way fit (generator +
    judge + self indicator) gives a self effect of −0.029 [−0.048, −0.010] on holdout250 and −0.068 [−0.115, −0.019]
    on textbook72. The pre-registered criterion is met (δ < 0 with the CI excluding 0 on both sets).
  - **k = 8 selection.** The penalty carries through: self effect −0.010 [−0.016, −0.005] on holdout250 and −0.022
    [−0.036, −0.010] on textbook72.
- **This resolves the phase-2 confound.** The judge-identity effects are small and inconsistent (judge s1 vs s0:
  +0.004 / +0.010 AUROC). So the sibling gain is not "s0 is a weak judge"; it is mostly the absence of self-judging.
- **The sibling-selection gain replicates for every generator.** Siblings vs self `completion_loglik` at k = 8:
  - holdout250: s0 +0.011, s1 +0.011, s2 +0.015 (all CIs exclude 0);
  - textbook72: s0 +0.030, s1 +0.020, s2 +0.026 (all exclude 0).
- **Count-stop with sibling selection replicates too.** At c = 6 (about 7 attempts on holdout250, 10–11.5 on
  textbook72): holdout250 +0.009 / +0.014 / +0.014; textbook72 +0.043 / +0.018 / +0.026. Eight of the twelve
  (generator, set, c ∈ {2, 6}) CIs exclude 0, and all twelve point estimates are positive.
- **Generator-seed variance at k = 8** (sd over 3 generators) is 0.005–0.014 on holdout250 and 0.008–0.033 on
  textbook72. The textbook72 ordering of methods is stable across seeds; the absolute levels move by up to 0.06.

## 1. Pools and checks

Pools were generated with Worker A's `code/collect_pools.py`, unchanged: T 0.8, max_action 512, max_steps 96, batch
2048, `--check_replay`.

- Generators and seeds: `la_T1_best12_s1_r16.pt` with sampling seed 1, and `_s2_r16.pt` with seed 2.
- GPUs 5 and 6 were used; they were idle.
- Pools: `pools_v2/{holdout250,textbook72}__{s1,s2}_r16__k64.{jsonl,labels.jsonl,summary.json,gen.json}`.
- Replay: 16,000/16,000 and 4,608/4,608 attempts per pool, 0 mismatches.
- `MANIFEST.json` entries were added by `code/manifest_phase3.py`.
- Caveat: the collector writes the row field `gen = "s0_r16"` as a constant. The true generator is recorded in
  `<stem>.gen.json` and in the manifest.

Scoring: each new pool was scored under judges s0, s1 and s2 (`score_loglik.py --ckpt … --tag s{j}`) with the same
masking and checks. All 6 × 3 runs had 0 replay errors, done mismatches, ND mismatches or bad state hashes, and used
raw ids throughout.

## 2. Generator × judge matrix (`results/B3_results.{json,txt}`, `figures/B3_judge_matrix.png`)

Diagonal = self-judging. "siblings" = mean masked loglik of the two non-generator judges; "xseed" = mean of all three.

**Within-theorem finished-only AUROC**

| set | generator | judge s0 | judge s1 | judge s2 | siblings | xseed |
|---|---|---|---|---|---|---|
| holdout250 | s0 | **0.788** | 0.828 | 0.786 | 0.829 | 0.833 |
| holdout250 | s1 | 0.822 | **0.775** | 0.813 | 0.824 | 0.832 |
| holdout250 | s2 | 0.824 | 0.842 | **0.807** | 0.856 | 0.866 |
| textbook72 | s0 | **0.687** | 0.783 | 0.737 | 0.782 | 0.780 |
| textbook72 | s1 | 0.829 | **0.758** | 0.833 | 0.852 | 0.864 |
| textbook72 | s2 | 0.825 | 0.831 | **0.771** | 0.841 | 0.843 |

**k = 8 selection accuracy by `completion_loglik`** (200 shared random 8-subsets per theorem)

| set | generator | judge s0 | judge s1 | judge s2 | siblings | xseed |
|---|---|---|---|---|---|---|
| holdout250 | s0 | **0.917** | 0.930 | 0.923 | 0.928 | 0.928 |
| holdout250 | s1 | 0.938 | **0.923** | 0.928 | 0.933 | 0.935 |
| holdout250 | s2 | 0.922 | 0.916 | **0.909** | 0.924 | 0.924 |
| textbook72 | s0 | **0.543** | 0.573 | 0.557 | 0.573 | 0.570 |
| textbook72 | s1 | 0.563 | **0.544** | 0.563 | 0.565 | 0.565 |
| textbook72 | s2 | 0.605 | 0.604 | **0.581** | 0.607 | 0.606 |

**k = 64 selection** (the whole pool, so one subset; noisy on textbook72):

- holdout250 self 0.936 / 0.924 / 0.916 against siblings 0.952 / 0.960 / 0.936;
- textbook72 self 0.625 / 0.583 / 0.653 against siblings 0.653 / 0.625 / 0.667.

**Self-judging penalty tests**

The two-way fit is cell = μ + generator + judge + δ·[generator = judge], least squares on 9 cells. The CI comes from a
theorem bootstrap (B = 2000; theorems resampled jointly across the three pools).

| set | metric | δ (self effect) [95 % CI] | judge effects (s1, s2 vs s0) | generator effects (s1, s2 vs s0) |
|---|---|---|---|---|
| holdout250 | within AUROC | **−0.029 [−0.048, −0.010]** | +0.004, −0.009 | +0.003, +0.024 |
| holdout250 | sel k=8 | **−0.010 [−0.016, −0.005]** | −0.003, −0.006 | +0.006, −0.008 |
| holdout250 | sel k=64 | −0.021 [−0.034, −0.009] | −0.009, −0.005 | −0.001, −0.021 |
| textbook72 | within AUROC | **−0.068 [−0.115, −0.019]** | +0.010, +0.000 | +0.071, +0.073 |
| textbook72 | sel k=8 | **−0.022 [−0.036, −0.010]** | +0.003, −0.003 | −0.001, +0.039 |
| textbook72 | sel k=64 | −0.028 [−0.067, +0.009] | −0.005, +0.005 | −0.023, +0.037 |

Per-generator paired differences (other judge − self), within-theorem AUROC; all in `B3_results.txt`:

| set | generator | − self, judge a | − self, judge b | siblings − self | xseed − self |
|---|---|---|---|---|---|
| holdout250 | s0 | s1 +0.040 [0.003, 0.076] | s2 −0.002 [−0.039, 0.036] | +0.041 [0.007, 0.074] | +0.045 [0.014, 0.075] |
| holdout250 | s1 | s0 +0.047 [0.008, 0.081] | s2 +0.037 [0.005, 0.069] | +0.049 [0.014, 0.082] | +0.057 [0.027, 0.086] |
| holdout250 | s2 | s0 +0.017 [−0.016, 0.051] | s1 +0.035 [0.002, 0.068] | +0.049 [0.017, 0.082] | +0.059 [0.030, 0.087] |
| textbook72 | s0 | s1 +0.095 [0.020, 0.170] | s2 +0.050 [−0.047, 0.138] | +0.095 [0.020, 0.174] | +0.093 [0.022, 0.167] |
| textbook72 | s1 | s0 +0.072 [−0.001, 0.146] | s2 +0.076 [0.014, 0.136] | +0.094 [0.022, 0.163] | +0.106 [0.053, 0.164] |
| textbook72 | s2 | s0 +0.053 [−0.017, 0.127] | s1 +0.060 [−0.014, 0.144] | +0.070 [−0.002, 0.147] | +0.072 [0.012, 0.136] |

Interpretation:

- **Observed.** Every one of the 12 single-judge comparisons has a positive point estimate except one (holdout250
  s0: judge s2 −0.002). Averaging judges (siblings, xseed) helps on top of that.
- **Hypothesis, not tested here.** One explanation is that a generator's own likelihood rewards its own systematic
  errors (high-confidence wrong steps it is prone to), which an independently trained seed does not share.
- **Remaining confound.** The xseed column includes the self judge and is still best or tied-best. So
  "ensemble of judges" and "not self" both contribute and are not separated.

## 3. Secondary: replication of selection and count-stop gains

**Siblings vs self `completion_loglik` selection, k = 8 and k = 64**

| set | generator | k=8 self → siblings, diff [CI] | k=64 self → siblings, diff [CI] |
|---|---|---|---|
| holdout250 | s0 | 0.917 → 0.928, +0.011 [0.001, 0.021] | 0.936 → 0.952, +0.016 [0.000, 0.036] |
| holdout250 | s1 | 0.923 → 0.933, +0.011 [0.001, 0.021] | 0.924 → 0.960, +0.036 [0.016, 0.060] |
| holdout250 | s2 | 0.909 → 0.924, +0.015 [0.004, 0.027] | 0.916 → 0.936, +0.020 [0.000, 0.044] |
| textbook72 | s0 | 0.543 → 0.573, +0.030 [0.011, 0.055] | 0.625 → 0.653, +0.028 [−0.042, 0.097] |
| textbook72 | s1 | 0.544 → 0.565, +0.020 [0.006, 0.038] | 0.583 → 0.625, +0.042 [−0.028, 0.125] |
| textbook72 | s2 | 0.581 → 0.607, +0.026 [0.003, 0.056] | 0.653 → 0.667, +0.014 [−0.056, 0.083] |

The s0-generator k = 8 values differ by about 0.001 from phase 2, because this script uses 200 shared subsets rather
than phase 2's 50 orderings.

**Count-stop (N_max 16; c fixed by the phase-2 dev choice) with self vs sibling selection**

| set | generator | c=2: att, self → siblings, diff [CI] | c=6: att, self → siblings, diff [CI] |
|---|---|---|---|
| holdout250 | s0 | 2.94, 0.915 → 0.920, +0.004 [0.000, 0.008] | 7.18, 0.923 → 0.932, +0.009 [−0.000, 0.018] |
| holdout250 | s1 | 2.69, 0.914 → 0.921, +0.007 [0.003, 0.012] | 7.07, 0.926 → 0.940, +0.014 [0.005, 0.024] |
| holdout250 | s2 | 2.79, 0.892 → 0.899, +0.007 [0.002, 0.012] | 7.04, 0.910 → 0.924, +0.014 [0.004, 0.025] |
| textbook72 | s0 | 6.79, 0.550 → 0.568, +0.018 [0.008, 0.031] | 11.50, 0.564 → 0.608, +0.043 [0.017, 0.076] |
| textbook72 | s1 | 6.31, 0.546 → 0.556, +0.009 [−0.001, 0.019] | 10.66, 0.562 → 0.580, +0.018 [−0.003, 0.039] |
| textbook72 | s2 | 5.11, 0.564 → 0.579, +0.014 [0.002, 0.028] | 10.10, 0.604 → 0.630, +0.026 [0.003, 0.057] |

Attempts are identical between the two selectors, because the selector does not change when count-stop stops.

## 4. Generator-seed variance of headline k = 8 selection numbers

Values are for generators s0 / s1 / s2, followed by mean ± sd over the three generators. Each pool was sampled once.

| set | random pick | completion pick | completion_loglik (self judge) | completion_loglik (sibling judges) | oracle pass@8 |
|---|---|---|---|---|---|
| holdout250 | 0.836 / 0.831 / 0.818 (0.829 ± 0.009) | 0.898 / 0.891 / 0.870 (0.887 ± 0.014) | 0.917 / 0.923 / 0.909 (0.916 ± 0.007) | 0.928 / 0.933 / 0.924 (0.928 ± 0.005) | 0.939 / 0.955 / 0.935 (0.943 ± 0.010) |
| textbook72 | 0.407 / 0.434 / 0.439 (0.427 ± 0.017) | 0.516 / 0.515 / 0.529 (0.520 ± 0.008) | 0.543 / 0.544 / 0.581 (0.556 ± 0.022) | 0.573 / 0.565 / 0.607 (0.581 ± 0.022) | 0.598 / 0.583 / 0.645 (0.609 ± 0.033) |

The method ordering (random < completion < self-loglik < sibling-loglik < oracle) holds for every generator on both
sets. Fraction of the random → oracle gap recovered by sibling selection: holdout250 0.89 / 0.83 / 0.90; textbook72
0.87 / 0.88 / 0.81.

## 5. Commands

```bash
cd ~/runs/proof-selection; PY=~/venvs/proof-state/bin/python; S=<sprint folder>; export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast LEAN_GATE_WORKERS=16
CUDA_VISIBLE_DEVICES=$((4+g)) $PY $S/code/collect_pools.py --set {textbook72,holdout250} --k 64 --seed $g --ckpt ckpts/la_T1_best12_s${g}_r16.pt --out pools_v2/<set>__s${g}_r16__k64.jsonl --check_replay   # g = 1, 2
CUDA_VISIBLE_DEVICES=7 $PY $S/code/score_loglik.py --pool pools_v2/<set>__s${g}_r16__k64.jsonl --ckpt ckpts/la_T1_best12_s${j}_r16.pt --tag s$j   # j = 0, 1, 2
$PY $S/code/manifest_phase3.py
$PY $S/code/phase3.py --out $S/results/B3_results.json     # the single test read
$PY $S/code/phase3_figs.py
```

## 6. Limitations

- **One pool per generator.** Each generator has a single sampling-seed pool, so the bootstrap covers theorems, not
  the sampling noise between pools. The three generators also differ in seed-specific training.
- **Small decomposition.** The two-way fit uses 9 cells and 6 parameters, leaving 3 residual degrees of freedom.
  Its CI is theorem-bootstrap only.
- **Within-theorem AUROC sample sizes.** It is defined only for theorems with both classes among finished attempts.
  These counts are in `B3_results.json` (`cells.auc_n`): holdout250 116 / 135 / 132, textbook72 34 / 41 / 42 theorems for generators s0 / s1 / s2.
- **"Siblings" is not the same pair across rows.** For generator s1 it is s0 + s2, so the column mixes judge
  identities.
- **Test exposure.** The test sets had been read in phases 1–2. No parameter here was tuned on them, but the decision
  to look at self-judging came from the phase-2 test results.
- **Likelihood.** T = 1 masked likelihood is a diagnostic quantity, not the sampling probability.
