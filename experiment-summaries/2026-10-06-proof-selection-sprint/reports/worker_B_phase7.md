---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 7: stopping with Worker A's trained pairwise judge

The judge is Worker A's `judge_pair`: the mean masked log-likelihood under `judge/pair_s0` and `judge/pair_s1`.
It is a pairwise-trained, recalibrated ranker initialized from s1_r16 and trained on s0's train pools. Per Worker A's
phase-4 log it is not a faithful likelihood.

- **Protocol.** Pre-registered in `../logs/worker_B.md` ("Phase 7") and frozen with sha256 prefixes in the log. I
  rescored rr600 with the judge for the dev split. The first test run crashed on a file-path bug before producing any
  output; I fixed the path, re-ran dev (byte-identical choice file), and logged the fix.
- **Code:** `code/phase7.py`, `code/phase7_figs.py`.
- **Results:** `results/B7_dev_choice.json`, `B7_results.{json,txt}`, `B7_table.md`.
- **Figure:** `figures/B7_headline.png`. B5 is unchanged.
- **Pools:** test pools from generators s0, s1 and s2, plus the second s0 draw (s0b). N_max is 16 and each policy is
  averaged over 50 orderings. "Pooled" means averaged over the three generators, with theorems resampled jointly
  (B = 2000).

## Table: dev-chosen operating points on test (mean over generators s0, s1, s2)

Each cell is accuracy @ mean attempts.

| set | m (dev target) | count-stop + siblings | count-stop + trained judge | sibling threshold (phase 5) | trained-judge threshold | oracle: Lean stops at first valid (at the trained-judge threshold's attempts) |
|---|---|---|---|---|---|---|
| holdout250 | 2 | 0.889 @ 1.59 (c=1) | 0.889 @ 1.59 (c=1) | 0.889 @ 1.59 | **0.889 @ 1.59** | 0.941 |
| holdout250 | 4 | 0.915 @ 2.79 (c=2) | 0.920 @ 2.79 (c=2) | 0.929 @ 3.85 | **0.935 @ 3.12** | 0.953 |
| holdout250 | 8 | 0.933 @ 7.09 (c=6) | 0.938 @ 7.09 (c=6) | 0.937 @ 8.22 | **0.941 @ 6.45** | 0.953 |
| textbook72 | 2 | 0.527 @ 3.99 (c=1) | 0.527 @ 3.99 (c=1) | 0.527 @ 3.99 | **0.527 @ 3.99** | 0.599 |
| textbook72 | 4 | 0.567 @ 6.08 (c=2) | 0.579 @ 6.08 (c=2) | 0.611 @ 9.83 | **0.630 @ 9.14** | 0.653 |
| textbook72 | 8 | 0.604 @ 10.75 (c=6) | 0.624 @ 10.75 (c=6) | 0.613 @ 13.40 | **0.633 @ 12.05** | 0.653 |

## 1. Count-stop: trained judge vs sibling likelihood at the same c

The paired difference is in accuracy. Attempts are identical, because the selector does not change when count-stop
stops.

| set | c | attempts | siblings | trained judge | diff, pooled over 3 generators | diff, second s0 draw |
|---|---|---|---|---|---|---|
| holdout250 | 2 | 2.79 | 0.915 | 0.920 | +0.006 [0.003, 0.009] | +0.007 [0.003, 0.011] |
| holdout250 | 4 | 5.01 | 0.928 | 0.933 | +0.005 [0.001, 0.010] | +0.005 [−0.001, 0.011] |
| holdout250 | 6 | 7.09 | 0.933 | 0.938 | +0.005 [−0.001, 0.011] | +0.003 [−0.005, 0.011] |
| textbook72 | 2 | 6.08 | 0.567 | 0.579 | +0.011 [0.004, 0.019] | +0.011 [0.002, 0.022] |
| textbook72 | 4 | 8.83 | 0.592 | 0.612 | +0.020 [0.008, 0.034] | +0.019 [−0.001, 0.043] |
| textbook72 | 6 | 10.75 | 0.604 | 0.624 | +0.019 [0.006, 0.036] | +0.019 [−0.004, 0.049] |

At c = 1 both selectors are identical.

## 2. Absolute-threshold stopping on the judge score

**Cost-matched comparison** at the policy's realised attempts (count curves interpolated as in phase 5; pooled over
the 3 generators):

| set | dev target m | attempts | accuracy | vs count-stop + judge @ same attempts | vs count-stop + siblings @ same attempts | gap to first-valid oracle closed (vs count + judge / vs count + siblings) |
|---|---|---|---|---|---|---|
| holdout250 | 3 | 2.32 | 0.928 | +0.021 [0.014, 0.027] | +0.024 [0.017, 0.032] | 46 % / 50 % |
| holdout250 | 4 | 3.12 | 0.935 | +0.017 [0.011, 0.023] | +0.021 [0.015, 0.029] | 48 % / 53 % |
| holdout250 | 8 | 6.45 | 0.941 | +0.006 [0.003, 0.009] | +0.010 [0.005, 0.016] | 30 % / 44 % |
| textbook72 | 3 | 7.94 | 0.622 | +0.017 [0.002, 0.032] | +0.037 [0.017, 0.061] | 36 % / 56 % |
| textbook72 | 4 | 9.14 | 0.630 | +0.015 [0.007, 0.025] | +0.035 [0.018, 0.056] | 40 % / 60 % |
| textbook72 | 8 | 12.05 | 0.633 | +0.006 [0.003, 0.011] | +0.024 [0.009, 0.042] | 23 % / 54 % |

At m = 2 every rule collapses to first-finished, as in phase 5.

**Against the phase-5 sibling threshold, at both rules' dev-chosen points** (not cost-matched; attempts stated):

| set | m | trained-judge threshold | sibling threshold | diff [95 % CI] |
|---|---|---|---|---|
| holdout250 | 3 | 0.928 @ 2.32 | 0.920 @ 2.73 | +0.008 [0.001, 0.015] |
| holdout250 | 4 | 0.935 @ 3.12 | 0.929 @ 3.85 | +0.006 [−0.001, 0.014] |
| holdout250 | 8 | 0.941 @ 6.45 | 0.937 @ 8.22 | +0.004 [−0.003, 0.010] |
| textbook72 | 3 | 0.622 @ 7.94 | 0.603 @ 8.26 | +0.019 [0.002, 0.041] |
| textbook72 | 4 | 0.630 @ 9.14 | 0.611 @ 9.83 | +0.020 [0.003, 0.038] |
| textbook72 | 8 | 0.633 @ 12.05 | 0.613 @ 13.40 | +0.020 [0.005, 0.037] |

**Budget transfer to textbook72.** At target m = 4 the judge threshold spends 9.14 attempts (2.3×). The sibling
threshold spends 9.83 (2.5×) and count-stop 6.08 (1.5×). The trained judge does not solve budget transfer.

## 3. Does the judge's absolute score transfer across theorems?

Pooled finished-only AUROC (all finished attempts of the set, any theorems):

| pool | judge | sibling likelihood | judge within-theorem (n) |
|---|---|---|---|
| holdout250 s0 | 0.906 | 0.790 | 0.893 (116) |
| holdout250 s1 | 0.883 | 0.800 | 0.905 (135) |
| holdout250 s2 | 0.881 | 0.801 | 0.883 (132) |
| holdout250 s0 second draw | 0.893 | 0.782 | 0.907 (134) |
| textbook72 s0 | 0.960 | 0.770 | 0.942 (34) |
| textbook72 s1 | 0.931 | 0.892 | 0.923 (41) |
| textbook72 s2 | 0.921 | 0.844 | 0.924 (42) |
| textbook72 s0 second draw | 0.937 | 0.742 | 0.938 (32) |

Yes. The judge's pooled finished-only AUROC (0.88–0.96) is much higher than the sibling likelihood's (0.74–0.89). Its
absolute score is a far better cross-theorem confidence signal. That is why the absolute threshold works better, and
the gap closed to the first-valid oracle roughly doubles relative to phase 5: about 50 % against 18–21 % on
holdout250 at 2–4 attempts.

## Limitations

- **Judge selection and dev data.** The judge was trained and early-stopped by Worker A on the s0 dev split that I
  also used to choose τ. Dev numbers are therefore optimistic for the judge. The test numbers are a single read.
- **The s1-generated pools.** The judge's initialization (s1_r16) is that pool's own generator. The judge is trained
  away from s1, and its within-theorem AUROC there (0.905) is no lower than on the other pools.
- **Statistics and cost.** One judge recipe and two judge seeds (averaged). Bootstrap intervals cover theorems only.
  N_max is 16. Each candidate costs two judge forward passes.
- **Exposure.** The test sets have been read in all earlier phases.
