---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 2: sibling-judge likelihood, threshold-free stopping, difficulty-aware allocation

Branch `dmitry/selection-sprint` (uncommitted). Generator: frozen `la_T1_best12_s0_r16.pt`; pools from Worker A,
seed 0. Pre-registration and freeze are in `../logs/worker_B.md`, sections "Phase 2":

- Pre-registration 03:36 / 03:48 UTC.
- Freeze 03:53 UTC, with sha256 prefixes of `phase2_alloc_fit.json`, `phase2_grids.json`, both dev-choice files, both
  phase-2 scripts, and the τ grid.

The test read happened once, after the freeze. Phase-1 midpoint-audit fixes are in `worker_B_report.md`:

- Corrected sentences.
- Corrected token costs: ensembles are charged 3 passes; the phase-1 total-token panels and `stopping_headline.json`
  were regenerated. Attempt, accuracy and generated-token arrays were verified byte-identical.
- Realised-cost comparators labelled as descriptive.

## Summary

- **Sibling judges recognise s0's proofs better than s0 itself.** The pre-registered primary comparison was
  within-theorem finished-only AUROC, siblings (mean of s1, s2) minus self (s0), as a paired theorem bootstrap:
  - dev1108 dev +0.060 [0.018, 0.104]; rr600 dev +0.042 [-0.003, 0.086];
  - holdout250 +0.041 [0.004, 0.078]; textbook72 +0.095 [0.019, 0.177].
  - Pooled finished-only: +0.11 to +0.13 on every set (all CIs exclude 0).
  - Selection with `completion_loglik_siblings` against s0 self-judging, at k = 8:
    holdout250 0.927 vs 0.918 (+0.009 [0.000, 0.020]); textbook72 0.571 vs 0.544 (+0.027 [0.011, 0.049]).
  - At k = 64: 0.952 vs 0.936 on holdout250 (oracle 0.960), and 0.653 vs 0.625 on textbook72 (oracle 0.694; CI spans 0).
- **What the sibling result does and does not show.** The pre-registered mechanism ("the generator's own errors are
  shared with its own likelihood") is **not** cleanly supported.
  - Adding s0 back in (xseed = mean of s0, s1, s2) is as good as siblings only: within-theorem 0.833 vs 0.829 on
    holdout250, 0.780 vs 0.782 on textbook72.
  - The two siblings differ. s1 alone is clearly better than s0. s2 alone ties s0 on holdout250 (-0.002
    [-0.041, 0.038]) and is +0.050 [-0.046, 0.143] on textbook72.
  - Observed: averaging several r16 judges is a robust improvement, and excluding the generator itself is not
    necessary. Interpretation (untested): the gain is ensembling, or seed quality, rather than generator-specific
    blindness.
- **Threshold-free stopping transfers its budget better than absolute thresholds, but beats count-stop only
  marginally.**
  - Budget transfer: the dev-chosen "m = 4" points spend 2.8 to 3.6 attempts on holdout250 and 6.7 to 7.0 on
    textbook72. The phase-1 absolute likelihood threshold spent 3.9 / 12.1.
  - Rank rule R2 (z-score of the incumbent against the theorem's own attempts) and count-stop trace the same
    accuracy-vs-attempts curve. Their differences at a dev target follow the attempt difference: on textbook72, r2 −
    count is +0.016 [0.007, 0.026] at m = 2 (Δatt +1.39), +0.001 at m = 4, and −0.010 [−0.023, 0.002] at m = 8
    (Δatt −1.17).
  - Count-stop and R2 beat first-finished by +0.012 to +0.041. That gain is bought with +1.0 to +7.0 more attempts
    (for example textbook72 m = 8: count +0.041 at Δatt +6.96).
  - R1 has almost no budget leverage. Its q sweep spans 2.56–2.78 attempts on dev1108, and on textbook72 it lands at
    5.7–7.0 whatever the target, so it is nearly a fixed policy. R2 is the usable rank rule.
- **Difficulty-aware allocation does not beat count-stop, and it does not move attempts to intermediate theorems.**
  - Policies da / db: continue while p(valid next | verifier difficulty) × (1 − P(incumbent valid)) ≥ λ.
  - At the dev-chosen m = 4 point they spend 15 attempts on hopeless theorems (pass rate < 0.05), the most of any
    policy, because the verifier's difficulty estimate does not identify hopelessness.
  - On textbook72 the budget overshoots badly (m = 4 → 11.9 attempts).
  - Against count-stop at the same dev target: holdout250 -0.000 [-0.004, 0.004]. On textbook72 it is +0.019
    [0.000, 0.039], but the realised attempts are 5.0 higher.
  - Descriptively, at realised cost, every adaptive policy is within +0.002 to +0.019 of fixed-k with the same
    selector.

## 0. Combined result: count-stop with sibling-judge selection (secondary, pre-registered)

Count-stop with `completion_loglik_siblings` as the selector, at the same dev-chosen c
(`results/B2_headline_siblings.txt`), compared with the s0 selector:

| set | m target | c | test att | acc, s0 selector | acc, siblings selector | siblings vs fixed-k(siblings) @ m |
|---|---|---|---|---|---|---|
| holdout250 | 4 | 2 | 2.93 | 0.918 | 0.921 | +0.007 [−0.000, 0.016] |
| holdout250 | 8 | 6 | 7.16 | 0.923 | 0.932 | +0.005 [0.000, 0.010] |
| textbook72 | 4 | 2 | 6.87 | 0.546 | 0.564 | +0.040 [0.020, 0.065] (Δatt +2.87) |
| textbook72 | 8 | 6 | 11.52 | 0.567 | 0.604 | +0.033 [0.017, 0.053] (Δatt +3.52) |

This is the main practical output of phase 2. It needs no threshold, its budget transfers better than the
likelihood threshold, and the better judge adds +0.009 to +0.037 at identical attempts. The siblings-selector da/db
rows are omitted, because their allocation model was fitted for the s0 selector.

## 1. Sibling judges (task 1)

`code/score_loglik.py --ckpt la_T1_best12_s{1,2}_r16.pt --tag s{1,2}` scored every pool (5 pools × 2 judges): the same
replay, masking and checks as phase 1. All 129,456 attempts had 0 replay errors, done mismatches, ND mismatches or bad
state hashes, because the states do not depend on the model. `code/make_xseed.py` builds:

- `loglik_mean_masked_{s1,s2,siblings,xseed}`;
- `completion_loglik_{s1,s2,siblings,xseed}`.

These are in `~/runs/proof-selection/scores/`.

**Within-theorem AUROC, finished attempts only** (`results/B2_auroc_*.json`; 95 % theorem bootstrap; figure
`figures/B2_xseed_auroc.png`):

| set | s0 (self) | s1 | s2 | siblings (s1, s2) | xseed (s0, s1, s2) |
|---|---|---|---|---|---|
| dev1108 dev | 0.787 | 0.846 | 0.832 | 0.846 | 0.856 |
| rr600 dev | 0.765 | 0.802 | 0.774 | 0.807 | 0.815 |
| holdout250 | 0.788 | 0.828 | 0.786 | 0.829 | 0.833 |
| textbook72 | 0.687 | 0.783 | 0.737 | 0.782 | 0.780 |

**Paired differences against s0 on test** (`results/B2_auroc_paired_test.jsonl`; dev in `B2_auroc_paired_dev.jsonl`):

| set | judge | pooled finished-only diff | within-theorem finished-only diff (n theorems) |
|---|---|---|---|
| holdout250 | siblings | +0.122 [0.040, 0.225] | +0.041 [0.004, 0.078] (116) |
| holdout250 | xseed | +0.118 [0.041, 0.220] | +0.045 [0.012, 0.078] |
| holdout250 | s1 | +0.132 [0.052, 0.234] | +0.040 [0.000, 0.078] |
| holdout250 | s2 | +0.085 [-0.004, 0.195] | -0.002 [-0.041, 0.038] |
| textbook72 | siblings | +0.125 [0.084, 0.174] | +0.095 [0.019, 0.177] (34) |
| textbook72 | xseed | +0.110 [0.075, 0.152] | +0.093 [0.022, 0.171] |
| textbook72 | s1 | +0.138 [0.090, 0.194] | +0.095 [0.019, 0.172] |
| textbook72 | s2 | +0.085 [0.025, 0.147] | +0.050 [-0.046, 0.143] |

**Fixed-k selection by `completion_loglik_*`** (R = 50 orderings; `results/B2_selection_paired.{txt,json}`):

| set | k | s0 self | siblings | xseed | siblings - self [95 % CI] | oracle pass@k | completion pick | verifier_ens pick |
|---|---|---|---|---|---|---|---|---|
| holdout250 | 8 | 0.918 | 0.927 | 0.928 | +0.009 [0.000, 0.020] | 0.939 | 0.901 | 0.906 |
| holdout250 | 64 | 0.936 | 0.952 | 0.956 | +0.016 [0.000, 0.036] | 0.960 | 0.911 | 0.924 |
| textbook72 | 8 | 0.544 | 0.571 | 0.570 | +0.027 [0.011, 0.049] | 0.593 | 0.512 | 0.516 |
| textbook72 | 64 | 0.625 | 0.653 | 0.653 | +0.028 [-0.028, 0.097] | 0.694 | 0.549 | 0.542 |

At k = 64 there is one subset, so the CI reflects theorems only. Holdout250 at k = 64 with siblings or xseed recovers
94 to 97 % of the gap between random pick (0.836) and the oracle.

Compute: each sibling judge costs one teacher-forced forward pass per attempt (prefill plus action tokens; about 25 s
per 16 k-attempt pool on one H200).

## 2. Threshold-free stopping and difficulty-aware allocation (task 2)

All of these are in `code/stopping_phase2.py`. N_max is 16 for every policy (the largest cap both dev pools support);
R = 50 orderings, the same as phase 1. Selection is by s0 `completion_loglik` (primary). A siblings selector is the
secondary variant (`*_sel_siblings*`; there the allocation model was not refit, so da/db are mis-specified).

| family | rule |
|---|---|
| ff | stop at first finished, N_max swept |
| count | stop after the c-th finished attempt |
| r1 | stop once ≥ 2 attempts and the finished incumbent's masked loglik ≥ the q-quantile of all of this theorem's attempts so far |
| r2 | stop once ≥ 3 attempts and the finished incumbent's z-score against the theorem's attempts so far ≥ z |
| r3 | stop when some finished ND text has appeared r times; return the most frequent ND (exploratory) |
| da / db | difficulty d = verifier_ens on attempt 1 (da) or the mean of attempts 1–2 (db); continue while p(d)·(1 − v(x_inc, d)) ≥ λ. p and v are logistic fits on the dev split (`stopping/phase2_alloc_fit.json`); the cost includes 3 verifier passes per attempt used for d |

The literal rule "incumbent ≥ τ-quantile of the theorem's own attempts" is degenerate, because the incumbent is their
maximum. R1 and R2 are the pre-registered non-degenerate versions.

**Dev-frozen comparison** (audit item 2; `code/phase2_headline.py`, `results/B2_headline.{json,txt}`):

- Each family's parameter is chosen on combined dev to hit m ∈ {2, 4, 8}.
- The fixed-k comparator is the k-mixture with mean exactly m, so its weights are frozen by m.
- Every policy is shown at its realised test accuracy and attempts.
- Differences are paired theorem bootstraps, with the attempt difference (Δatt) stated.

**holdout250**

| m (dev target) | policy | test att | test acc | vs fixed-k @ m (Δatt) | vs dev-chosen FF (Δatt) | vs dev-chosen count (Δatt) |
|---|---|---|---|---|---|---|
| 2 | fixed-k mixture | 2.00 | 0.889 | — | | |
| 2 | ff | 1.61 | 0.904 | +0.016 [0.005, 0.028] (−0.39) | — | −0.001 (−0.09) |
| 2 | count (c = 1) | 1.70 | 0.905 | +0.017 [0.006, 0.030] (−0.30) | +0.001 (+0.09) | — |
| 2 | r1 | 2.63 | 0.916 | +0.027 [0.016, 0.040] (+0.63) | +0.012 [0.008, 0.016] (+1.03) | +0.010 [0.007, 0.014] (+0.93) |
| 2 | r2 | 3.53 | 0.918 | +0.029 [0.018, 0.042] (+1.53) | +0.014 [0.009, 0.019] (+1.92) | +0.012 [0.008, 0.017] (+1.83) |
| 2 | da | 1.70 | 0.905 | +0.016 (−0.30) | +0.001 (+0.09) | −0.001 (0.00) |
| 4 | fixed-k mixture | 4.00 | 0.908 | — | | |
| 4 | ff | 1.70 | 0.905 | −0.002 [−0.010, 0.007] (−2.30) | — | −0.013 (−1.23) |
| 4 | count (c = 2) | 2.93 | 0.918 | +0.010 [0.003, 0.018] (−1.07) | +0.013 [0.008, 0.017] (+1.23) | — |
| 4 | r1 | 2.81 | 0.918 | +0.010 [0.003, 0.018] (−1.19) | +0.012 (+1.11) | −0.000 [−0.002, 0.002] (−0.12) |
| 4 | r2 | 3.59 | 0.920 | +0.012 [0.006, 0.020] (−0.41) | +0.014 (+1.89) | +0.002 [−0.001, 0.004] (+0.66) |
| 4 | r3 | 9.33 | 0.923 | +0.015 (+5.33) | +0.018 (+7.63) | +0.005 [0.000, 0.010] (+6.40) |
| 4 | da | 3.42 | 0.918 | +0.010 [0.002, 0.020] (−0.58) | +0.012 (+1.72) | −0.000 [−0.004, 0.004] (+0.49) |
| 4 | db | 3.47 | 0.919 | +0.011 [0.005, 0.020] (−0.53) | +0.014 (+1.76) | +0.001 [−0.001, 0.004] (+0.53) |
| 8 | fixed-k mixture | 8.00 | 0.918 | — | | |
| 8 | count (c = 6) | 7.16 | 0.923 | +0.006 [0.002, 0.010] (−0.84) | +0.018 (+5.46) | — |
| 8 | r2 | 7.32 | 0.923 | +0.005 [0.001, 0.010] (−0.68) | +0.017 (+5.62) | −0.001 [−0.004, 0.002] (+0.16) |
| 8 | da | 6.65 | 0.924 | +0.006 [0.002, 0.012] (−1.35) | +0.019 (+4.95) | +0.001 [−0.001, 0.003] (−0.51) |
| 8 | db | 6.51 | 0.923 | +0.006 [0.001, 0.011] (−1.49) | +0.018 (+4.81) | −0.000 (−0.65) |

**textbook72**

| m (dev target) | policy | test att | test acc | vs fixed-k @ m (Δatt) | vs dev-chosen FF (Δatt) | vs dev-chosen count (Δatt) |
|---|---|---|---|---|---|---|
| 2 | fixed-k mixture | 2.00 | 0.464 | — | | |
| 2 | ff | 4.11 | 0.521 | +0.057 [0.034, 0.082] (+2.11) | — | −0.004 (−0.45) |
| 2 | count | 4.56 | 0.526 | +0.061 (+2.56) | +0.004 (+0.45) | — |
| 2 | r1 | 5.71 | 0.538 | +0.074 (+3.71) | +0.017 [0.008, 0.028] (+1.60) | +0.013 [0.004, 0.023] (+1.15) |
| 2 | r2 | 5.95 | 0.541 | +0.077 (+3.95) | +0.020 [0.011, 0.030] (+1.84) | +0.016 [0.007, 0.026] (+1.39) |
| 2 | da | 4.65 | 0.526 | +0.062 (+2.65) | +0.005 (+0.54) | +0.001 (+0.10) |
| 4 | fixed-k mixture | 4.00 | 0.510 | — | | |
| 4 | ff | 4.56 | 0.526 | +0.016 [−0.001, 0.035] (+0.56) | — | −0.020 (−2.32) |
| 4 | count (c = 2) | 6.87 | 0.546 | +0.036 [0.018, 0.058] (+2.87) | +0.020 [0.007, 0.036] (+2.32) | — |
| 4 | r1 | 7.02 | 0.545 | +0.035 (+3.02) | +0.019 (+2.47) | −0.001 [−0.010, 0.007] (+0.15) |
| 4 | r2 | 6.74 | 0.547 | +0.037 [0.019, 0.056] (+2.74) | +0.021 [0.009, 0.035] (+2.18) | +0.001 [−0.007, 0.009] (−0.13) |
| 4 | r3 | 12.09 | 0.561 | +0.051 (+8.09) | +0.036 (+7.53) | +0.016 [0.000, 0.033] (+5.21) |
| 4 | da | 11.86 | 0.564 | +0.054 (+7.86) | +0.039 (+7.30) | +0.019 [0.000, 0.039] (+4.98) |
| 4 | db | 10.93 | 0.562 | +0.052 (+6.93) | +0.036 (+6.38) | +0.016 [0.000, 0.034] (+4.06) |
| 8 | fixed-k mixture | 8.00 | 0.544 | — | | |
| 8 | count (c = 6) | 11.52 | 0.567 | +0.023 [0.007, 0.039] (+3.52) | +0.041 (+6.96) | — |
| 8 | r2 | 10.35 | 0.557 | +0.013 [0.000, 0.026] (+2.35) | +0.031 (+5.79) | −0.010 [−0.023, 0.002] (−1.17) |
| 8 | da | 13.92 | 0.566 | +0.022 (+5.92) | +0.041 (+9.36) | −0.001 [−0.009, 0.005] (+2.40) |

On textbook72 every dev-frozen comparison against fixed-k at m is cost-mismatched (the policies overspend), so it says
little about efficiency. The descriptive comparison against fixed-k at the realised cost (`B2_headline.txt`, last
column) is small for every policy:

- m = 4: count +0.009, r1 +0.007, r2 +0.011, da +0.007, db +0.009 (textbook72);
- m = 4: count +0.018, r2 +0.014, da +0.014 (holdout250).

**Where the attempts go** (m = 4 dev-chosen points; `results/B2_allocation_by_bucket.txt`; mean attempts by
per-theorem pass-rate bucket 0–0.05 / 0.05–0.25 / 0.25–0.5 / 0.5–0.9 / 0.9–1):

| set | policy | attempts by bucket | accuracy by bucket |
|---|---|---|---|
| holdout250 (14/7/10/52/167 theorems) | fixed-4 | 4 / 4 / 4 / 4 / 4 | 0.04 / 0.29 / 0.77 / 0.96 / 1.00 |
| | count | 13.6 / 5.8 / 4.5 / 2.3 / 2.0 | 0.10 / 0.50 / 0.87 / 0.95 / 1.00 |
| | r2 | 11.8 / 4.8 / 4.0 / 3.0 / 3.0 | 0.11 / 0.47 / 0.87 / 0.96 / 1.00 |
| | da | 15.3 / 8.3 / 7.1 / 3.2 / 2.1 | 0.10 / 0.56 / 0.89 / 0.94 / 1.00 |
| textbook72 (28/8/9/7/20) | fixed-4 | 4 / 4 / 4 / 4 / 4 | 0.02 / 0.40 / 0.72 / 0.97 / 0.99 |
| | count | 12.7 / 5.4 / 4.0 / 2.6 / 2.0 | 0.06 / 0.49 / 0.79 / 0.97 / 1.00 |
| | da | 15.3 / 14.5 / 13.6 / 8.6 / 6.4 | 0.05 / 0.59 / 0.85 / 0.99 / 0.99 |

Why da/db burn attempts on hopeless theorems: p(d) depends only on the first attempt's verifier score. It never
conditions on what the theorem's later attempts show, and the fitted slope on d is small (0.28 logit per unit; dev
fit in `stopping/phase2_alloc_fit.json`). So p stays above λ for any plausible d, and ten attempts with nothing
finished never lower the value of continuing.

Every adaptive policy saves attempts on easy theorems, as intended. The largest share of the budget, though, goes to
theorems with pass rate < 0.05, where nothing finishes and the policies run to N_max. The verifier-based difficulty
estimate does not stop that: da spends 15.3 attempts there. So the hoped-for move of attempts from easy and hopeless
theorems to intermediate ones only half happens (easy → intermediate yes; hopeless no). Accuracy gains come from the
0.05–0.5 buckets.

**Post-hoc give-up rule (dev only; not read on test).** Rule: stop after c finished attempts, or give up after g
attempts with none finished (`code/stopping_giveup.py`, `results/B2_giveup_choice*.json`). On dev, giving up is
dominated at matched attempts:

- c = 2, g = 16: 0.870 at 3.60 attempts, against c = 3, g = 3: 0.834 at 3.68.
- c = 1, g = 16: 0.847 at 2.10, against c = 2, g = 2: 0.808 at 2.33.

The dev choice was g = 16 (plain count-stop) for every target, so no test read was made. The late attempts on rarely
finishing theorems pay off more than extra finished attempts elsewhere. Spending on "hopeless" theorems is therefore
not the waste it looks like at the sampled pass rates.

`figures/B2_allocation_acc_vs_attempts.png` shows, per test set, accuracy against mean attempts (N_max 16) for:

- fixed-k uniform, first-finished, count-stop, R1, R2, R3, da, db;
- oracle pass@k and oracle stop-at-first-valid;
- stars at the dev-chosen points;
- the phase-1 secondary rules `completion_minstep` and `completion_loglik_sum` (dev-chosen points with test attempts
  ≤ 16) as red markers.

## 3. Commands

```bash
cd ~/runs/proof-selection; PY=~/venvs/proof-state/bin/python; S=<sprint folder>; export CUDA_VISIBLE_DEVICES=7 ...
for sd in s1 s2; do for p in <5 pools>; do $PY $S/code/score_loglik.py --pool pools_v2/$p.jsonl --ckpt ckpts/la_T1_best12_${sd}_r16.pt --tag $sd; done; done
$PY $S/code/make_xseed.py <5 stems>
$PY $S/code/auroc.py --pool pools_v2/<dev pool> --split dev --out $S/results/B2_auroc_<set>__dev.json     # dev, then test after freeze
$PY $S/code/auroc_paired.py pools_v2/<pool> loglik_mean_masked_siblings loglik_mean_masked [dev]
$PY $S/code/stopping.py --pool pools_v2/<pool> [--split dev] --scorers completion_loglik,completion_loglik_s1,completion_loglik_s2,completion_loglik_siblings,completion_loglik_xseed --grid stopping/tau_grid_dev_v.json --tag _xseed
$PY $S/code/stopping_phase2.py --fit                                   # dev split only -> phase2_alloc_fit.json, phase2_grids.json
$PY $S/code/stopping_phase2.py --pool pools_v2/<pool> [--split dev] [--selector siblings]
$PY $S/code/phase2_headline.py --dev stopping/{dev1108,rr600}*__dev__phase2.npz --test stopping/{holdout250,textbook72}*__phase2.npz --out $S/results/B2_headline.json
$PY $S/code/phase2_figs.py
```

## 4. Limitations

- **One generator.** Every judge scores s0's proofs. Whether s1 judges s1's own proofs as weakly as s0 judges s0's is
  untested (there are no s1 or s2 pools), so "self-judging is worse" is confounded with "s0 is a weaker judge".
- **N_max = 16.** All phase-2 policies are capped at 16. Count-stop, R2 and da can therefore not use the 64 test
  attempts, and the phase-1 64-cap curves are not directly comparable.
- **The allocation model.** p and v are logistic fits with two or three parameters, fitted and λ-swept on the same
  dev split (in-sample). The verifier was early-stopped on dev by Worker A. The siblings-selector da/db reuse the s0
  v-model (mis-specified).
- **Budgets on textbook72.** Every dev-chosen budget overshoots there. Count and rank rules overshoot less than the
  absolute likelihood threshold or da/db, but dev-frozen comparisons on textbook72 remain cost-mismatched.
- **R3 (agreement / majority vote).** It cannot reach budgets below about 9 attempts and was exploratory.
- **Statistics.** Pointwise CIs; 72 textbook theorems (34 with both classes among finished attempts).
- **Test exposure.** Before the phase-2 freeze, the label-free ND multiplicity of holdout250 was inspected (to check
  R3 was not degenerate). Phase-1 test results were known when phase 2 was designed: these test sets are no longer
  fresh for stopping-rule design.
