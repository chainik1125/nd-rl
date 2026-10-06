---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 5: absolute-confidence stopping with sibling judges

Pre-registered in `../logs/worker_B.md` ("Phase 5", 04:55 UTC) and frozen at 04:56 UTC (sha256 prefixes in the log);
the test was read once. Code: `code/phase5.py` (frozen), `code/phase5_matched.py` (descriptive cost-matched add-on,
logged after the read), `code/phase5_figs.py`. Results: `results/B5_dev_choice.json`, `results/B5_results.{json,txt}`,
`results/B5_matched.{json,txt}`. Figure: `figures/B5_headline.png`, which is B4 plus the sibling-threshold curve;
B4 is unchanged.

## Policies

All policies use N_max 16 and 50 orderings, and select the incumbent by the same judge.

| policy | stopping rule |
|---|---|
| T-sib / T-x | stop when the incumbent's judge loglik ≥ τ |
| C-sib / C-x | stop when the dev-fitted P(valid \| judge loglik, log steps) ≥ p* |
| count-sib | stop after c finished attempts (the comparator) |

- The judge is sib (mean of the two non-generator r16 seeds) or x (all three).
- τ and p* were chosen on the s0-generated dev split for m ∈ {2, 3, 4, 6, 8}, then applied unchanged to all three
  generators on test.

## Answers

1. **Primary (holdout250, m ≈ 2 and 4).** At m = 2 every rule is identical to first-finished (diff ≈ 0, all CIs
   include 0).
   - Any rule that returns a finished proof must wait for one, so it cannot spend less than first-finished. The
     "below count-stop's c = 1 point" region is unreachable by construction.
   - At m = 4 the frozen dev-matched comparison gives +0.014 [0.009, 0.021] for T-sib and +0.008 [0.003, 0.013] for
     C-sib.
   - That comparison is cost-mismatched: count-stop's dev choice for m = 4 is c = 2, which spends 2.79 attempts
     against T-sib's 3.85.
   - **At equal realised attempts (descriptive) the gain is small but positive.** Count-stop curve with
     siblings is linearly mixed between adjacent c; all generators pooled:

| set | policy | attempts | accuracy | count-stop at same attempts | diff [95 % CI] | gap to oracle stop-at-first-valid closed |
|---|---|---|---|---|---|---|
| holdout250 | C-sib (m=3) | 2.29 | 0.915 | 0.904 | +0.010 [0.006, 0.015] | 21 % |
| holdout250 | C-sib (m=4) | 2.93 | 0.922 | 0.916 | +0.007 [0.002, 0.012] | 18 % |
| holdout250 | T-sib (m=3) | 2.73 | 0.920 | 0.913 | +0.007 [0.003, 0.012] | 18 % |
| holdout250 | T-sib (m=4) | 3.85 | 0.929 | 0.923 | +0.006 [0.001, 0.011] | 20 % |
| holdout250 | T-sib (m=8) | 8.22 | 0.937 | 0.934 | +0.003 [−0.000, 0.005] | 14 % |
| textbook72 | T-sib (m=3) | 8.26 | 0.603 | 0.587 | +0.016 [0.003, 0.029] | 23 % |
| textbook72 | T-sib (m=4) | 9.83 | 0.611 | 0.600 | +0.011 [−0.001, 0.022] | 20 % |
| textbook72 | C-sib (m=4) | 8.44 | 0.600 | 0.588 | +0.012 [−0.002, 0.025] | 18 % |

   Oracle stop-at-first-valid at those attempts is 0.953 (holdout250) and 0.653 (textbook72). Per generator,
   s1 and s2 show the gain and s0 the least: holdout250 T-sib at m = 4 gives +0.007 / +0.015 / +0.020 for s0 / s1 /
   s2 in the frozen comparison.

2. **Budget transfer to textbook72.** Realised attempts ÷ dev target at m = 4:

| policy | realised attempts | ratio to target |
|---|---|---|
| T-sib | 9.83 | 2.5× |
| C-sib | 8.44 | 2.1× |
| T-x | 10.10 | 2.5× |
| count-stop (c = 2) | 6.08 | 1.5× |
| phase-1 self-likelihood threshold | 12.1 | 3.0× |

   The sibling thresholds transfer budget better than the self threshold, but worse than count-stop. Calibration
   on (loglik, log steps) helps somewhat.

3. **xseed vs siblings.** The two are indistinguishable: T-x and C-x lie within ±0.002 of T-sib and C-sib
   everywhere.

## Verdict

Absolute-confidence stopping with sibling judges is a modest, consistent improvement over count-stop at equal cost
on holdout250. It closes about 15–20 % of the gap to oracle stop-at-first-valid, and holds for each generator except
s0 at low budgets. On textbook72 it overshoots the dev budget about 2–2.5×, but at equal realised cost it is +0.011
to +0.016, with CIs that are marginal. About 80 % of the gap to "stop at the first valid proof" remains.

The purple curve in `figures/B5_headline.png` sweeps τ on test, so it is a curve reading, not a dev-chosen operating
point.

## Limitations

- Dev data exist only for the s0 generator, so thresholds for the s1 and s2 generators are transferred from it.
- The cost-matched comparison was added after the frozen read (descriptive).
- When the policy spends more than count-stop's c = 8 point, the count-stop comparator is c = 8.
- N_max is 16.
- Each generator has one pool.

## Robustness check: likelihood at the sampling temperature T = 0.8

The s0_r16 test pools were re-scored under judges s0 (self) and s1 (sibling) with logits / 0.8 before the
log-softmax (`score_loglik.py --temperature 0.8`, tags `s0_t08`, `s1_t08`). Masking and replay checks were the same
(0 errors). Metrics: within-theorem finished-only AUROC and k = 8 `completion_loglik` selection (200 shared
subsets). The diff column is sibling − self, as a paired theorem bootstrap (B = 2000). Output:
`results/B5_t08_check.txt`.

| set | T | AUROC self | AUROC s1 | AUROC diff [CI] | k=8 self | k=8 s1 | k=8 diff [CI] |
|---|---|---|---|---|---|---|---|
| holdout250 | 1.0 | 0.788 | 0.828 | +0.040 [0.002, 0.076] | 0.917 | 0.930 | +0.013 [0.004, 0.023] |
| holdout250 | 0.8 | 0.786 | 0.830 | +0.044 [0.006, 0.082] | 0.917 | 0.931 | +0.013 [0.004, 0.024] |
| textbook72 | 1.0 | 0.687 | 0.783 | +0.095 [0.020, 0.170] | 0.543 | 0.573 | +0.030 [0.007, 0.059] |
| textbook72 | 0.8 | 0.682 | 0.788 | +0.105 [0.027, 0.186] | 0.541 | 0.573 | +0.032 [0.008, 0.062] |

The self-judging penalty and the k = 8 selection ordering are unchanged at T = 0.8. If anything the penalty is
slightly larger. The phase-1 to phase-5 conclusions do not depend on scoring at T = 1.
