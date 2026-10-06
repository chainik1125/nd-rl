---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B phase 6: sampling-run variance (second draw of the s0_r16 test pools)

Draw 1 is the phase-1 pool (sampling seed 0). Draw 2 is a fresh pool from the same generator `la_T1_best12_s0_r16.pt`
with sampling seed 100, stems `holdout250__s0_r16__k64b` and `textbook72__s0_r16__k64b`. Settings are the same,
using Worker A's `collect_pools.py` unchanged.

- Replay: 16,000 / 16,000 and 4,608 / 4,608 attempts, 0 mismatches.
- Labels are in separate files; `MANIFEST.json` entries were added (phase 6).
- Draw 2 was scored under judges s0, s1 and s2 with the same masking (0 replay / done / ND / hash errors).

This phase only describes variance and makes no test choices (log line only). Code: `code/phase6.py`. Numbers:
`results/B6_draw_compare.json` and `results/B6_draw_compare.txt`.

Columns:

- **Diff:** draw 2 − draw 1, as a paired theorem bootstrap (B = 1000; same theorems, resampled jointly).
- **Half-width:** the theorem-bootstrap 95 % half-width of each draw's own estimate. This is what the earlier phases
  reported as uncertainty.
- **Within-theorem AUROC:** its paired diff uses only theorems with both classes among finished attempts in both
  draws, so it can differ from draw2 − draw1 of the means.

| set | quantity | draw 1 | draw 2 | draw 2 − draw 1 [95 % CI] | theorem-bootstrap half-width (draw 1 / draw 2) |
|---|---|---|---|---|---|
| holdout250 | per-attempt validity | 0.836 | 0.832 | -0.004 [-0.009, +0.001] | 0.034 / 0.034 |
| holdout250 | pass@8 (unbiased) | 0.939 | 0.938 | -0.001 [-0.006, +0.003] | 0.028 / 0.028 |
| holdout250 | pass@64 | 0.960 | 0.956 | -0.004 [-0.012, +0.000] | 0.026 / 0.026 |
| holdout250 | k=8 random pick | 0.836 | 0.832 | -0.004 [-0.009, +0.001] | 0.034 / 0.034 |
| holdout250 | k=8 completion pick | 0.898 | 0.895 | -0.003 [-0.008, +0.002] | 0.031 / 0.030 |
| holdout250 | k=8 completion_loglik (self) | 0.917 | 0.915 | -0.003 [-0.010, +0.003] | 0.032 / 0.032 |
| holdout250 | k=8 completion_loglik (siblings) | 0.928 | 0.927 | -0.001 [-0.007, +0.005] | 0.030 / 0.031 |
| holdout250 | k=8 oracle | 0.939 | 0.939 | -0.001 [-0.006, +0.003] | 0.028 / 0.028 |
| holdout250 | within-theorem AUROC, self | 0.788 | 0.813 | +0.012 [-0.019, +0.041] | 0.040 / 0.033 |
| holdout250 | within-theorem AUROC, siblings | 0.829 | 0.848 | +0.027 [+0.001, +0.052] | 0.030 / 0.026 |
| holdout250 | count-stop c=2 + siblings: attempts | 2.934 | 2.944 | +0.010 [-0.051, +0.090] | 0.360 / 0.374 |
| holdout250 | count-stop c=2 + siblings: accuracy | 0.920 | 0.915 | -0.004 [-0.011, +0.002] | 0.029 / 0.030 |
| holdout250 | count-stop c=6 + siblings: attempts | 7.169 | 7.148 | -0.021 [-0.083, +0.037] | 0.340 / 0.322 |
| holdout250 | count-stop c=6 + siblings: accuracy | 0.933 | 0.930 | -0.003 [-0.010, +0.003] | 0.028 / 0.030 |
| textbook72 | per-attempt validity | 0.407 | 0.404 | -0.003 [-0.013, +0.007] | 0.093 / 0.092 |
| textbook72 | pass@8 (unbiased) | 0.597 | 0.590 | -0.007 [-0.022, +0.006] | 0.105 / 0.103 |
| textbook72 | pass@64 | 0.694 | 0.667 | -0.028 [-0.083, +0.028] | 0.104 / 0.104 |
| textbook72 | k=8 random pick | 0.407 | 0.404 | -0.003 [-0.013, +0.007] | 0.093 / 0.092 |
| textbook72 | k=8 completion pick | 0.516 | 0.505 | -0.011 [-0.029, +0.005] | 0.100 / 0.099 |
| textbook72 | k=8 completion_loglik (self) | 0.543 | 0.542 | -0.001 [-0.020, +0.018] | 0.104 / 0.102 |
| textbook72 | k=8 completion_loglik (siblings) | 0.573 | 0.555 | -0.018 [-0.044, +0.002] | 0.101 / 0.101 |
| textbook72 | k=8 oracle | 0.598 | 0.588 | -0.010 [-0.025, +0.005] | 0.106 / 0.104 |
| textbook72 | within-theorem AUROC, self | 0.687 | 0.764 | +0.039 [-0.025, +0.098] | 0.084 / 0.072 |
| textbook72 | within-theorem AUROC, siblings | 0.782 | 0.817 | -0.020 [-0.069, +0.023] | 0.085 / 0.062 |
| textbook72 | count-stop c=2 + siblings: attempts | 6.833 | 6.788 | -0.045 [-0.376, +0.298] | 1.215 / 1.202 |
| textbook72 | count-stop c=2 + siblings: accuracy | 0.565 | 0.558 | -0.007 [-0.031, +0.014] | 0.100 / 0.100 |
| textbook72 | count-stop c=6 + siblings: attempts | 11.482 | 11.407 | -0.076 [-0.235, +0.079] | 1.007 / 0.993 |
| textbook72 | count-stop c=6 + siblings: accuracy | 0.601 | 0.582 | -0.019 [-0.048, +0.003] | 0.102 / 0.103 |

**Does the sibling-over-self gain replicate in the second draw?** Sibling − self, paired theorem bootstrap within each
draw:

| set | draw | within-theorem AUROC | k=8 selection |
|---|---|---|---|
| holdout250 | draw1 | +0.041 [+0.007, +0.076] | +0.010 [+0.001, +0.021] |
| holdout250 | draw2 | +0.036 [+0.003, +0.068] | +0.013 [+0.002, +0.025] |
| textbook72 | draw1 | +0.095 [+0.020, +0.173] | +0.030 [+0.010, +0.055] |
| textbook72 | draw2 | +0.053 [+0.002, +0.100] | +0.012 [-0.001, +0.027] |

## Reading

- **Sampling noise is smaller than theorem noise.** For every accuracy-type number (validity, pass@k, k = 8 selection,
  count-stop accuracy), the draw-to-draw difference is 0.001–0.004 on holdout250 and 0.001–0.028 on textbook72. The
  theorem-bootstrap half-widths are about 0.03 and about 0.10, so the existing intervals already cover sampling noise.
- **Except for within-theorem AUROC.** It is the noisiest quantity, because it rests on only the theorems that happen
  to have both valid and invalid finished attempts:
  - the self AUROC moves from 0.687 to 0.764 on textbook72 (paired diff +0.039 [−0.025, 0.098]);
  - the sibling AUROC moves +0.027 [0.001, 0.052] on holdout250.
  - Its absolute level should be quoted with that caveat.
- **Count-stop budgets are reproducible.** Attempts differ by at most 0.08 between draws.
- **The sibling-over-self gain replicates.**
  - holdout250: AUROC +0.041 → +0.036; k = 8 selection +0.010 → +0.013. All CIs exclude 0.
  - textbook72: AUROC +0.095 → +0.053, CI excludes 0. k = 8 selection +0.030 → +0.012 [−0.001, 0.027].
  - The textbook72 effect is smaller in draw 2, and its k = 8 CI now touches 0.
