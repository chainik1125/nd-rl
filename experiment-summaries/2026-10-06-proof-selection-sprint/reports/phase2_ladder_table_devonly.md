| model | finished-invalid train examples (in pair theorems) | pair theorems | best step | within-thm finished AUROC: dev (orig pools) |
|---|---|---|---|---|
| verifier2_r25 | 799 (703) | 455 | 100 | 0.642 |
| verifier2_r50 | 1598 (1423) | 756 | 100 | 0.633 |
| verifier2_r100 | 3195 (2862) | 1227 | 300 | 0.657 |
| verifier2_rall_s0 | 10676 (9499) | 1492 | 400 | 0.627 |
| verifier2_rall_s1 | 10676 (9499) | 1492 | 500 | 0.662 |
| verifier2_rall_s2 | 10676 (9499) | 1492 | 400 | 0.657 |
| verifier2_ens (mean of rall s0-s2) | 10676 (9499) | 1492 |  | 0.665 |
| verifier2_ll (rall + loglik feature) | 10676 (9499) | 1492 | 1400 | 0.682 |
| *loglik_mean_masked* (reference) | | | | 0.776 |
| *verifier_ens* (reference) | | | | 0.587 |
| *verifier_hist_ens* (reference) | | | | 0.622 |

n theorems with both classes among finished attempts: dev (orig pools) 185
