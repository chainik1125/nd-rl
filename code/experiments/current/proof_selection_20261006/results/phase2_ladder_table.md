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
