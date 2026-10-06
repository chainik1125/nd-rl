### textbook72__s0_r16__k64

| policy | m=1.5 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| fixed-k random pick | 0.407 | 0.407 | 0.407 | 0.407 | 0.407 |
| fixed-k completion pick | 0.435 | 0.459 | 0.495 | 0.512 | 0.526 |
| stop at first finished (ceiling 0.549 at 8.81 att) | 0.459 | 0.486 | 0.520 | 0.547 | - |
| fixed-k completion_loglik pick | 0.437 | 0.464 | 0.510 | 0.544 | 0.566 |
| adaptive completion_loglik-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.532 | 0.563 | 0.597 |
| fixed-k completion_loglik_sum pick | 0.438 | 0.466 | 0.508 | 0.533 | 0.550 |
| adaptive completion_loglik_sum-stop (frontier over N_max 2-64 x tau) | 0.460 | 0.490 | 0.530 | 0.558 | 0.577 |
| fixed-k completion_minstep pick | 0.438 | 0.466 | 0.507 | 0.533 | 0.552 |
| adaptive completion_minstep-stop (frontier over N_max 2-64 x tau) | 0.461 | 0.488 | 0.529 | 0.560 | 0.590 |
| fixed-k loglik_mean pick | 0.426 | 0.441 | 0.464 | 0.475 | 0.491 |
| adaptive loglik_mean-stop (frontier over N_max 2-64 x tau) | 0.446 | 0.461 | 0.491 | 0.501 | 0.502 |
| fixed-k completion_verifier_ens pick | 0.438 | 0.465 | 0.504 | 0.516 | 0.523 |
| adaptive completion_verifier_ens-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.486 | 0.522 | 0.549 | 0.556 |
| fixed-k completion_verifier_hist_ens pick | 0.437 | 0.462 | 0.499 | 0.512 | 0.517 |
| adaptive completion_verifier_hist_ens-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.524 | 0.551 | - |
| fixed-k shallow_lr pick | 0.438 | 0.466 | 0.508 | 0.535 | 0.547 |
| adaptive shallow_lr-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.489 | 0.528 | 0.555 | 0.579 |
| fixed-k shallow_lr_env pick | 0.436 | 0.461 | 0.496 | 0.509 | 0.508 |
| adaptive shallow_lr_env-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.525 | 0.551 | - |
| fixed-k verifier_ens pick | 0.438 | 0.465 | 0.504 | 0.516 | 0.523 |
| adaptive verifier_ens-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.486 | 0.522 | 0.549 | 0.557 |
| fixed-k verifier_hist_ens pick | 0.437 | 0.462 | 0.499 | 0.512 | 0.517 |
| adaptive verifier_hist_ens-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.524 | 0.550 | - |
| fixed-k verifier_hist_rank_s0 pick | 0.437 | 0.463 | 0.499 | 0.511 | 0.510 |
| adaptive verifier_hist_rank_s0-stop (frontier over N_max 2-64 x tau) | 0.437 | 0.463 | 0.499 | 0.511 | - |
| fixed-k verifier_hist_s0 pick | 0.436 | 0.462 | 0.500 | 0.513 | 0.516 |
| adaptive verifier_hist_s0-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.524 | 0.550 | - |
| fixed-k verifier_hist_s1 pick | 0.437 | 0.463 | 0.501 | 0.515 | 0.523 |
| adaptive verifier_hist_s1-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.523 | 0.554 | - |
| fixed-k verifier_hist_s2 pick | 0.436 | 0.461 | 0.497 | 0.512 | 0.520 |
| adaptive verifier_hist_s2-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.523 | 0.551 | - |
| fixed-k verifier_s0 pick | 0.438 | 0.466 | 0.507 | 0.520 | 0.524 |
| adaptive verifier_s0-stop (frontier over N_max 2-64 x tau) | 0.460 | 0.487 | 0.524 | 0.549 | - |
| fixed-k verifier_s1 pick | 0.438 | 0.466 | 0.502 | 0.516 | 0.522 |
| adaptive verifier_s1-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.486 | 0.520 | 0.548 | 0.556 |
| fixed-k verifier_s2 pick | 0.436 | 0.462 | 0.501 | 0.512 | 0.514 |
| adaptive verifier_s2-stop (frontier over N_max 2-64 x tau) | 0.459 | 0.487 | 0.523 | 0.553 | - |
| oracle pass@k (fixed-k) | 0.444 | 0.477 | 0.540 | 0.593 | 0.631 |
| oracle stop at first valid | 0.467 | 0.506 | 0.587 | 0.635 | 0.674 |

| adaptive vs fixed-k same selector (N_max 64, nearest grid point) | m | att | diff [95% CI] |
|---|---|---|---|
| completion N16 | 4 | 4.56 | +0.027 [+0.014, +0.044] |
| completion N64 | 8 | 8.81 | +0.034 [+0.008, +0.071] |
| completion nsweep | 2 | 1.88 | +0.028 [+0.017, +0.043] |
| completion nsweep | 4 | 3.95 | +0.026 [+0.014, +0.043] |
| completion nsweep | 8 | 7.97 | +0.034 [+0.009, +0.069] |
| completion_loglik N16 | 4 | 4.56 | +0.010 [-0.008, +0.030] |
| completion_loglik N16 | 8 | 7.98 | +0.013 [-0.005, +0.033] |
| completion_loglik N64 | 8 | 8.81 | +0.001 [-0.032, +0.044] |
| completion_loglik_sum N16 | 4 | 4.56 | +0.012 [-0.006, +0.033] |
| completion_loglik_sum N16 | 8 | 7.99 | +0.007 [-0.012, +0.025] |
| completion_loglik_sum N64 | 8 | 8.81 | +0.011 [-0.023, +0.053] |
| completion_minstep N16 | 4 | 4.56 | +0.014 [-0.005, +0.036] |
| completion_minstep N16 | 8 | 7.98 | +0.010 [-0.009, +0.026] |
| completion_minstep N64 | 8 | 8.81 | +0.012 [-0.023, +0.057] |
| loglik_mean N16 | 2 | 2.02 | +0.015 [+0.000, +0.031] |
| loglik_mean N16 | 4 | 4.01 | +0.020 [+0.001, +0.042] |
| loglik_mean N16 | 8 | 7.93 | +0.013 [-0.005, +0.032] |
| loglik_mean N64 | 2 | 1.90 | +0.015 [+0.001, +0.031] |
| loglik_mean N64 | 4 | 3.95 | +0.026 [+0.003, +0.061] |
| loglik_mean N64 | 8 | 7.63 | +0.026 [+0.000, +0.059] |
| completion_verifier_ens N16 | 2 | 2.01 | -0.014 [-0.029, +0.008] |
| completion_verifier_ens N16 | 4 | 3.74 | +0.010 [-0.006, +0.029] |
| completion_verifier_ens N16 | 8 | 7.97 | +0.002 [-0.018, +0.020] |
| completion_verifier_ens N64 | 2 | 2.02 | -0.018 [-0.041, +0.011] |
| completion_verifier_ens N64 | 4 | 3.72 | +0.003 [-0.017, +0.024] |
| completion_verifier_ens N64 | 8 | 8.80 | +0.032 [+0.000, +0.074] |
| completion_verifier_hist_ens N16 | 2 | 1.98 | -0.001 [-0.018, +0.023] |
| completion_verifier_hist_ens N16 | 4 | 3.99 | +0.017 [+0.001, +0.038] |
| completion_verifier_hist_ens N16 | 8 | 8.01 | +0.009 [-0.006, +0.027] |
| completion_verifier_hist_ens N64 | 2 | 2.01 | -0.002 [-0.021, +0.023] |
| completion_verifier_hist_ens N64 | 4 | 4.30 | +0.011 [-0.018, +0.050] |
| completion_verifier_hist_ens N64 | 8 | 8.81 | +0.037 [+0.005, +0.081] |
| shallow_lr N16 | 2 | 1.96 | +0.001 [-0.016, +0.021] |
| shallow_lr N16 | 4 | 3.97 | +0.005 [-0.014, +0.028] |
| shallow_lr N16 | 8 | 8.01 | +0.010 [-0.006, +0.029] |
| shallow_lr N64 | 2 | 1.97 | +0.001 [-0.017, +0.024] |
| shallow_lr N64 | 4 | 3.93 | -0.013 [-0.043, +0.027] |
| shallow_lr N64 | 8 | 8.09 | -0.007 [-0.034, +0.029] |
| shallow_lr_env N16 | 2 | 1.99 | -0.016 [-0.039, +0.011] |
| shallow_lr_env N16 | 4 | 3.43 | +0.002 [-0.022, +0.027] |
| shallow_lr_env N16 | 8 | 7.97 | +0.007 [-0.012, +0.026] |
| shallow_lr_env N64 | 4 | 4.06 | -0.039 [-0.069, +0.003] |
| shallow_lr_env N64 | 8 | 9.01 | +0.042 [+0.008, +0.085] |
| verifier_ens N16 | 2 | 2.01 | -0.014 [-0.029, +0.008] |
| verifier_ens N16 | 4 | 3.74 | +0.010 [-0.006, +0.029] |
| verifier_ens N16 | 8 | 7.99 | +0.002 [-0.018, +0.020] |
| verifier_ens N64 | 2 | 2.02 | -0.018 [-0.041, +0.011] |
| verifier_ens N64 | 4 | 3.72 | +0.003 [-0.017, +0.024] |
| verifier_ens N64 | 8 | 8.84 | +0.032 [+0.000, +0.074] |
| verifier_hist_ens N16 | 2 | 1.98 | -0.001 [-0.018, +0.023] |
| verifier_hist_ens N16 | 4 | 3.99 | +0.017 [+0.001, +0.038] |
| verifier_hist_ens N16 | 8 | 8.02 | +0.009 [-0.005, +0.027] |
| verifier_hist_ens N64 | 2 | 2.01 | -0.002 [-0.021, +0.023] |
| verifier_hist_ens N64 | 4 | 4.30 | +0.011 [-0.018, +0.050] |
| verifier_hist_ens N64 | 8 | 7.10 | +0.037 [+0.006, +0.080] |
| verifier_hist_s0 N16 | 2 | 2.06 | +0.002 [-0.017, +0.025] |
| verifier_hist_s0 N16 | 4 | 4.04 | +0.012 [-0.007, +0.035] |
| verifier_hist_s0 N16 | 8 | 7.99 | +0.013 [-0.004, +0.031] |
| verifier_hist_s0 N64 | 2 | 1.86 | +0.005 [-0.015, +0.026] |
| verifier_hist_s0 N64 | 4 | 4.13 | -0.004 [-0.030, +0.033] |
| verifier_hist_s0 N64 | 8 | 8.84 | +0.037 [+0.004, +0.080] |
| verifier_hist_s1 N16 | 2 | 2.04 | -0.003 [-0.021, +0.019] |
| verifier_hist_s1 N16 | 4 | 4.25 | +0.021 [+0.005, +0.040] |
| verifier_hist_s1 N16 | 8 | 7.96 | +0.012 [-0.004, +0.030] |
| verifier_hist_s1 N64 | 2 | 1.93 | -0.001 [-0.019, +0.020] |
| verifier_hist_s1 N64 | 4 | 3.92 | +0.009 [-0.018, +0.046] |
| verifier_hist_s1 N64 | 8 | 8.38 | +0.034 [+0.002, +0.075] |
| verifier_hist_s2 N16 | 2 | 1.95 | -0.001 [-0.018, +0.021] |
| verifier_hist_s2 N16 | 4 | 4.08 | +0.020 [+0.002, +0.042] |
| verifier_hist_s2 N16 | 8 | 7.99 | +0.018 [+0.001, +0.037] |
| verifier_hist_s2 N64 | 2 | 1.95 | -0.004 [-0.024, +0.022] |
| verifier_hist_s2 N64 | 4 | 3.81 | +0.003 [-0.022, +0.030] |
| verifier_hist_s2 N64 | 8 | 9.02 | +0.036 [+0.004, +0.080] |
| verifier_s0 N16 | 2 | 1.94 | -0.013 [-0.030, +0.009] |
| verifier_s0 N16 | 4 | 4.32 | +0.014 [-0.003, +0.034] |
| verifier_s0 N16 | 8 | 8.05 | -0.003 [-0.024, +0.017] |
| verifier_s0 N64 | 2 | 2.01 | -0.016 [-0.033, +0.008] |
| verifier_s0 N64 | 4 | 3.95 | +0.000 [-0.022, +0.025] |
| verifier_s0 N64 | 8 | 7.95 | +0.029 [-0.002, +0.070] |
| verifier_s1 N16 | 2 | 1.95 | -0.008 [-0.026, +0.016] |
| verifier_s1 N16 | 4 | 3.75 | +0.012 [-0.003, +0.030] |
| verifier_s1 N16 | 8 | 7.96 | +0.003 [-0.017, +0.021] |
| verifier_s1 N64 | 2 | 1.87 | -0.012 [-0.035, +0.014] |
| verifier_s1 N64 | 4 | 4.18 | +0.019 [-0.005, +0.051] |
| verifier_s2 N16 | 2 | 1.99 | -0.012 [-0.027, +0.009] |
| verifier_s2 N16 | 4 | 4.24 | +0.018 [+0.001, +0.039] |
| verifier_s2 N16 | 8 | 8.01 | +0.003 [-0.017, +0.021] |
| verifier_s2 N64 | 2 | 1.93 | -0.010 [-0.028, +0.011] |
| verifier_s2 N64 | 4 | 3.98 | -0.006 [-0.027, +0.019] |
| verifier_s2 N64 | 8 | 7.92 | +0.038 [+0.005, +0.079] |
