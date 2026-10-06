### holdout250__s0_r16__k64

| policy | m=1.5 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| fixed-k random pick | 0.836 | 0.836 | 0.836 | 0.836 | 0.836 |
| fixed-k completion pick | 0.861 | 0.880 | 0.894 | 0.901 | 0.905 |
| stop at first finished (ceiling 0.911 at 2.62 att) | 0.903 | 0.908 | - | - | - |
| fixed-k completion_loglik pick | 0.865 | 0.889 | 0.908 | 0.918 | 0.927 |
| adaptive completion_loglik-stop (frontier over N_max 2-64 x tau) | 0.905 | 0.913 | 0.922 | 0.929 | 0.936 |
| fixed-k completion_loglik_sum pick | 0.865 | 0.888 | 0.908 | 0.917 | 0.923 |
| adaptive completion_loglik_sum-stop (frontier over N_max 2-64 x tau) | 0.907 | 0.913 | 0.924 | 0.930 | 0.933 |
| fixed-k completion_minstep pick | 0.864 | 0.886 | 0.905 | 0.912 | 0.916 |
| adaptive completion_minstep-stop (frontier over N_max 2-64 x tau) | 0.907 | 0.914 | 0.921 | 0.924 | 0.927 |
| fixed-k loglik_mean pick | 0.860 | 0.879 | 0.893 | 0.899 | 0.904 |
| adaptive loglik_mean-stop (frontier over N_max 2-64 x tau) | 0.878 | 0.885 | 0.896 | 0.902 | 0.906 |
| fixed-k completion_verifier_ens pick | 0.862 | 0.882 | 0.899 | 0.907 | 0.913 |
| adaptive completion_verifier_ens-stop (frontier over N_max 2-64 x tau) | 0.905 | 0.911 | 0.919 | 0.920 | 0.921 |
| fixed-k completion_verifier_hist_ens pick | 0.864 | 0.885 | 0.903 | 0.908 | 0.906 |
| adaptive completion_verifier_hist_ens-stop (frontier over N_max 2-64 x tau) | 0.905 | 0.912 | 0.917 | - | - |
| fixed-k shallow_lr pick | 0.864 | 0.887 | 0.905 | 0.913 | 0.919 |
| adaptive shallow_lr-stop (frontier over N_max 2-64 x tau) | 0.906 | 0.912 | 0.920 | 0.923 | 0.925 |
| fixed-k shallow_lr_env pick | 0.862 | 0.883 | 0.896 | 0.901 | 0.899 |
| adaptive shallow_lr_env-stop (frontier over N_max 2-64 x tau) | 0.903 | 0.909 | - | - | - |
| fixed-k verifier_ens pick | 0.862 | 0.882 | 0.898 | 0.906 | 0.913 |
| adaptive verifier_ens-stop (frontier over N_max 2-64 x tau) | 0.904 | 0.910 | 0.918 | 0.920 | 0.920 |
| fixed-k verifier_hist_ens pick | 0.863 | 0.885 | 0.903 | 0.907 | 0.905 |
| adaptive verifier_hist_ens-stop (frontier over N_max 2-64 x tau) | 0.905 | 0.912 | 0.917 | - | - |
| fixed-k verifier_hist_rank_s0 pick | 0.863 | 0.885 | 0.902 | 0.906 | 0.902 |
| adaptive verifier_hist_rank_s0-stop (frontier over N_max 2-64 x tau) | 0.863 | 0.885 | 0.902 | - | - |
| fixed-k verifier_hist_s0 pick | 0.863 | 0.885 | 0.902 | 0.906 | 0.906 |
| adaptive verifier_hist_s0-stop (frontier over N_max 2-64 x tau) | 0.904 | 0.911 | 0.918 | - | - |
| fixed-k verifier_hist_s1 pick | 0.863 | 0.885 | 0.902 | 0.909 | 0.909 |
| adaptive verifier_hist_s1-stop (frontier over N_max 2-64 x tau) | 0.904 | 0.910 | 0.917 | - | - |
| fixed-k verifier_hist_s2 pick | 0.863 | 0.884 | 0.901 | 0.907 | 0.905 |
| adaptive verifier_hist_s2-stop (frontier over N_max 2-64 x tau) | 0.904 | 0.911 | 0.917 | - | - |
| fixed-k verifier_s0 pick | 0.862 | 0.882 | 0.896 | 0.903 | 0.907 |
| adaptive verifier_s0-stop (frontier over N_max 2-64 x tau) | 0.903 | 0.908 | 0.913 | 0.914 | 0.915 |
| fixed-k verifier_s1 pick | 0.862 | 0.883 | 0.901 | 0.907 | 0.910 |
| adaptive verifier_s1-stop (frontier over N_max 2-64 x tau) | 0.903 | 0.910 | 0.917 | - | - |
| fixed-k verifier_s2 pick | 0.861 | 0.881 | 0.895 | 0.902 | 0.907 |
| adaptive verifier_s2-stop (frontier over N_max 2-64 x tau) | 0.903 | 0.912 | 0.920 | - | - |
| oracle pass@k (fixed-k) | 0.871 | 0.900 | 0.925 | 0.939 | 0.950 |
| oracle stop at first valid | 0.934 | 0.948 | 0.960 | - | - |

| adaptive vs fixed-k same selector (N_max 64, nearest grid point) | m | att | diff [95% CI] |
|---|---|---|---|
| completion N16 | 2 | 1.70 | +0.037 [+0.022, +0.053] |
| completion nsweep | 2 | 1.99 | +0.028 [+0.018, +0.048] |
| completion_loglik N16 | 2 | 2.00 | +0.023 [+0.012, +0.043] |
| completion_loglik N16 | 4 | 4.00 | +0.011 [+0.003, +0.020] |
| completion_loglik N16 | 8 | 7.99 | +0.008 [+0.002, +0.013] |
| completion_loglik N64 | 4 | 4.01 | +0.012 [+0.000, +0.029] |
| completion_loglik N64 | 8 | 7.99 | +0.009 [-0.001, +0.022] |
| completion_loglik_sum N16 | 2 | 2.00 | +0.024 [+0.013, +0.042] |
| completion_loglik_sum N16 | 4 | 3.99 | +0.010 [+0.002, +0.019] |
| completion_loglik_sum N16 | 8 | 7.92 | +0.007 [+0.002, +0.013] |
| completion_loglik_sum N64 | 4 | 4.01 | +0.010 [-0.001, +0.027] |
| completion_loglik_sum N64 | 8 | 8.05 | +0.009 [-0.004, +0.023] |
| completion_minstep N16 | 2 | 1.99 | +0.029 [+0.017, +0.047] |
| completion_minstep N16 | 4 | 3.98 | +0.012 [+0.003, +0.020] |
| completion_minstep N16 | 8 | 8.03 | +0.004 [-0.002, +0.010] |
| completion_minstep N64 | 4 | 4.00 | +0.015 [+0.004, +0.030] |
| completion_minstep N64 | 8 | 8.00 | +0.011 [-0.001, +0.023] |
| loglik_mean N16 | 2 | 2.02 | +0.002 [-0.005, +0.013] |
| loglik_mean N16 | 4 | 4.00 | +0.002 [-0.004, +0.010] |
| loglik_mean N16 | 8 | 8.07 | +0.002 [-0.003, +0.007] |
| loglik_mean N64 | 2 | 1.98 | -0.006 [-0.017, +0.018] |
| loglik_mean N64 | 4 | 4.00 | -0.008 [-0.015, -0.000] |
| loglik_mean N64 | 8 | 7.97 | -0.000 [-0.011, +0.012] |
| completion_verifier_ens N16 | 2 | 2.00 | +0.029 [+0.018, +0.048] |
| completion_verifier_ens N16 | 4 | 3.96 | +0.012 [+0.004, +0.022] |
| completion_verifier_ens N16 | 8 | 8.00 | +0.005 [-0.001, +0.011] |
| completion_verifier_ens N64 | 2 | 1.84 | +0.027 [+0.011, +0.048] |
| completion_verifier_ens N64 | 4 | 3.99 | +0.018 [+0.006, +0.034] |
| completion_verifier_ens N64 | 8 | 7.95 | +0.003 [-0.011, +0.020] |
| completion_verifier_hist_ens N16 | 2 | 2.00 | +0.027 [+0.016, +0.046] |
| completion_verifier_hist_ens N16 | 4 | 3.98 | +0.002 [-0.007, +0.014] |
| completion_verifier_hist_ens N16 | 8 | 7.99 | -0.001 [-0.008, +0.007] |
| completion_verifier_hist_ens N64 | 2 | 1.98 | +0.018 [+0.006, +0.041] |
| completion_verifier_hist_ens N64 | 4 | 4.00 | +0.014 [+0.004, +0.033] |
| completion_verifier_hist_ens N64 | 8 | 8.04 | +0.006 [-0.010, +0.021] |
| shallow_lr N16 | 2 | 2.00 | +0.023 [+0.013, +0.040] |
| shallow_lr N16 | 4 | 4.02 | +0.009 [+0.001, +0.019] |
| shallow_lr N16 | 8 | 7.94 | +0.003 [-0.004, +0.009] |
| shallow_lr N64 | 2 | 1.97 | +0.001 [-0.011, +0.031] |
| shallow_lr N64 | 4 | 3.99 | +0.013 [+0.003, +0.029] |
| shallow_lr N64 | 8 | 7.98 | +0.009 [-0.002, +0.023] |
| shallow_lr_env N16 | 2 | 2.00 | +0.024 [+0.014, +0.040] |
| shallow_lr_env N16 | 4 | 4.05 | +0.008 [+0.002, +0.016] |
| shallow_lr_env N16 | 8 | 8.07 | +0.001 [-0.006, +0.008] |
| shallow_lr_env N64 | 2 | 1.86 | +0.017 [+0.001, +0.043] |
| shallow_lr_env N64 | 4 | 4.00 | +0.018 [+0.008, +0.033] |
| shallow_lr_env N64 | 8 | 8.06 | +0.007 [-0.007, +0.021] |
| verifier_ens N16 | 2 | 2.00 | +0.028 [+0.018, +0.047] |
| verifier_ens N16 | 4 | 3.97 | +0.011 [+0.003, +0.021] |
| verifier_ens N16 | 8 | 8.00 | +0.005 [-0.002, +0.011] |
| verifier_ens N64 | 2 | 1.84 | +0.027 [+0.011, +0.048] |
| verifier_ens N64 | 4 | 3.97 | +0.017 [+0.005, +0.034] |
| verifier_ens N64 | 8 | 8.01 | +0.005 [-0.009, +0.021] |
| verifier_hist_ens N16 | 2 | 2.00 | +0.026 [+0.015, +0.045] |
| verifier_hist_ens N16 | 4 | 3.98 | +0.002 [-0.007, +0.014] |
| verifier_hist_ens N16 | 8 | 7.99 | -0.001 [-0.009, +0.007] |
| verifier_hist_ens N64 | 2 | 1.98 | +0.019 [+0.007, +0.042] |
| verifier_hist_ens N64 | 4 | 3.97 | +0.014 [+0.004, +0.032] |
| verifier_hist_ens N64 | 8 | 7.98 | +0.005 [-0.010, +0.022] |
| verifier_hist_s0 N16 | 2 | 2.00 | +0.025 [+0.014, +0.044] |
| verifier_hist_s0 N16 | 4 | 4.05 | +0.005 [-0.004, +0.017] |
| verifier_hist_s0 N16 | 8 | 8.00 | +0.001 [-0.007, +0.008] |
| verifier_hist_s0 N64 | 2 | 2.02 | +0.019 [+0.007, +0.041] |
| verifier_hist_s0 N64 | 4 | 3.98 | +0.010 [-0.000, +0.027] |
| verifier_hist_s0 N64 | 8 | 7.83 | +0.005 [-0.010, +0.021] |
| verifier_hist_s1 N16 | 2 | 2.01 | +0.024 [+0.013, +0.042] |
| verifier_hist_s1 N16 | 4 | 3.99 | +0.008 [-0.002, +0.021] |
| verifier_hist_s1 N16 | 8 | 7.89 | +0.001 [-0.006, +0.009] |
| verifier_hist_s1 N64 | 2 | 1.97 | +0.020 [+0.008, +0.042] |
| verifier_hist_s1 N64 | 4 | 3.82 | +0.014 [+0.002, +0.031] |
| verifier_hist_s1 N64 | 8 | 8.09 | +0.001 [-0.014, +0.017] |
| verifier_hist_s2 N16 | 2 | 2.00 | +0.027 [+0.017, +0.046] |
| verifier_hist_s2 N16 | 4 | 4.00 | +0.004 [-0.006, +0.015] |
| verifier_hist_s2 N16 | 8 | 7.99 | +0.000 [-0.007, +0.008] |
| verifier_hist_s2 N64 | 2 | 2.05 | +0.021 [+0.009, +0.042] |
| verifier_hist_s2 N64 | 4 | 4.00 | +0.016 [+0.006, +0.032] |
| verifier_hist_s2 N64 | 8 | 7.97 | +0.007 [-0.005, +0.021] |
| verifier_s0 N16 | 2 | 2.00 | +0.024 [+0.014, +0.043] |
| verifier_s0 N16 | 4 | 3.98 | +0.010 [+0.002, +0.018] |
| verifier_s0 N16 | 8 | 7.96 | +0.002 [-0.004, +0.008] |
| verifier_s0 N64 | 2 | 1.82 | +0.027 [+0.011, +0.049] |
| verifier_s0 N64 | 4 | 4.08 | +0.017 [+0.004, +0.034] |
| verifier_s0 N64 | 8 | 7.94 | +0.005 [-0.009, +0.019] |
| verifier_s1 N16 | 2 | 2.00 | +0.027 [+0.016, +0.046] |
| verifier_s1 N16 | 4 | 4.04 | +0.010 [+0.002, +0.020] |
| verifier_s1 N16 | 8 | 8.05 | +0.005 [-0.001, +0.012] |
| verifier_s1 N64 | 2 | 1.84 | +0.024 [+0.009, +0.045] |
| verifier_s1 N64 | 4 | 4.07 | +0.015 [+0.003, +0.032] |
| verifier_s1 N64 | 8 | 8.00 | +0.014 [+0.002, +0.026] |
| verifier_s2 N16 | 2 | 2.00 | +0.028 [+0.018, +0.046] |
| verifier_s2 N16 | 4 | 4.02 | +0.014 [+0.006, +0.025] |
| verifier_s2 N16 | 8 | 8.04 | +0.004 [-0.002, +0.011] |
| verifier_s2 N64 | 2 | 1.87 | +0.029 [+0.014, +0.049] |
| verifier_s2 N64 | 4 | 3.99 | +0.020 [+0.008, +0.037] |
| verifier_s2 N64 | 8 | 8.02 | +0.009 [-0.005, +0.026] |
