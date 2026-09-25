---
written_on: 2026-09-25
---

# Length generalisation: report

Generated 2026-09-25 03:46 UTC by `analyse.py` from `/scratch/cowifornia/nd-rl/ab3/lengthexp/eval`. k = 32 samples per theorem at T = 0.8; pass@k are unbiased estimates from n_ok of k, averaged over theorems.

Evaluated checkpoints:

| label | arm | seed | units | checkpoint sha256 |
|---|---|---|---|---|
| A_s101 | A | 101 | 22 | 6d0205019ec6 |
| A_s102 | A | 102 | 22 | 020d96e4f4be |
| A_s103 | A | 103 | 22 | 8525671d72d9 |
| B_s101 | B | 101 | 22 | 70a0f06a28d4 |
| B_s102 | B | 102 | 22 | 2530c356952f |
| B_s103 | B | 103 | 22 | 03329e788005 |
| C_s101 | C | 101 | 22 | 9ec083a96666 |
| C_s102 | C | 102 | 22 | f316a3d7a3c3 |
| C_s103 | C | 103 | 22 | 8ebe0f591744 |
| D_s101 | D | 101 | 22 | c95556a27ebb |
| D_s102 | D | 102 | 22 | 7a985c2c1266 |
| D_s103 | D | 103 | 22 | 369a4b9664b3 |
| pretrain | pretrain | - | 22 | f4409e4f82d8 |

Test sets (std): L6 n=300 (120 textbook, 7 schemata), L7 n=300 (120 textbook, 5 schemata), L8 n=300 (120 textbook, 14 schemata), L9 n=300 (120 textbook, 13 schemata), L10 n=300 (120 textbook, 12 schemata), L11 n=300 (120 textbook, 6 schemata), L12 n=300 (120 textbook, 5 schemata), L13 n=300 (102 textbook, 2 schemata), L14 n=158 (114 textbook, 1 schemata), L15 n=300 (0 textbook, 0 schemata), L16 n=300 (0 textbook, 0 schemata), L17 n=300 (0 textbook, 0 schemata), L18 n=300 (0 textbook, 0 schemata).

## Headline

| arm (train max) | runs | pass@1 at its own training max (d = 0) | slope over d >= 1 [95% t-interval] | slope over L >= 12 [95% t-interval] | pass@1 pooled over L >= 12 [seed range] |
|---|---|---|---|---|---|
| pretrain (6) | 1 | 0.1578 | -3.57 | - | 0.0000 |
| A (7) | 3 | 0.5048 [0.4684, 0.5324] | - | - | 0.0000 [0.0000, 0.0000] |
| B (9) | 3 | 0.1882 [0.1840, 0.1929] | -5.66 | - | 0.0000 [0.0000, 0.0000] |
| C (11) | 3 | 0.1274 [0.1263, 0.1293] | -2.99 [-4.99, -0.99] | -2.99 [-4.99, -0.99] | 0.0055 [0.0051, 0.0061] |
| D (11) | 3 | 0.1199 [0.1155, 0.1231] | -3.32 [-8.52, 1.89] | -3.32 [-8.52, 1.89] | 0.0055 [0.0048, 0.0066] |

## 1. pass@k by arm and length

Mean over seeds, [min, max] over seeds in brackets.

**pass@1**

| arm (train max) | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain (6) | 0.158 | 0.031 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A (7) | 0.589 [0.558, 0.632] | 0.505 [0.468, 0.532] | 0.043 [0.025, 0.074] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| B (9) | 0.610 [0.590, 0.626] | 0.435 [0.407, 0.469] | 0.246 [0.214, 0.273] | 0.188 [0.184, 0.193] | 0.028 [0.021, 0.034] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C (11) | 0.562 [0.555, 0.569] | 0.364 [0.360, 0.368] | 0.191 [0.173, 0.213] | 0.171 [0.161, 0.184] | 0.109 [0.099, 0.118] | 0.127 [0.126, 0.129] | 0.034 [0.031, 0.039] | 0.002 [0.001, 0.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| D (11) | 0.562 [0.551, 0.569] | 0.353 [0.338, 0.382] | 0.196 [0.175, 0.209] | 0.184 [0.176, 0.190] | 0.106 [0.098, 0.111] | 0.120 [0.116, 0.123] | 0.033 [0.029, 0.038] | 0.003 [0.000, 0.005] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**pass@4**

| arm (train max) | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain (6) | 0.259 | 0.069 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A (7) | 0.736 [0.709, 0.771] | 0.591 [0.544, 0.623] | 0.077 [0.049, 0.124] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| B (9) | 0.724 [0.710, 0.745] | 0.521 [0.491, 0.556] | 0.305 [0.269, 0.335] | 0.244 [0.240, 0.246] | 0.049 [0.037, 0.057] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C (11) | 0.692 [0.684, 0.699] | 0.448 [0.444, 0.451] | 0.249 [0.220, 0.287] | 0.219 [0.207, 0.229] | 0.150 [0.126, 0.166] | 0.160 [0.157, 0.162] | 0.050 [0.047, 0.054] | 0.005 [0.003, 0.006] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| D (11) | 0.692 [0.672, 0.719] | 0.432 [0.409, 0.467] | 0.258 [0.223, 0.276] | 0.232 [0.227, 0.239] | 0.144 [0.134, 0.153] | 0.150 [0.141, 0.158] | 0.050 [0.044, 0.060] | 0.007 [0.000, 0.011] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**pass@32**

| arm (train max) | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain (6) | 0.377 | 0.120 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A (7) | 0.840 [0.833, 0.847] | 0.653 [0.600, 0.690] | 0.133 [0.100, 0.187] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| B (9) | 0.797 [0.783, 0.820] | 0.597 [0.557, 0.643] | 0.363 [0.337, 0.380] | 0.292 [0.290, 0.293] | 0.084 [0.067, 0.100] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C (11) | 0.792 [0.790, 0.797] | 0.534 [0.527, 0.547] | 0.293 [0.260, 0.330] | 0.267 [0.257, 0.273] | 0.190 [0.170, 0.200] | 0.187 [0.177, 0.197] | 0.064 [0.063, 0.067] | 0.012 [0.007, 0.017] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| D (11) | 0.786 [0.747, 0.817] | 0.501 [0.473, 0.530] | 0.307 [0.263, 0.337] | 0.279 [0.267, 0.287] | 0.177 [0.163, 0.190] | 0.176 [0.167, 0.190] | 0.070 [0.057, 0.090] | 0.017 [0.003, 0.027] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## 2. pass@1 against distance beyond the training length

![main](fig_main.png)

![main, generator only](fig_main_gen.png)

Reading: if the curves overlap in d, training on longer proofs only moved the edge; if arms with a larger training max have flatter curves (a smaller log-odds drop per extra line), they generalise more. Slopes are from a grouped-binomial logistic fit per run (log-odds of pass@1 per extra line); the t-interval is over seeds (3 seeds: t = 4.30), the bootstrap SE resamples theorems within a run. The "common" window uses the same absolute lengths for every arm, so differences in curvature over different d ranges cannot masquerade as slope differences.

**Window: own (d >= 1).**

| arm (train max) | lengths used | slope per run (log-odds per line) | mean | t-interval over seeds (95%) | mean within-run bootstrap SE |
|---|---|---|---|---|---|
| pretrain (6) | 7-18 | -3.57 | -3.57 | - | 1.02 |
| A (7) | 8-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| B (9) | 10-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), -5.66, n/a (successes at fewer than 2 lengths (slope not estimable)) | -5.66 | - | 0.48 |
| C (11) | 12-18 | -2.42, -2.64, -3.91 | -2.99 | [-4.99, -0.99] | 0.77 |
| D (11) | 12-18 | -5.73, -2.02, -2.19 | -3.32 | [-8.52, 1.89] | 0.62 |

- trend of slope with training max: +1.336 log-odds/line per extra training line, exact permutation p = 0.2500 (4 relabelings of 4 runs); omnibus F = 8.28, permutation p = 0.2500
- B-C: difference -2.673, Welch p = -, exact permutation p = 0.2500 (smallest possible 0.25)
- D vs C (not part of the A/B/C length tests): slope difference D - C -0.324, Welch p = 0.8211, exact permutation p = 1.0000 (3 vs 3 runs)

**Window: common (L >= 12).**

| arm (train max) | lengths used | slope per run (log-odds per line) | mean | t-interval over seeds (95%) | mean within-run bootstrap SE |
|---|---|---|---|---|---|
| pretrain (6) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| A (7) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| B (9) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| C (11) | 12-18 | -2.42, -2.64, -3.91 | -2.99 | [-4.99, -0.99] | 0.79 |
| D (11) | 12-18 | -5.73, -2.02, -2.19 | -3.32 | [-8.52, 1.89] | 0.62 |

- D vs C (not part of the A/B/C length tests): slope difference D - C -0.324, Welch p = 0.8211, exact permutation p = 1.0000 (3 vs 3 runs)

**Window: own, generator theorems only.**

| arm (train max) | lengths used | slope per run (log-odds per line) | mean | t-interval over seeds (95%) | mean within-run bootstrap SE |
|---|---|---|---|---|---|
| pretrain (6) | 7-18 | -3.59 | -3.59 | - | 1.00 |
| A (7) | 8-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| B (9) | 10-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), -5.68, n/a (successes at fewer than 2 lengths (slope not estimable)) | -5.68 | - | 0.50 |
| C (11) | 12-18 | -2.49, -2.72, -4.02 | -3.08 | [-5.13, -1.02] | 0.78 |
| D (11) | 12-18 | -5.85, -2.08, -2.26 | -3.39 | [-8.68, 1.89] | 0.63 |

- trend of slope with training max: +1.304 log-odds/line per extra training line, exact permutation p = 0.2500 (4 relabelings of 4 runs); omnibus F = 7.47, permutation p = 0.2500
- B-C: difference -2.607, Welch p = -, exact permutation p = 0.2500 (smallest possible 0.25)
- D vs C (not part of the A/B/C length tests): slope difference D - C -0.318, Welch p = 0.8268, exact permutation p = 1.0000 (3 vs 3 runs)

**Window: common (L >= 12), generator only.**

| arm (train max) | lengths used | slope per run (log-odds per line) | mean | t-interval over seeds (95%) | mean within-run bootstrap SE |
|---|---|---|---|---|---|
| pretrain (6) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| A (7) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| B (9) | 12-18 | n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)), n/a (successes at fewer than 2 lengths (slope not estimable)) | - | - | - |
| C (11) | 12-18 | -2.49, -2.72, -4.02 | -3.08 | [-5.13, -1.02] | 0.75 |
| D (11) | 12-18 | -5.85, -2.08, -2.26 | -3.39 | [-8.68, 1.89] | 0.69 |

- D vs C (not part of the A/B/C length tests): slope difference D - C -0.318, Welch p = 0.8268, exact permutation p = 1.0000 (3 vs 3 runs)

## 3. Lengths of correct proofs vs the L_true label

Cell: fraction of correct proofs shorter than L_true / median (lines - L_true), pooled over seeds. A proof shorter than L_true shows that the label (from the restricted minlen search) is too high for that theorem. Last row: theorems (any arm) with such a proof.

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.000 / +0 (n=1515) | 0.000 / +0 (n=293) | 0.000 / +0 (n=9) |  |  |  |  |  |  |  |  |  |  |
| A | 0.000 / +0 (n=16977) | 0.000 / +0 (n=14538) | 0.000 / +0 (n=1227) |  |  |  |  |  |  |  |  |  |  |
| B | 0.000 / +0 (n=17554) | 0.000 / +0 (n=12537) | 0.000 / +0 (n=7087) | 0.000 / +0 (n=5419) | 0.000 / +0 (n=813) | 0.000 / +0 (n=1) |  |  |  |  |  |  |  |
| C | 0.000 / +0 (n=16200) | 0.000 / +0 (n=10480) | 0.000 / +0 (n=5510) | 0.000 / +0 (n=4919) | 0.000 / +0 (n=3136) | 0.000 / +0 (n=3669) | 0.000 / +0 (n=975) | 0.000 / +0 (n=63) |  |  |  |  |  |
| D | 0.000 / +0 (n=16175) | 0.000 / +0 (n=10155) | 0.000 / +0 (n=5659) | 0.000 / +0 (n=5297) | 0.000 / +0 (n=3061) | 0.000 / +0 (n=3453) | 0.000 / +0 (n=942) | 0.000 / +0 (n=97) |  |  |  |  |  |
| theorems with any shorter correct proof | 0/300 | 0/300 | 0/300 | 0/300 | 0/300 | 0/300 | 0/300 | 0/300 | 0/158 | 0/300 | 0/300 | 0/300 | 0/300 |

![proof lengths](fig_prooflen.png)

## 4. Where failed samples go wrong

Cell: mean failure position / fraction of failures at positions max..max+2 of the arm's own training max / fraction early QED / fraction out of tokens / fraction that ran past the N64 line-index vocabulary. Failure position = first line the incremental checker rejects, or the QED slot for an early QED.

| arm (train max) | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain (6) | 4.2 / 0.03 / 0.00 / 0.00 / 0.01 | 4.3 / 0.04 / 0.00 / 0.00 / 0.00 | 4.0 / 0.05 / 0.00 / 0.00 / 0.00 | 4.0 / 0.05 / 0.00 / 0.00 / 0.00 | 4.1 / 0.05 / 0.00 / 0.00 / 0.01 | 4.1 / 0.04 / 0.00 / 0.00 / 0.00 | 4.1 / 0.04 / 0.00 / 0.00 / 0.00 | 3.9 / 0.07 / 0.01 / 0.00 / 0.00 | 3.2 / 0.02 / 0.00 / 0.00 / 0.00 | 4.1 / 0.07 / 0.00 / 0.00 / 0.00 | 4.3 / 0.04 / 0.00 / 0.00 / 0.00 | 4.3 / 0.06 / 0.00 / 0.00 / 0.00 | 4.3 / 0.04 / 0.00 / 0.00 / 0.00 |
| A (7) | 4.5 / 0.01 / 0.00 / 0.00 / 0.00 | 4.5 / 0.02 / 0.00 / 0.00 / 0.00 | 4.2 / 0.02 / 0.00 / 0.00 / 0.00 | 4.4 / 0.01 / 0.00 / 0.00 / 0.00 | 4.2 / 0.01 / 0.00 / 0.00 / 0.00 | 4.2 / 0.00 / 0.00 / 0.00 / 0.00 | 4.2 / 0.00 / 0.00 / 0.00 / 0.00 | 4.4 / 0.00 / 0.01 / 0.00 / 0.00 | 3.6 / 0.00 / 0.00 / 0.00 / 0.00 | 5.1 / 0.06 / 0.00 / 0.00 / 0.00 | 5.1 / 0.04 / 0.00 / 0.00 / 0.00 | 5.2 / 0.03 / 0.00 / 0.00 / 0.00 | 5.1 / 0.01 / 0.00 / 0.00 / 0.00 |
| B (9) | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.01 / 0.00 / 0.00 | 4.3 / 0.00 / 0.00 / 0.00 / 0.00 | 4.5 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.01 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.8 / 0.00 / 0.01 / 0.00 / 0.00 | 3.9 / 0.00 / 0.00 / 0.00 / 0.00 | 6.0 / 0.02 / 0.00 / 0.00 / 0.00 | 6.0 / 0.03 / 0.00 / 0.00 / 0.00 | 6.1 / 0.04 / 0.00 / 0.00 / 0.00 | 6.1 / 0.03 / 0.00 / 0.00 / 0.00 |
| C (11) | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.8 / 0.00 / 0.00 / 0.00 / 0.00 | 4.3 / 0.00 / 0.00 / 0.00 / 0.00 | 4.5 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.6 / 0.00 / 0.00 / 0.00 / 0.00 | 5.1 / 0.01 / 0.00 / 0.00 / 0.00 | 5.3 / 0.00 / 0.00 / 0.00 / 0.00 | 4.0 / 0.00 / 0.00 / 0.00 / 0.00 | 6.3 / 0.02 / 0.00 / 0.00 / 0.00 | 6.2 / 0.01 / 0.00 / 0.00 / 0.00 | 6.5 / 0.03 / 0.00 / 0.00 / 0.00 | 6.7 / 0.05 / 0.00 / 0.00 / 0.00 |
| D (11) | 4.6 / 0.00 / 0.00 / 0.00 / 0.00 | 4.6 / 0.00 / 0.00 / 0.00 / 0.00 | 4.3 / 0.00 / 0.00 / 0.00 / 0.00 | 4.4 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 4.7 / 0.00 / 0.00 / 0.00 / 0.00 | 5.0 / 0.01 / 0.00 / 0.00 / 0.00 | 5.1 / 0.00 / 0.01 / 0.00 / 0.00 | 3.9 / 0.00 / 0.00 / 0.00 / 0.00 | 6.8 / 0.02 / 0.00 / 0.00 / 0.00 | 6.6 / 0.02 / 0.00 / 0.00 / 0.00 | 6.8 / 0.04 / 0.00 / 0.00 / 0.00 | 6.9 / 0.06 / 0.00 / 0.00 / 0.00 |

![first error](fig_first_error.png)

![hazard](fig_hazard.png)

If failures pile up where each arm stops having training data (a length-specific circuit), the hazard curves spike at each arm's own dashed line; if they spread evenly, the hazard is flat in position.

**First line index of samples** (the abs tokenizer has only N1..N64):

| arm | median first line index | 90th pct | 99th pct | fraction that cannot fit 18 lines (start > N47) |
|---|---|---|---|---|
| pretrain | 31 | 54 | 61 | 0.219 |
| A | 21 | 37 | 41 | 0.000 |
| B | 21 | 37 | 41 | 0.000 |
| C | 21 | 37 | 41 | 0.000 |
| D | 21 | 37 | 41 | 0.000 |

## 5. Training-side diagnostics

| run | status | rounds | samples | held-out greedy | compute used | budget exhausted | fraction of the target subset solved at the end, by L_true (subset size) |
|---|---|---|---|---|---|---|---|
| A_s101 | ok | 73 | 928512 | 0.964 | 1.00 | False | L6 0.95 of 1000, L7 0.75 of 2500 |
| A_s102 | ok | 73 | 930048 | 0.965 | 1.00 | False | L6 0.94 of 1000, L7 0.74 of 2500 |
| A_s103 | ok | 73 | 931968 | 0.957 | 1.00 | False | L6 0.94 of 1000, L7 0.69 of 2500 |
| B_s101 | ok | 65 | 826880 | 0.951 | 1.00 | False | L6 0.92 of 1000, L7 0.70 of 1666, L8 0.68 of 1667, L9 0.41 of 1667 |
| B_s102 | ok | 65 | 829952 | 0.958 | 1.00 | False | L6 0.92 of 1000, L7 0.64 of 1666, L8 0.60 of 1667, L9 0.40 of 1667 |
| B_s103 | ok | 65 | 826112 | 0.948 | 1.00 | False | L6 0.90 of 1000, L7 0.65 of 1666, L8 0.59 of 1667, L9 0.42 of 1667 |
| C_s101 | ok | 64 | 814080 | 0.959 | 1.00 | False | L6 0.88 of 1000, L7 0.59 of 1000, L8 0.53 of 1000, L9 0.36 of 1000, L10 0.22 of 1000, L11 0.35 of 1000 |
| C_s102 | ok | 64 | 811776 | 0.947 | 1.00 | False | L6 0.89 of 1000, L7 0.61 of 1000, L8 0.61 of 1000, L9 0.39 of 1000, L10 0.23 of 1000, L11 0.34 of 1000 |
| C_s103 | ok | 64 | 814080 | 0.954 | 1.00 | False | L6 0.88 of 1000, L7 0.59 of 1000, L8 0.50 of 1000, L9 0.35 of 1000, L10 0.22 of 1000, L11 0.34 of 1000 |
| D_s101 | ok | 65 | 830720 | 0.953 | 1.00 | False | L6 0.88 of 1000, L7 0.68 of 1000, L8 0.57 of 1000, L9 0.39 of 1000, L10 0.22 of 1000, L11 0.23 of 1000 |
| D_s102 | ok | 65 | 832000 | 0.952 | 0.99 | False | L6 0.87 of 1000, L7 0.68 of 1000, L8 0.52 of 1000, L9 0.35 of 1000, L10 0.23 of 1000, L11 0.27 of 1000 |
| D_s103 | ok | 65 | 832000 | 0.963 | 0.99 | False | L6 0.88 of 1000, L7 0.63 of 1000, L8 0.54 of 1000, L9 0.39 of 1000, L10 0.21 of 1000, L11 0.25 of 1000 |

Lines of the proofs the run found (and so trained on), by the target's L_true: max / 90th percentile (written length).

| run | L_true 6 | L_true 7 | L_true 8 | L_true 9 | L_true 10 | L_true 11 | share of found proofs longer than max_len_train | longest |
|---|---|---|---|---|---|---|---|---|
| A_s101 | 8 / 7 | 8 / 7 |  |  |  |  | 0.042 | 8 |
| A_s102 | 8 / 7 | 9 / 7 |  |  |  |  | 0.045 | 9 |
| A_s103 | 8 / 7 | 9 / 7 |  |  |  |  | 0.042 | 9 |
| B_s101 | 9 / 7 | 11 / 8 | 11 / 9 | 11 / 9 |  |  | 0.022 | 11 |
| B_s102 | 10 / 7 | 10 / 8 | 11 / 9 | 11 / 9 |  |  | 0.019 | 11 |
| B_s103 | 10 / 7 | 10 / 8 | 11 / 9 | 11 / 9 |  |  | 0.022 | 11 |
| C_s101 | 11 / 7 | 11 / 8 | 13 / 9 | 12 / 10 | 12 / 11 | 13 / 11 | 0.008 | 13 |
| C_s102 | 9 / 7 | 11 / 9 | 12 / 9 | 12 / 10 | 12 / 11 | 13 / 11 | 0.010 | 13 |
| C_s103 | 10 / 6 | 12 / 8 | 13 / 9 | 12 / 10 | 12 / 11 | 12 / 11 | 0.008 | 13 |
| D_s101 | 9 / 7 | 12 / 8 | 13 / 9 | 12 / 10 | 12 / 11 | 14 / 11 | 0.011 | 14 |
| D_s102 | 11 / 7 | 12 / 8 | 13 / 9 | 12 / 10 | 13 / 11 | 13 / 11 | 0.011 | 13 |
| D_s103 | 11 / 7 | 12 / 8 | 13 / 9 | 12 / 10 | 13 / 11 | 12 / 11 | 0.007 | 13 |

![training](fig_training.png)

Targets arm D solved during training (distinct targets with at least one found proof), by family. The merged pool is 30% constructed.

| run | solved targets | constructed share | standard | andi | chaink1 | chaink2 |
|---|---|---|---|---|---|---|
| D_s101 | 2970 | 32.8% | 1995 | 363 | 219 | 393 |
| D_s102 | 2916 | 34.3% | 1917 | 393 | 200 | 406 |
| D_s103 | 2904 | 34.3% | 1907 | 349 | 242 | 406 |

Share of the training subset solved by the end, by L_true and family (mean over D runs). The subset has 1,000 targets per length drawn uniformly within the length, so the per-family denominators are the pool shares x 1,000 (estimated; the subset is not logged by family).

| L_true | standard | andi | chaink1 | chaink2 |
|---|---|---|---|---|
| L6 | 88% (612 of ~699) | 98% (40 of ~41) | 82% (48 of ~59) | 88% (176 of ~201) |
| L7 | 57% (397 of ~700) | 93% (140 of ~150) | 83% (58 of ~70) | 85% (68 of ~80) |
| L8 | 49% (342 of ~700) | 62% (93 of ~150) | 64% (36 of ~57) | 76% (71 of ~93) |
| L9 | 34% (241 of ~700) | 44% (66 of ~150) | 49% (31 of ~63) | 48% (42 of ~87) |
| L10 | 20% (137 of ~700) | 16% (24 of ~150) | 40% (30 of ~75) | 35% (26 of ~74) |
| L11 | 30% (211 of ~700) | 4% (5 of ~150) | 23% (17 of ~73) | 25% (19 of ~77) |

## 6. Matched-construction control

Cell: pass@1 standard / matched.

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.158 / 0.010 | 0.031 / 0.002 | 0.001 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |  |
| A | 0.589 / 0.135 | 0.505 / 0.132 | 0.043 / 0.034 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |  |
| B | 0.610 / 0.124 | 0.435 / 0.121 | 0.246 / 0.098 | 0.188 / 0.104 | 0.028 / 0.051 | 0.000 / 0.002 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |  |
| C | 0.562 / 0.116 | 0.364 / 0.107 | 0.191 / 0.075 | 0.171 / 0.079 | 0.109 / 0.067 | 0.127 / 0.040 | 0.034 / 0.014 | 0.002 / 0.004 | 0.000 / 0.000 |  |  |  |  |
| D | 0.562 / 0.456 | 0.353 / 0.402 | 0.196 / 0.300 | 0.184 / 0.194 | 0.106 / 0.125 | 0.120 / 0.062 | 0.033 / 0.027 | 0.003 / 0.002 | 0.000 / 0.000 |  |  |  |  |

![matched](fig_matched.png)

**pass@1 by construction family** (mL = matched, sL = sealed; mean over seeds):

| arm | family | mL6 | mL7 | mL8 | mL9 | mL10 | mL11 | mL12 | mL13 | mL14 | sL15 | sL16 | sL17 | sL18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | andi | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pretrain | chaink1 | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pretrain | chaink2 | 0.007 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A | andi | 0.146 | 0.030 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A | chaink1 | 0.292 | 0.358 | 0.072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A | chaink2 | 0.060 | 0.130 | 0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | andi | 0.120 | 0.044 | 0.006 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | chaink1 | 0.276 | 0.237 | 0.195 | 0.234 | 0.104 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | chaink2 | 0.068 | 0.164 | 0.193 | 0.195 | 0.088 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | andi | 0.125 | 0.039 | 0.006 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | chaink1 | 0.207 | 0.206 | 0.129 | 0.188 | 0.153 | 0.078 | 0.026 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | chaink2 | 0.069 | 0.148 | 0.158 | 0.138 | 0.104 | 0.081 | 0.031 | 0.013 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| D | andi | 0.395 | 0.378 | 0.169 | 0.131 | 0.037 | 0.017 | 0.004 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D | chaink1 | 0.512 | 0.466 | 0.378 | 0.264 | 0.210 | 0.117 | 0.023 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D | chaink2 | 0.512 | 0.395 | 0.479 | 0.259 | 0.197 | 0.094 | 0.080 | 0.003 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |

## 7. Attempt lengths

Cell: median written lines of all samples / share of samples with at least L_true written lines (pooled over seeds). A proof cannot be correct with fewer lines than the theorem needs.

**Standard sets.**

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | 6 / 75% | 6 / 38% | 6 / 4% | 6 / 0% | 6 / 0% | 6 / 0% | 7 / 0% | 7 / 0% | 6 / 0% | 7 / 0% | 7 / 0% | 7 / 0% | 7 / 0% |
| A | 6 / 93% | 7 / 75% | 7 / 12% | 7 / 0% | 7 / 0% | 7 / 0% | 7 / 0% | 7 / 0% | 6 / 0% | 7 / 0% | 7 / 0% | 7 / 0% | 7 / 0% |
| B | 6 / 94% | 7 / 76% | 7 / 44% | 7 / 33% | 7 / 7% | 8 / 0% | 7 / 0% | 8 / 0% | 6 / 0% | 9 / 0% | 9 / 0% | 9 / 0% | 9 / 0% |
| C | 6 / 93% | 7 / 70% | 7 / 40% | 7 / 33% | 7 / 25% | 7 / 24% | 7 / 9% | 9 / 3% | 6 / 0% | 9 / 0% | 9 / 0% | 9 / 0% | 9 / 0% |
| D | 6 / 92% | 7 / 65% | 7 / 37% | 7 / 33% | 7 / 24% | 7 / 23% | 7 / 8% | 8 / 2% | 6 / 0% | 10 / 0% | 9 / 0% | 10 / 0% | 10 / 0% |

**Matched constructed sets.**

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| pretrain | 7 / 84% | 7 / 67% | 7 / 34% | 7 / 10% | 7 / 6% | 7 / 3% | 7 / 1% | 7 / 1% | 7 / 0% |
| A | 7 / 90% | 7 / 80% | 7 / 41% | 7 / 4% | 7 / 1% | 7 / 0% | 7 / 0% | 7 / 0% | 7 / 0% |
| B | 7 / 91% | 7 / 82% | 8 / 64% | 8 / 41% | 9 / 22% | 9 / 3% | 9 / 1% | 9 / 0% | 9 / 0% |
| C | 7 / 87% | 7 / 78% | 8 / 60% | 8 / 41% | 9 / 32% | 9 / 21% | 9 / 12% | 9 / 4% | 9 / 1% |
| D | 6 / 95% | 7 / 81% | 8 / 70% | 9 / 54% | 9 / 41% | 10 / 25% | 9 / 15% | 9 / 5% | 9 / 1% |

## 8. Arm D (composition-augmented RL pool) vs arm C

Arm D is arm C exactly (recipe, max_len_train 11, max_proof_lines_train 11, compute budget, seeds 101-103) except the RL pool: `ab3/frozen_comp` holds `data_comp/targets_L6-11.jsonl` (run 2's targets plus constructed ANDI/chain targets, 30% of each length bin). D is not part of the A/B/C length-trend tests.

**Conclusion.** Adding constructed composition targets to the RL pool teaches the model to compose short pieces. It does
not change length generalisation.

- **Standard sets: D is arm C.**
  - pass@1 differs from C by -1.1 to +1.3 points at every length from 6 to 18, and pass@4 by -1.6 to +1.3 points.
  - Past 11 lines, the log-odds drop is -3.32 per line for D and -2.99 for C (D - C = -0.32; Welch p = 0.82, permutation
    p = 1.0 with 3 vs 3 runs).
  - Both arms are at 0% from L14 to L18.
- **Constructed sets: D is much better at short lengths, and the gain shrinks with length.**
  - On the matched set (all families), D - C is +34.0 points at L6 (11.6% to 45.6%), +29.5 at L7, +22.5 at L8, +11.5 at
    L9, +5.8 at L10, +2.2 at L11, +1.3 at L12, and about 0 at L13-14.
  - ANDI rises from 12.5% to 39.5% at L6, from 3.9% to 37.8% at L7 and from 0.1% to 13.1% at L9, but reaches only 1.7%
    at L11.
  - Sealed L15-18 is 0.0% for both arms in every family.
  - The gain fades inside D's own training range (L10-11), not only beyond it.
- **Why the gain stops (section 5).** Expert iteration learns only from proofs the model finds, and D found them only
  for short constructed targets. By the end of training D had solved 93-98% of its ANDI training targets at L6-7, 62% at
  L8, 44% at L9, 16% at L10 and 4% at L11. The per-family denominators are estimated (see section 5).
- **Attempt length is unchanged on long theorems.** D writes a median of 9-10 lines at L15-18. At most 0.3% of its
  attempts there reach L_true, against at most 0.1% for C.

**Standard sets, pass@1** (mean [min, max] over seeds):

| length | C | D | D - C |
|---|---|---|---|
| L6 | 0.562 [0.555, 0.569] | 0.562 [0.551, 0.569] | -0.1 pts |
| L7 | 0.364 [0.360, 0.368] | 0.353 [0.338, 0.382] | -1.1 pts |
| L8 | 0.191 [0.173, 0.213] | 0.196 [0.175, 0.209] | +0.5 pts |
| L9 | 0.171 [0.161, 0.184] | 0.184 [0.176, 0.190] | +1.3 pts |
| L10 | 0.109 [0.099, 0.118] | 0.106 [0.098, 0.111] | -0.3 pts |
| L11 | 0.127 [0.126, 0.129] | 0.120 [0.116, 0.123] | -0.8 pts |
| L12 | 0.034 [0.031, 0.039] | 0.033 [0.029, 0.038] | -0.1 pts |
| L13 | 0.002 [0.001, 0.003] | 0.003 [0.000, 0.005] | +0.1 pts |
| L14 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L15 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L17 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |

**Standard sets, pass@4** (mean [min, max] over seeds):

| length | C | D | D - C |
|---|---|---|---|
| L6 | 0.692 [0.684, 0.699] | 0.692 [0.672, 0.719] | +0.0 pts |
| L7 | 0.448 [0.444, 0.451] | 0.432 [0.409, 0.467] | -1.6 pts |
| L8 | 0.249 [0.220, 0.287] | 0.258 [0.223, 0.276] | +0.9 pts |
| L9 | 0.219 [0.207, 0.229] | 0.232 [0.227, 0.239] | +1.3 pts |
| L10 | 0.150 [0.126, 0.166] | 0.144 [0.134, 0.153] | -0.7 pts |
| L11 | 0.160 [0.157, 0.162] | 0.150 [0.141, 0.158] | -1.0 pts |
| L12 | 0.050 [0.047, 0.054] | 0.050 [0.044, 0.060] | -0.0 pts |
| L13 | 0.005 [0.003, 0.006] | 0.007 [0.000, 0.011] | +0.2 pts |
| L14 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L15 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L17 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| L18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |

**Log-odds drop per line past 11** (grouped-binomial logistic fit per run over L >= 12, as in section 2):

| window | arm | slope per run | mean | 95% t-interval / test |
|---|---|---|---|---|
| own (d >= 1) | C | -2.42, -2.64, -3.91 | -2.99 | [-4.99, -0.99] |
| own (d >= 1) | D | -5.73, -2.02, -2.19 | -3.32 | [-8.52, 1.89] |
| own (d >= 1) | D - C |  | -0.32 | Welch p = 0.8211, permutation p = 1.0000 |
| own, generator theorems only | C | -2.49, -2.72, -4.02 | -3.08 | [-5.13, -1.02] |
| own, generator theorems only | D | -5.85, -2.08, -2.26 | -3.39 | [-8.68, 1.89] |
| own, generator theorems only | D - C |  | -0.32 | Welch p = 0.8268, permutation p = 1.0000 |

**Constructed sets, pass@1 by family** (matched L6-14, sealed L15-18; mean [min, max] over seeds):

| set | family | theorems | C | D | D - C |
|---|---|---|---|---|---|
| matched L6 | andi | 144 | 0.125 [0.116, 0.138] | 0.395 [0.368, 0.418] | +27.0 pts |
| matched L6 | chaink1 | 44 | 0.207 [0.188, 0.227] | 0.512 [0.467, 0.577] | +30.4 pts |
| matched L6 | chaink2 | 112 | 0.069 [0.053, 0.078] | 0.512 [0.488, 0.555] | +44.3 pts |
| matched L6 | all | 300 | 0.116 [0.108, 0.122] | 0.456 [0.429, 0.471] | +34.0 pts |
| matched L7 | andi | 148 | 0.039 [0.027, 0.052] | 0.378 [0.365, 0.387] | +33.9 pts |
| matched L7 | chaink1 | 67 | 0.206 [0.167, 0.235] | 0.466 [0.464, 0.471] | +26.0 pts |
| matched L7 | chaink2 | 85 | 0.148 [0.138, 0.158] | 0.395 [0.356, 0.420] | +24.6 pts |
| matched L7 | all | 300 | 0.107 [0.093, 0.117] | 0.402 [0.385, 0.415] | +29.5 pts |
| matched L8 | andi | 152 | 0.006 [0.004, 0.008] | 0.169 [0.153, 0.187] | +16.2 pts |
| matched L8 | chaink1 | 66 | 0.129 [0.102, 0.143] | 0.378 [0.321, 0.415] | +25.0 pts |
| matched L8 | chaink2 | 82 | 0.158 [0.138, 0.171] | 0.479 [0.454, 0.506] | +32.1 pts |
| matched L8 | all | 300 | 0.075 [0.064, 0.081] | 0.300 [0.290, 0.306] | +22.5 pts |
| matched L9 | andi | 155 | 0.001 [0.000, 0.002] | 0.131 [0.126, 0.139] | +13.0 pts |
| matched L9 | chaink1 | 69 | 0.188 [0.183, 0.196] | 0.264 [0.246, 0.286] | +7.7 pts |
| matched L9 | chaink2 | 76 | 0.138 [0.120, 0.148] | 0.259 [0.231, 0.280] | +12.2 pts |
| matched L9 | all | 300 | 0.079 [0.074, 0.082] | 0.194 [0.185, 0.199] | +11.5 pts |
| matched L10 | andi | 141 | 0.000 [0.000, 0.000] | 0.037 [0.033, 0.045] | +3.7 pts |
| matched L10 | chaink1 | 72 | 0.153 [0.139, 0.161] | 0.210 [0.207, 0.216] | +5.7 pts |
| matched L10 | chaink2 | 87 | 0.104 [0.072, 0.137] | 0.197 [0.185, 0.219] | +9.3 pts |
| matched L10 | all | 300 | 0.067 [0.054, 0.077] | 0.125 [0.121, 0.129] | +5.8 pts |
| matched L11 | andi | 149 | 0.000 [0.000, 0.000] | 0.017 [0.008, 0.026] | +1.7 pts |
| matched L11 | chaink1 | 77 | 0.078 [0.065, 0.088] | 0.117 [0.108, 0.135] | +4.0 pts |
| matched L11 | chaink2 | 74 | 0.081 [0.072, 0.098] | 0.094 [0.087, 0.103] | +1.3 pts |
| matched L11 | all | 300 | 0.040 [0.035, 0.044] | 0.062 [0.057, 0.066] | +2.2 pts |
| matched L12 | andi | 150 | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.005] | +0.4 pts |
| matched L12 | chaink1 | 78 | 0.026 [0.016, 0.038] | 0.023 [0.010, 0.032] | -0.3 pts |
| matched L12 | chaink2 | 72 | 0.031 [0.028, 0.036] | 0.080 [0.071, 0.092] | +4.9 pts |
| matched L12 | all | 300 | 0.014 [0.011, 0.017] | 0.027 [0.022, 0.030] | +1.3 pts |
| matched L13 | andi | 151 | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.003] | +0.2 pts |
| matched L13 | chaink1 | 74 | 0.003 [0.000, 0.007] | 0.002 [0.000, 0.003] | -0.1 pts |
| matched L13 | chaink2 | 75 | 0.013 [0.010, 0.017] | 0.003 [0.000, 0.007] | -0.9 pts |
| matched L13 | all | 300 | 0.004 [0.003, 0.004] | 0.002 [0.002, 0.003] | -0.1 pts |
| matched L14 | andi | 152 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| matched L14 | chaink1 | 77 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| matched L14 | chaink2 | 71 | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.003] | +0.0 pts |
| matched L14 | all | 300 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | +0.0 pts |
| sealed L15 | andi | 155 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L15 | chaink1 | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L15 | chaink2 | 61 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L15 | all | 300 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L16 | andi | 149 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L16 | chaink1 | 83 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L16 | chaink2 | 68 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L16 | all | 300 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L17 | andi | 222 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L17 | chaink1 | 29 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L17 | chaink2 | 49 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L17 | all | 300 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L18 | andi | 281 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L18 | chaink1 | 7 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L18 | chaink2 | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |
| sealed L18 | all | 300 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 pts |

## Checks

- Incremental checker vs nd_verify disagreements over 2686528 samples: none.
- Samples that hit max_new: 982.
