# propensity_capability: frozen tables

Written by `analyze.py` under PREREG.md (sha256 `6fd57cad8189affa0b7ac8362f32645e290626b11bab702cea7a149bcd2e1261`). Every rate is % of attempts over the
applicable theorems of one set, with a 95 % bootstrap interval over theorems (2000 resamples, seeds shown separately, never pooled).
Reading thresholds: a change counts at |Δlog10| ≥ 0.15 with the interval excluding 0; flat iff the interval lies inside ±0.3; rare iff below 1.0 % at both checkpoints; n ≥ 20 applicable theorems.

## Is the split informative here (PREREG, decided first)

**Verdict: informative.** Primary cells read: 30 of 54 (unresolved or no reading: 24). E fails against U in 0 of 138 checkpoint cells.
(a) more than half not read: False; (b) no dose-level reading anywhere: False; (c) E fails in more than a quarter: False.

## Dose-level readings (primary: first-move propensity from E, capability from F)

Each dose's RL end is at its own compute (x at the end in the last column); doses are not compared here.

| dose | set | direction | per seed | dose reading | Δlog10 P, mean (range) | Δlog10 C, mean (range) | x at RL ends |
|---|---|---|---|---|---|---|---|
| 0 % | strict106 | reductio | no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **no reading: forced attempts mostly refused or capped** | -2.87 (-5.18 to -0.25) | +0.73 (+0.45 to +1.18) | 5.07, 5.93, 5.94 |
| 0 % | strict106 | negation | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 5.07, 5.93, 5.94 |
| 0 % | strict106 | conditional | no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **no reading: forced attempts mostly refused or capped** | +0.10 (+0.05 to +0.14) | +0.00 (+0.00 to +0.00) | 5.07, 5.93, 5.94 |
| 0 % | strict106 | cases | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 5.07, 5.93, 5.94 |
| 0 % | transfer_other | reductio | candidate capability gain / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **seeds disagree** | -5.15 (-6.33 to -4.32) | +1.06 (+0.72 to +1.24) | 5.07, 5.93, 5.94 |
| 0 % | transfer_other | negation | candidate capability gain / candidate capability gain / mix | **seeds disagree** | +0.14 (-0.05 to +0.36) | +1.17 (+0.91 to +1.32) | 5.07, 5.93, 5.94 |
| 0 % | transfer_other | conditional | candidate capability gain / candidate capability gain / candidate capability gain | **candidate capability gain** | +0.08 (+0.05 to +0.13) | +0.39 (+0.34 to +0.46) | 5.07, 5.93, 5.94 |
| 0 % | transfer_other | cases | mix / mix / mix | **mix** | +0.65 (+0.49 to +0.85) | +0.47 (+0.47 to +0.47) | 5.07, 5.93, 5.94 |
| 0.1 % | strict106 | reductio | no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **no reading: forced attempts mostly refused or capped** | -2.73 (-3.94 to -1.46) | +1.15 (+0.71 to +1.51) | 4.42, 5.43, 5.63 |
| 0.1 % | strict106 | negation | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 4.42, 5.43, 5.63 |
| 0.1 % | strict106 | conditional | no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **no reading: forced attempts mostly refused or capped** | +0.14 (+0.08 to +0.16) | -0.23 (-0.70 to +0.00) | 4.42, 5.43, 5.63 |
| 0.1 % | strict106 | cases | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 4.42, 5.43, 5.63 |
| 0.1 % | transfer_other | reductio | no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **no reading: forced attempts mostly refused or capped** | -4.34 (-5.51 to -3.35) | +1.36 (+1.22 to +1.52) | 4.42, 5.43, 5.63 |
| 0.1 % | transfer_other | negation | mix / no reading, forced attempts mostly refused or capped / no reading, forced attempts mostly refused or capped | **seeds disagree** | +0.29 (+0.17 to +0.47) | +1.24 (+1.13 to +1.33) | 4.42, 5.43, 5.63 |
| 0.1 % | transfer_other | conditional | candidate capability gain / candidate capability gain / candidate capability gain | **candidate capability gain** | +0.08 (+0.02 to +0.13) | +0.39 (+0.35 to +0.41) | 4.42, 5.43, 5.63 |
| 0.1 % | transfer_other | cases | mix / candidate capability gain / mix | **seeds disagree** | +0.38 (+0.14 to +0.58) | +0.50 (+0.41 to +0.60) | 4.42, 5.43, 5.63 |
| 1 % | strict106 | reductio | no reading, forced attempts mostly refused or capped / candidate capability gain / no reading, forced attempts mostly refused or capped | **seeds disagree** | +0.63 (-0.06 to +1.19) | +1.21 (+0.65 to +1.95) | 0.41, 0.45, 0.69 |
| 1 % | strict106 | negation | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 0.41, 0.45, 0.69 |
| 1 % | strict106 | conditional | no reading, forced attempts mostly refused or capped / mix / candidate capability gain | **seeds disagree** | +0.14 (+0.10 to +0.19) | +1.80 (+0.79 to +2.47) | 0.41, 0.45, 0.69 |
| 1 % | strict106 | cases | no reading, never applicable / no reading, never applicable / no reading, never applicable | **never applicable** | n/a | n/a | 0.41, 0.45, 0.69 |
| 1 % | transfer_other | reductio | candidate capability gain / candidate capability gain / no reading, forced attempts mostly refused or capped | **candidate capability gain (2 of 3)** | +0.01 (-1.53 to +1.45) | +1.14 (+1.03 to +1.25) | 0.41, 0.45, 0.69 |
| 1 % | transfer_other | negation | no reading, forced attempts mostly refused or capped / mix / candidate capability gain | **seeds disagree** | +0.16 (+0.05 to +0.28) | +1.26 (+1.24 to +1.29) | 0.41, 0.45, 0.69 |
| 1 % | transfer_other | conditional | candidate capability gain / candidate capability gain / candidate capability gain | **candidate capability gain** | +0.09 (+0.07 to +0.11) | +0.36 (+0.33 to +0.39) | 0.41, 0.45, 0.69 |
| 1 % | transfer_other | cases | mix / mix / mix | **mix** | +0.38 (+0.31 to +0.46) | +0.47 (+0.42 to +0.51) | 0.41, 0.45, 0.69 |

## Per seed: base → RL end

P = exact first-move probability (E). C = solved when forced (F). States: up, down, flat, rare, unresolved.

| dose | seed | set | direction | n | P base → end | P ratio | P | C base → end | C ratio | C | reading | share of Δ(P̄C̄) from P | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 % | 0 | strict106 | reductio | 106 | 3.3e-05 → 2.1e-08 | x10^-3.2 [10^-5.7, 10^-2.4] | rare | 2.8 → 8.0 | x2.81 [1.51, 7.61] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0 % | 0 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 0 % | 0 | strict106 | conditional | 33 | 89.4 → 100.0 | x1.12 [1.05, 1.22] | flat | 0 → 0 | x1 [1, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0 % | 0 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 0 % | 0 | transfer_other | reductio | 240 | 2.1e-06 → 2.2e-13 | x10^-4.3 [10^-4.7, 10^-3.6] | rare | 1.1 → 20.2 | x17.4 [10, 41.5] | up | **candidate capability gain** | n/a |  |
| 0 % | 0 | transfer_other | negation | 44 | 71.1 → 62.7 | x0.882 [0.646, 1.16] | flat | 3.4 → 73.9 | x21.1 [11.6, 65] | up | **candidate capability gain** | n/a |  |
| 0 % | 0 | transfer_other | conditional | 84 | 85.0 → 97.6 | x1.15 [1.08, 1.22] | flat | 28.8 → 84.0 | x2.92 [2.36, 3.73] | up | **candidate capability gain** | n/a |  |
| 0 % | 0 | transfer_other | cases | 80 | 20.1 → 62.0 | x3.09 [2.31, 4.28] | up | 30.3 → 90.1 | x2.97 [2.31, 4.09] | up | **mix** | 51 % |  |
| 0 % | 1 | strict106 | reductio | 106 | 2.8e-05 → 1.8e-10 | x10^-5.2 [10^-5.7, 10^-4.5] | rare | 0.9 → 13.6 | x15.1 [8.73, 39.7] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0 % | 1 | strict106 | negation | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0 % | 1 | strict106 | conditional | 33 | 75.7 → 100.0 | x1.32 [1.16, 1.56] | flat | 0 → 0 | x1 [1, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0 % | 1 | strict106 | cases | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0 % | 1 | transfer_other | reductio | 240 | 2.1e-04 → 7.0e-11 | x10^-6.3 [10^-6.7, 10^-5.0] | rare | 1.8 → 9.3 | x5.21 [3.08, 11.9] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0 % | 1 | transfer_other | negation | 44 | 45.3 → 61.4 | x1.35 [0.934, 1.95] | flat | 3.8 → 73.5 | x19 [8.68, 153] | up | **candidate capability gain** | n/a | alarm end |
| 0 % | 1 | transfer_other | conditional | 84 | 87.6 → 98.8 | x1.13 [1.07, 1.2] | flat | 37.4 → 86.7 | x2.32 [1.93, 2.92] | up | **candidate capability gain** | n/a | alarm end |
| 0 % | 1 | transfer_other | cases | 80 | 14.3 → 60.0 | x4.21 [2.86, 6.2] | up | 27.8 → 81.6 | x2.93 [2.27, 4.1] | up | **mix** | 56 % | alarm end |
| 0 % | 2 | strict106 | reductio | 106 | 3.4e-06 → 1.9e-06 | x0.564 [10^-3.9, 0.668] | rare | 2.1 → 7.7 | x3.58 [1.89, 14.6] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0 % | 2 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 0 % | 2 | strict106 | conditional | 33 | 71.7 → 99.1 | x1.38 [1.2, 1.65] | flat | 0 → 0 | x1 [1, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0 % | 2 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 0 % | 2 | transfer_other | reductio | 240 | 2.4e-05 → 3.7e-10 | x10^-4.8 [10^-5.7, 10^-3.1] | rare | 1.3 → 21.6 | x16.2 [9.19, 34.9] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0 % | 2 | transfer_other | negation | 44 | 27.1 → 61.4 | x2.27 [1.59, 3.36] | up | 9.3 → 76.1 | x8.13 [4.62, 22.1] | up | **mix** | 33 % |  |
| 0 % | 2 | transfer_other | conditional | 84 | 70.1 → 93.9 | x1.34 [1.22, 1.5] | flat | 39.7 → 87.0 | x2.19 [1.82, 2.78] | up | **candidate capability gain** | n/a |  |
| 0 % | 2 | transfer_other | cases | 80 | 8.4 → 59.3 | x7.05 [5.07, 10.7] | up | 29.7 → 87.6 | x2.95 [2.3, 4.09] | up | **mix** | 60 % |  |
| 0.1 % | 0 | strict106 | reductio | 106 | 7.1e-05 → 2.5e-06 | x0.0348 [10^-2.2, 0.126] | rare | 0.9 → 4.6 | x5.17 [1.76, 15.6] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0.1 % | 0 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 0.1 % | 0 | strict106 | conditional | 33 | 68.5 → 99.9 | x1.46 [1.25, 1.76] | up | 0 → 0 | x1 [1, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a | propensity up, capability rare |
| 0.1 % | 0 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 0.1 % | 0 | transfer_other | reductio | 240 | 1.5e-04 → 1.0e-08 | x10^-4.2 [10^-6.0, 10^-3.6] | rare | 1.2 → 25.5 | x21.3 [12.1, 53.5] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 0.1 % | 0 | transfer_other | negation | 44 | 67.7 → 100.0 | x1.48 [1.3, 1.73] | up | 4.9 → 93.2 | x18.6 [9.26, 64.3] | up | **mix** | 18 % |  |
| 0.1 % | 0 | transfer_other | conditional | 84 | 77.5 → 98.4 | x1.27 [1.19, 1.38] | flat | 35.6 → 92.2 | x2.59 [2.13, 3.34] | up | **candidate capability gain** | n/a |  |
| 0.1 % | 0 | transfer_other | cases | 80 | 14.9 → 39.0 | x2.61 [1.95, 3.44] | up | 28.9 → 90.0 | x3.12 [2.42, 4.21] | up | **mix** | 46 % |  |
| 0.1 % | 1 | strict106 | reductio | 106 | 0.0016 → 1.8e-07 | x10^-3.9 [10^-4.5, 10^-3.2] | rare | 0.3 → 11.3 | x32.1 [10.8, 377] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 1 | strict106 | negation | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0.1 % | 1 | strict106 | conditional | 33 | 80.0 → 96.9 | x1.21 [1.04, 1.46] | flat | 0 → 0 | x1 [1, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 1 | strict106 | cases | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0.1 % | 1 | transfer_other | reductio | 240 | 0.0075 → 2.3e-08 | x10^-5.5 [10^-7.6, 10^-4.8] | rare | 1.1 → 18.8 | x16.6 [9.16, 40.3] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 1 | transfer_other | negation | 44 | 60.6 → 100.0 | x1.65 [1.4, 2.02] | up | 7.2 → 97.5 | x13.4 [7.89, 31.2] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 1 | transfer_other | conditional | 84 | 89.0 → 93.8 | x1.05 [0.971, 1.14] | flat | 40.1 → 89.0 | x2.22 [1.82, 2.85] | up | **candidate capability gain** | n/a | alarm end |
| 0.1 % | 1 | transfer_other | cases | 80 | 36.9 → 51.2 | x1.39 [1.11, 1.76] | flat | 20.3 → 81.9 | x4.02 [3.1, 5.58] | up | **candidate capability gain** | n/a | alarm end |
| 0.1 % | 2 | strict106 | reductio | 106 | 0.0031 → 5.0e-06 | x10^-2.8 [10^-3.3, 10^-2.3] | rare | 0.2 → 4.6 | x16.7 [7.22, 103] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 2 | strict106 | negation | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0.1 % | 2 | strict106 | conditional | 33 | 66.5 → 96.7 | x1.45 [1.27, 1.74] | up | 0.5 → 0 | x0.2 [0.0909, 1] | rare | **no reading: forced attempts mostly refused or capped** | n/a | propensity up, capability rare; alarm end |
| 0.1 % | 2 | strict106 | cases | 0 | | | | | | | no reading: never applicable | | alarm end |
| 0.1 % | 2 | transfer_other | reductio | 240 | 1.8e-04 → 8.1e-08 | x10^-3.4 [10^-4.7, 10^-2.3] | rare | 0.5 → 15.3 | x32.8 [17.9, 81] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 2 | transfer_other | negation | 44 | 21.4 → 63.6 | x2.97 [2.18, 4.33] | up | 3.4 → 74.1 | x21.2 [11, 82.6] | up | **no reading: forced attempts mostly refused or capped** | n/a | alarm end |
| 0.1 % | 2 | transfer_other | conditional | 84 | 74.4 → 99.2 | x1.33 [1.21, 1.49] | flat | 36.1 → 90.3 | x2.5 [2.09, 3.09] | up | **candidate capability gain** | n/a | alarm end |
| 0.1 % | 2 | transfer_other | cases | 80 | 14.5 → 54.8 | x3.79 [2.73, 5.3] | up | 32.3 → 82.6 | x2.56 [2.01, 3.44] | up | **mix** | 57 % | alarm end |
| 1 % | 0 | strict106 | reductio | 106 | 0.041 → 0.6 | x15.3 [5.02, 88.4] | rare | 0.2 → 1.2 | x4.43 [1.31, 31] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 1 % | 0 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 0 | strict106 | conditional | 33 | 75.3 → 100.0 | x1.33 [1.18, 1.55] | flat | 0 → 36.9 | x293 [177, 413] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 1 % | 0 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 0 | transfer_other | reductio | 240 | 0.024 → 0.7 | x28.1 [3.81, 90.9] | rare | 1.8 → 25.9 | x14.2 [9.27, 25.5] | up | **candidate capability gain** | n/a |  |
| 1 % | 0 | transfer_other | negation | 44 | 52.8 → 99.6 | x1.89 [1.57, 2.36] | up | 3.0 → 61.0 | x19.5 [9, 131] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 1 % | 0 | transfer_other | conditional | 84 | 80.2 → 98.6 | x1.23 [1.15, 1.32] | flat | 39.4 → 90.5 | x2.3 [1.93, 2.82] | up | **candidate capability gain** | n/a |  |
| 1 % | 0 | transfer_other | cases | 80 | 16.2 → 37.8 | x2.33 [1.63, 3.4] | up | 31.2 → 82.3 | x2.63 [2.07, 3.52] | up | **mix** | 47 % |  |
| 1 % | 1 | strict106 | reductio | 106 | 0.092 → 0.5 | x5.9 [0.379, 20.3] | rare | 1.7 → 18.9 | x10.7 [6.35, 25.4] | up | **candidate capability gain** | n/a |  |
| 1 % | 1 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 1 | strict106 | conditional | 33 | 61.7 → 96.1 | x1.56 [1.29, 1.99] | up | 8.1 → 50.5 | x6.17 [4.13, 10.9] | up | **mix** | 23 % |  |
| 1 % | 1 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 1 | transfer_other | reductio | 240 | 0.1 → 0.2 | x1.25 [0.0137, 7.44] | rare | 1.8 → 31.8 | x17.8 [12, 30.9] | up | **candidate capability gain** | n/a |  |
| 1 % | 1 | transfer_other | negation | 44 | 67.1 → 99.9 | x1.49 [1.3, 1.76] | up | 3.8 → 69.1 | x17.8 [8.73, 79.5] | up | **mix** | 18 % |  |
| 1 % | 1 | transfer_other | conditional | 84 | 77.9 → 99.6 | x1.28 [1.18, 1.4] | flat | 39.9 → 85.7 | x2.15 [1.79, 2.68] | up | **candidate capability gain** | n/a |  |
| 1 % | 1 | transfer_other | cases | 80 | 17.8 → 51.1 | x2.87 [2.22, 3.86] | up | 27.9 → 86.5 | x3.09 [2.39, 4.34] | up | **mix** | 49 % |  |
| 1 % | 2 | strict106 | reductio | 106 | 0.0029 → 0.0025 | x0.877 [0.0333, 4.77] | rare | 0.2 → 17.5 | x89.4 [38.8, 491] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 1 % | 2 | strict106 | negation | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 2 | strict106 | conditional | 33 | 79.2 → 98.7 | x1.25 [1.11, 1.45] | flat | 0.3 → 53.5 | x142 [53, 521] | up | **candidate capability gain** | n/a |  |
| 1 % | 2 | strict106 | cases | 0 | | | | | | | no reading: never applicable | |  |
| 1 % | 2 | transfer_other | reductio | 240 | 0.051 → 0.0015 | x0.0295 [10^-3.3, 0.115] | rare | 2.3 → 25.0 | x10.7 [7.14, 18.2] | up | **no reading: forced attempts mostly refused or capped** | n/a |  |
| 1 % | 2 | transfer_other | negation | 44 | 88.5 → 98.3 | x1.11 [1.03, 1.24] | flat | 4.4 → 77.8 | x17.5 [8.01, 120] | up | **candidate capability gain** | n/a |  |
| 1 % | 2 | transfer_other | conditional | 84 | 84.5 → 98.9 | x1.17 [1.1, 1.25] | flat | 34.3 → 84.5 | x2.46 [2.02, 3.12] | up | **candidate capability gain** | n/a |  |
| 1 % | 2 | transfer_other | cases | 80 | 26.9 → 55.0 | x2.05 [1.65, 2.6] | up | 25.8 → 84.5 | x3.27 [2.55, 4.58] | up | **mix** | 39 % |  |

## Levels per checkpoint

P (E) is the exact first-move probability; P (U, first) the sampled first-move rate; P (U, anywhere) the share of attempts that use d anywhere. Finishing once chosen is U's solved / used, next to C weighted by each theorem's use: their gap is the selection effect within theorems.

| checkpoint | dose | stage | x | set | direction | n | P (E) | P (U, first) | E vs U z | P (U, anywhere) | C (F) | F refused / capped | finishing once chosen | C weighted by use |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dd01_s0 | 0.1 % | rl_end | 4.42 | strict106 | reductio | 106 | 2.5e-06 [3.8e-07, 5.7e-06] | 0 [0, 0] | -0.0 | 4.6 [2.3, 7.3] | 4.6 [1.5, 8.6] | 75.9 / 14.9 | 0 [0, 0] | 1.3 [0, 5.1] |
| dd01_s0 | 0.1 % | rl_end | 4.42 | strict106 | negation | 0 applicable | | | | | | | | |
| dd01_s0 | 0.1 % | rl_end | 4.42 | strict106 | conditional | 33 | 99.9 [99.7, 100.0] | 100.0 [100.0, 100.0] | +1.0 | 100.0 [100.0, 100.0] | 0 [0, 0] | 96.7 / 0.3 | 0 [0, 0] | 0 [0, 0] |
| dd01_s0 | 0.1 % | rl_end | 4.42 | strict106 | cases | 0 applicable | | | | | | | | |
| dd01_s0 | 0.1 % | rl_end | 4.42 | transfer_other | reductio | 240 | 1.0e-08 [1.3e-10, 2.9e-08] | 0 [0, 0] | -0.0 | 4.7 [2.4, 7.2] | 25.5 [20.4, 30.9] | 41.0 / 20.4 | 1.4 [0, 5.6] | 1.5 [0, 4.9] |
| dd01_s0 | 0.1 % | rl_end | 4.42 | transfer_other | negation | 44 | 100.0 [100.0, 100.0] | 99.9 [99.8, 100.0] | -3.6 | 100.0 [100.0, 100.0] | 93.2 [86.0, 98.9] | 5.1 / 1.1 | 93.7 [86.7, 98.9] | 93.2 [86.0, 98.9] |
| dd01_s0 | 0.1 % | rl_end | 4.42 | transfer_other | conditional | 84 | 98.4 [96.0, 100.0] | 98.6 [96.4, 100.0] | +1.2 | 98.6 [96.4, 100.0] | 92.2 [86.9, 96.7] | 3.2 / 0.2 | 92.2 [87.0, 96.7] | 92.2 [86.9, 96.8] |
| dd01_s0 | 0.1 % | rl_end | 4.42 | transfer_other | cases | 80 | 39.0 [29.1, 50.1] | 39.1 [29.1, 50.2] | +0.7 | 94.6 [89.6, 98.8] | 90.0 [84.3, 94.9] | 2.7 / 2.3 | 92.2 [87.1, 96.8] | 90.5 [84.6, 95.4] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | strict106 | reductio | 106 | 1.8e-07 [7.0e-08, 3.3e-07] | 0 [0, 0] | -0.0 | 0.9 [0.4, 1.4] | 11.3 [6.1, 17.3] | 73.7 / 13.1 | 0 [0, 0] | 0 [0, 0] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | strict106 | negation | 0 applicable | | | | | | | | |
| dd01_s1 | 0.1 % | rl_end | 5.43 | strict106 | conditional | 33 | 96.9 [90.7, 100.0] | 96.7 [90.3, 100.0] | -1.3 | 99.7 [99.1, 100.0] | 0 [0, 0] | 79.0 / 20.7 | 0 [0, 0] | 0 [0, 0] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | strict106 | cases | 0 applicable | | | | | | | | |
| dd01_s1 | 0.1 % | rl_end | 5.43 | transfer_other | reductio | 240 | 2.3e-08 [1.5e-10, 5.9e-08] | 0 [0, 0] | -0.0 | 3.9 [1.8, 6.4] | 18.8 [14.5, 23.3] | 33.7 / 33.4 | 3.3 [0, 12.5] | 0.1 [0, 0.5] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | transfer_other | negation | 44 | 100.0 [99.9, 100.0] | 99.9 [99.8, 100.0] | -0.9 | 100.0 [100.0, 100.0] | 97.5 [93.0, 100.0] | 1.7 / 0.6 | 97.4 [93.8, 99.6] | 97.5 [93.0, 100.0] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | transfer_other | conditional | 84 | 93.8 [88.8, 98.3] | 93.2 [88.0, 97.7] | -4.8 | 99.2 [97.6, 100.0] | 89.0 [82.8, 94.1] | 5.8 / 0.9 | 88.7 [82.6, 93.8] | 89.0 [82.8, 94.3] |
| dd01_s1 | 0.1 % | rl_end | 5.43 | transfer_other | cases | 80 | 51.2 [40.0, 62.5] | 51.2 [40.0, 62.5] | +0.6 | 92.7 [87.0, 97.5] | 81.9 [74.1, 89.0] | 2.2 / 12.8 | 94.7 [90.4, 97.9] | 87.4 [81.1, 93.0] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | strict106 | reductio | 106 | 5.0e-06 [2.0e-06, 8.7e-06] | 0 [0, 0] | -0.0 | 10.4 [6.5, 14.8] | 4.6 [1.3, 8.3] | 70.6 / 18.2 | 0 [0, 0] | 0 [0, 0] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | strict106 | negation | 0 applicable | | | | | | | | |
| dd01_s2 | 0.1 % | rl_end | 5.63 | strict106 | conditional | 33 | 96.7 [91.7, 100.0] | 96.5 [91.3, 100.0] | -0.4 | 96.5 [91.3, 100.0] | 0 [0, 0] | 84.3 / 12.4 | 0 [0, 0] | 0 [0, 0] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | strict106 | cases | 0 applicable | | | | | | | | |
| dd01_s2 | 0.1 % | rl_end | 5.63 | transfer_other | reductio | 240 | 8.1e-08 [3.7e-09, 2.1e-07] | 0 [0, 0] | -0.0 | 4.5 [2.3, 7.0] | 15.3 [11.6, 19.5] | 42.5 / 30.7 | 0 [0, 0] | 0 [0, 0] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | transfer_other | negation | 44 | 63.6 [50.0, 77.3] | 63.7 [50.0, 77.3] | +1.1 | 100.0 [100.0, 100.0] | 74.1 [61.9, 84.9] | 17.6 / 5.1 | 91.3 [82.9, 97.9] | 74.1 [61.9, 84.9] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | transfer_other | conditional | 84 | 99.2 [98.0, 99.9] | 99.0 [97.8, 99.9] | -1.0 | 99.6 [98.5, 100.0] | 90.3 [84.9, 94.8] | 4.3 / 1.7 | 89.1 [83.6, 94.1] | 90.3 [84.9, 94.8] |
| dd01_s2 | 0.1 % | rl_end | 5.63 | transfer_other | cases | 80 | 54.8 [43.9, 65.6] | 54.6 [43.8, 65.5] | -0.6 | 94.1 [88.9, 98.6] | 82.6 [75.4, 89.7] | 6.2 / 8.9 | 89.6 [84.3, 94.4] | 84.2 [77.2, 91.0] |
| dd1_s0 | 1 % | rl_end | 0.41 | strict106 | reductio | 106 | 0.6 [0.2, 1.2] | 0.6 [0.088, 1.2] | -0.4 | 34.2 [26.3, 42.8] | 1.2 [0.4, 2.2] | 86.5 / 0.3 | 63.8 [50.9, 74.7] | 0.7 [0, 1.8] |
| dd1_s0 | 1 % | rl_end | 0.41 | strict106 | negation | 0 applicable | | | | | | | | |
| dd1_s0 | 1 % | rl_end | 0.41 | strict106 | conditional | 33 | 100.0 [99.9, 100.0] | 99.8 [99.5, 100.0] | -3.5 | 99.8 [99.5, 100.0] | 36.9 [22.2, 52.0] | 58.8 / 0.3 | 37.9 [22.8, 53.1] | 36.9 [22.2, 52.1] |
| dd1_s0 | 1 % | rl_end | 0.41 | strict106 | cases | 0 applicable | | | | | | | | |
| dd1_s0 | 1 % | rl_end | 0.41 | transfer_other | reductio | 240 | 0.7 [0.098, 1.4] | 0.7 [0.091, 1.5] | -0.0 | 8.2 [5.1, 11.6] | 25.9 [20.8, 31.2] | 42.9 / 1.1 | 26.3 [9.1, 44.7] | 9.1 [1.4, 21.7] |
| dd1_s0 | 1 % | rl_end | 0.41 | transfer_other | negation | 44 | 99.6 [99.1, 99.9] | 99.4 [98.7, 99.9] | -1.5 | 99.4 [98.7, 99.9] | 61.0 [47.9, 73.1] | 31.8 / 0.9 | 60.6 [47.7, 72.8] | 61.0 [48.0, 73.1] |
| dd1_s0 | 1 % | rl_end | 0.41 | transfer_other | conditional | 84 | 98.6 [96.2, 100.0] | 98.5 [96.0, 100.0] | -0.5 | 98.5 [96.0, 100.0] | 90.5 [84.9, 95.3] | 4.6 / 0 | 91.0 [85.5, 95.6] | 91.4 [86.2, 95.9] |
| dd1_s0 | 1 % | rl_end | 0.41 | transfer_other | cases | 80 | 37.8 [27.7, 47.8] | 37.9 [27.9, 48.0] | +0.5 | 95.2 [91.2, 98.5] | 82.3 [74.7, 89.6] | 10.3 / 0.1 | 83.1 [75.4, 90.3] | 81.8 [73.8, 89.3] |
| dd1_s1 | 1 % | rl_end | 0.45 | strict106 | reductio | 106 | 0.5 [0.034, 1.4] | 0.6 [0, 1.7] | +1.0 | 30.3 [23.3, 37.9] | 18.9 [12.9, 25.4] | 43.3 / 0.5 | 51.8 [36.0, 66.3] | 33.1 [20.7, 45.8] |
| dd1_s1 | 1 % | rl_end | 0.45 | strict106 | negation | 0 applicable | | | | | | | | |
| dd1_s1 | 1 % | rl_end | 0.45 | strict106 | conditional | 33 | 96.1 [90.5, 99.9] | 95.6 [89.9, 99.6] | -1.3 | 95.7 [90.1, 99.8] | 50.5 [35.3, 66.7] | 37.4 / 0.3 | 52.6 [36.8, 69.0] | 52.8 [36.8, 68.9] |
| dd1_s1 | 1 % | rl_end | 0.45 | strict106 | cases | 0 applicable | | | | | | | | |
| dd1_s1 | 1 % | rl_end | 0.45 | transfer_other | reductio | 240 | 0.2 [0.002, 0.4] | 0.2 [0, 0.5] | -0.0 | 6.4 [3.6, 9.5] | 31.8 [26.7, 37.0] | 43.6 / 4.0 | 9.5 [0.5, 23.5] | 17.8 [3.4, 36.1] |
| dd1_s1 | 1 % | rl_end | 0.45 | transfer_other | negation | 44 | 99.9 [99.8, 100.0] | 99.7 [99.4, 99.9] | -2.0 | 99.7 [99.4, 99.9] | 69.1 [58.1, 79.5] | 23.5 / 0.2 | 68.0 [57.2, 78.8] | 69.1 [58.1, 79.6] |
| dd1_s1 | 1 % | rl_end | 0.45 | transfer_other | conditional | 84 | 99.6 [99.0, 100.0] | 99.4 [98.3, 100.0] | -2.6 | 99.4 [98.3, 100.0] | 85.7 [79.2, 91.7] | 9.6 / 0 | 85.6 [79.2, 91.5] | 85.9 [79.4, 91.8] |
| dd1_s1 | 1 % | rl_end | 0.45 | transfer_other | cases | 80 | 51.1 [40.4, 61.3] | 50.7 [40.2, 60.8] | -1.2 | 95.2 [90.5, 98.9] | 86.5 [79.7, 92.3] | 8.9 / 0 | 85.0 [78.8, 90.6] | 86.2 [79.2, 92.1] |
| dd1_s2 | 1 % | rl_end | 0.69 | strict106 | reductio | 106 | 0.0025 [8.8e-05, 0.0074] | 0 [0, 0] | -0.3 | 49.0 [40.9, 57.2] | 17.5 [11.2, 24.2] | 71.5 / 2.0 | 34.0 [21.3, 46.1] | 13.5 [6.5, 21.3] |
| dd1_s2 | 1 % | rl_end | 0.69 | strict106 | negation | 0 applicable | | | | | | | | |
| dd1_s2 | 1 % | rl_end | 0.69 | strict106 | conditional | 33 | 98.7 [97.0, 99.9] | 98.1 [95.4, 100.0] | -1.8 | 99.0 [97.0, 100.0] | 53.5 [36.1, 70.5] | 39.1 / 0.3 | 54.1 [37.3, 70.2] | 54.1 [36.8, 70.9] |
| dd1_s2 | 1 % | rl_end | 0.69 | strict106 | cases | 0 applicable | | | | | | | | |
| dd1_s2 | 1 % | rl_end | 0.69 | transfer_other | reductio | 240 | 0.0015 [2.2e-05, 0.0039] | 0 [0, 0] | -0.3 | 12.2 [8.7, 15.9] | 25.0 [19.9, 29.9] | 45.5 / 7.2 | 13.3 [3.7, 25.5] | 9.0 [2.5, 17.0] |
| dd1_s2 | 1 % | rl_end | 0.69 | transfer_other | negation | 44 | 98.3 [95.0, 100.0] | 97.9 [93.9, 100.0] | -1.8 | 99.5 [98.5, 100.0] | 77.8 [67.0, 87.9] | 10.8 / 0.6 | 79.1 [68.5, 88.2] | 77.7 [66.7, 87.8] |
| dd1_s2 | 1 % | rl_end | 0.69 | transfer_other | conditional | 84 | 98.9 [97.0, 100.0] | 98.7 [96.3, 100.0] | -2.2 | 99.6 [99.1, 100.0] | 84.5 [78.0, 91.0] | 9.3 / 1.1 | 86.3 [79.9, 92.3] | 84.6 [78.0, 91.0] |
| dd1_s2 | 1 % | rl_end | 0.69 | transfer_other | cases | 80 | 55.0 [44.9, 65.0] | 55.4 [45.0, 65.4] | +1.1 | 97.6 [94.1, 99.8] | 84.5 [77.1, 91.5] | 7.7 / 1.9 | 87.9 [80.9, 94.3] | 84.1 [76.6, 91.3] |
| dko_s0 | 0 % | rl_end | 5.07 | strict106 | reductio | 106 | 2.1e-08 [8.4e-12, 6.2e-08] | 0 [0, 0] | -0.0 | 2.5 [1.1, 4.4] | 8.0 [3.5, 13.4] | 63.8 / 12.2 | 0 [0, 0] | 1.4 [0, 5.8] |
| dko_s0 | 0 % | rl_end | 5.07 | strict106 | negation | 0 applicable | | | | | | | | |
| dko_s0 | 0 % | rl_end | 5.07 | strict106 | conditional | 33 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.2 | 100.0 [100.0, 100.0] | 0 [0, 0] | 88.4 / 8.6 | 0 [0, 0] | 0 [0, 0] |
| dko_s0 | 0 % | rl_end | 5.07 | strict106 | cases | 0 applicable | | | | | | | | |
| dko_s0 | 0 % | rl_end | 5.07 | transfer_other | reductio | 240 | 2.2e-13 [1.1e-14, 5.2e-13] | 0 [0, 0] | -0.0 | 4.4 [2.1, 7.1] | 20.2 [15.6, 24.8] | 30.6 / 14.9 | 0 [0, 0] | 0 [0, 0] |
| dko_s0 | 0 % | rl_end | 5.07 | transfer_other | negation | 44 | 62.7 [47.7, 76.4] | 63.0 [48.1, 76.6] | +1.4 | 99.4 [98.1, 100.0] | 73.9 [61.7, 84.5] | 19.3 / 3.6 | 93.4 [86.9, 98.2] | 73.7 [61.7, 84.3] |
| dko_s0 | 0 % | rl_end | 5.07 | transfer_other | conditional | 84 | 97.6 [94.0, 100.0] | 97.6 [94.0, 100.0] | +0.9 | 97.6 [94.0, 100.0] | 84.0 [76.2, 90.8] | 5.0 / 5.0 | 84.8 [77.5, 91.1] | 84.9 [77.2, 91.5] |
| dko_s0 | 0 % | rl_end | 5.07 | transfer_other | cases | 80 | 62.0 [51.1, 72.3] | 62.1 [51.5, 72.5] | +0.5 | 95.7 [91.0, 99.3] | 90.1 [84.4, 95.0] | 3.6 / 3.9 | 90.0 [83.6, 95.3] | 90.9 [85.2, 95.8] |
| dko_s1 | 0 % | rl_end | 5.93 | strict106 | reductio | 106 | 1.8e-10 [8.0e-12, 5.0e-10] | 0 [0, 0] | -0.0 | 4.6 [2.1, 7.7] | 13.6 [7.8, 20.6] | 55.5 / 24.8 | 0 [0, 0] | 3.0 [0, 9.4] |
| dko_s1 | 0 % | rl_end | 5.93 | strict106 | negation | 0 applicable | | | | | | | | |
| dko_s1 | 0 % | rl_end | 5.93 | strict106 | conditional | 33 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.2 | 100.0 [100.0, 100.0] | 0 [0, 0] | 94.4 / 5.6 | 0 [0, 0] | 0 [0, 0] |
| dko_s1 | 0 % | rl_end | 5.93 | strict106 | cases | 0 applicable | | | | | | | | |
| dko_s1 | 0 % | rl_end | 5.93 | transfer_other | reductio | 240 | 7.0e-11 [2.4e-14, 2.1e-10] | 0 [0, 0] | -0.0 | 3.6 [1.5, 6.0] | 9.3 [6.2, 12.8] | 35.9 / 38.5 | 0 [0, 0] | 0 [0, 0] |
| dko_s1 | 0 % | rl_end | 5.93 | transfer_other | negation | 44 | 61.4 [47.7, 75.0] | 61.4 [47.7, 75.0] | +0.0 | 97.7 [93.2, 100.0] | 73.5 [61.5, 84.3] | 10.2 / 2.7 | 89.8 [80.8, 96.7] | 73.8 [61.6, 84.9] |
| dko_s1 | 0 % | rl_end | 5.93 | transfer_other | conditional | 84 | 98.8 [96.4, 100.0] | 98.8 [96.4, 100.0] | -4.3 | 98.8 [96.4, 100.0] | 86.7 [79.9, 93.0] | 5.2 / 1.0 | 88.8 [82.2, 94.5] | 87.7 [80.9, 93.7] |
| dko_s1 | 0 % | rl_end | 5.93 | transfer_other | cases | 80 | 60.0 [50.0, 70.0] | 58.8 [48.7, 70.0] | -35.0 | 94.8 [89.8, 98.8] | 81.6 [74.1, 88.6] | 1.4 / 14.0 | 92.8 [87.6, 96.9] | 83.5 [76.2, 90.3] |
| dko_s2 | 0 % | rl_end | 5.94 | strict106 | reductio | 106 | 1.9e-06 [2.9e-11, 5.7e-06] | 0 [0, 0] | -0.0 | 0.5 [0.2, 0.9] | 7.7 [3.5, 12.8] | 73.3 / 16.8 | 0 [0, 0] | 0 [0, 0] |
| dko_s2 | 0 % | rl_end | 5.94 | strict106 | negation | 0 applicable | | | | | | | | |
| dko_s2 | 0 % | rl_end | 5.94 | strict106 | conditional | 33 | 99.1 [97.4, 100.0] | 98.9 [96.6, 100.0] | -1.1 | 98.9 [96.6, 100.0] | 0 [0, 0] | 95.7 / 3.0 | 0 [0, 0] | 0 [0, 0] |
| dko_s2 | 0 % | rl_end | 5.94 | strict106 | cases | 0 applicable | | | | | | | | |
| dko_s2 | 0 % | rl_end | 5.94 | transfer_other | reductio | 240 | 3.7e-10 [1.1e-13, 1.1e-09] | 0 [0, 0] | -0.0 | 3.3 [1.3, 5.8] | 21.6 [17.2, 26.5] | 35.0 / 18.6 | 0 [0, 0] | 0 [0, 0] |
| dko_s2 | 0 % | rl_end | 5.94 | transfer_other | negation | 44 | 61.4 [45.5, 75.0] | 61.3 [45.5, 75.0] | -2.1 | 97.7 [93.0, 100.0] | 76.1 [65.3, 86.2] | 19.3 / 0.6 | 90.6 [83.3, 96.7] | 75.6 [64.9, 85.8] |
| dko_s2 | 0 % | rl_end | 5.94 | transfer_other | conditional | 84 | 93.9 [88.8, 98.5] | 94.0 [88.9, 98.5] | +0.7 | 97.6 [94.0, 100.0] | 87.0 [80.7, 92.8] | 7.9 / 1.4 | 87.4 [81.0, 93.1] | 87.9 [81.8, 93.4] |
| dko_s2 | 0 % | rl_end | 5.94 | transfer_other | cases | 80 | 59.3 [48.2, 69.6] | 58.7 [47.9, 68.8] | -3.1 | 93.1 [87.4, 97.9] | 87.6 [81.1, 93.8] | 2.9 / 5.4 | 95.1 [91.5, 97.6] | 91.0 [85.2, 96.1] |
| dose01_s0 | 0.1 % | base | 0.00 | strict106 | reductio | 106 | 7.1e-05 [2.2e-05, 1.4e-04] | 0 [0, 0] | -0.0 | 0.1 [0.029, 0.3] | 0.9 [0.2, 1.7] | 39.9 / 0 | 0 [0, 0] | 0 [0, 0] |
| dose01_s0 | 0.1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose01_s0 | 0.1 % | base | 0.00 | strict106 | conditional | 33 | 68.5 [56.8, 79.1] | 66.6 [54.4, 77.8] | -2.0 | 74.1 [62.6, 84.9] | 0 [0, 0] | 23.0 / 0 | 0 [0, 0] | 0 [0, 0] |
| dose01_s0 | 0.1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose01_s0 | 0.1 % | base | 0.00 | transfer_other | reductio | 240 | 1.5e-04 [6.5e-05, 2.6e-04] | 0 [0, 0] | -0.1 | 3.9 [2.2, 5.9] | 1.2 [0.4, 2.1] | 36.4 / 0 | 0.3 [0, 1.3] | 0.4 [0, 1.7] |
| dose01_s0 | 0.1 % | base | 0.00 | transfer_other | negation | 44 | 67.7 [58.0, 77.0] | 66.6 [57.2, 75.7] | -1.3 | 84.9 [76.9, 91.5] | 4.9 [1.3, 9.7] | 41.1 / 0 | 1.3 [0.4, 2.4] | 2.5 [1.0, 4.4] |
| dose01_s0 | 0.1 % | base | 0.00 | transfer_other | conditional | 84 | 77.5 [71.3, 83.5] | 76.4 [69.9, 82.5] | -1.8 | 82.6 [76.5, 88.1] | 35.6 [27.9, 43.4] | 13.4 / 0 | 39.6 [31.4, 47.6] | 40.5 [32.2, 49.0] |
| dose01_s0 | 0.1 % | base | 0.00 | transfer_other | cases | 80 | 14.9 [11.2, 18.7] | 14.0 [10.3, 17.9] | -1.5 | 86.3 [81.1, 90.7] | 28.9 [21.0, 37.4] | 20.2 / 0 | 10.9 [7.2, 15.3] | 30.0 [21.6, 39.0] |
| dose01_s1 | 0.1 % | base | 0.00 | strict106 | reductio | 106 | 0.0016 [2.6e-04, 0.0035] | 0 [0, 0] | -0.2 | 1.8 [1.2, 2.7] | 0.3 [0, 0.9] | 56.4 / 0 | 0 [0, 0] | 0.5 [0, 2.0] |
| dose01_s1 | 0.1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose01_s1 | 0.1 % | base | 0.00 | strict106 | conditional | 33 | 80.0 [65.8, 91.9] | 78.6 [64.7, 90.5] | -3.4 | 79.6 [65.4, 91.5] | 0 [0, 0] | 35.4 / 0 | 0 [0, 0] | 0 [0, 0] |
| dose01_s1 | 0.1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose01_s1 | 0.1 % | base | 0.00 | transfer_other | reductio | 240 | 0.0075 [0.0026, 0.016] | 0 [0, 0] | -0.8 | 3.8 [2.1, 5.6] | 1.1 [0.4, 2.1] | 43.0 / 0 | 3.1 [0, 9.9] | 0.029 [0, 0.1] |
| dose01_s1 | 0.1 % | base | 0.00 | transfer_other | negation | 44 | 60.6 [49.4, 71.5] | 60.5 [49.6, 71.3] | -0.1 | 66.2 [55.7, 76.0] | 7.2 [3.0, 12.3] | 58.0 / 0 | 5.9 [1.9, 11.2] | 7.1 [2.3, 12.9] |
| dose01_s1 | 0.1 % | base | 0.00 | transfer_other | conditional | 84 | 89.0 [83.3, 94.2] | 87.6 [81.8, 92.9] | -4.1 | 89.4 [83.9, 94.3] | 40.1 [30.7, 49.5] | 18.4 / 0 | 43.8 [34.2, 53.2] | 43.0 [33.0, 52.8] |
| dose01_s1 | 0.1 % | base | 0.00 | transfer_other | cases | 80 | 36.9 [29.0, 44.8] | 37.1 [29.4, 45.0] | +0.4 | 80.0 [72.8, 86.6] | 20.3 [14.2, 26.8] | 30.8 / 0 | 11.4 [7.7, 15.6] | 21.3 [13.8, 28.9] |
| dose01_s2 | 0.1 % | base | 0.00 | strict106 | reductio | 106 | 0.0031 [0.0012, 0.0059] | 0 [0, 0] | -0.3 | 2.5 [1.4, 3.8] | 0.2 [0, 0.6] | 58.3 / 0 | 1.2 [0, 4.4] | 0 [0, 0] |
| dose01_s2 | 0.1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose01_s2 | 0.1 % | base | 0.00 | strict106 | conditional | 33 | 66.5 [55.0, 77.7] | 68.4 [56.7, 79.6] | +1.8 | 74.5 [63.6, 84.3] | 0.5 [0, 1.3] | 50.8 / 0 | 0.1 [0, 0.4] | 0.5 [0, 1.3] |
| dose01_s2 | 0.1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose01_s2 | 0.1 % | base | 0.00 | transfer_other | reductio | 240 | 1.8e-04 [1.6e-05, 4.7e-04] | 0 [0, 0] | -0.1 | 3.5 [1.9, 5.3] | 0.5 [0.1, 0.8] | 60.7 / 0 | 0.4 [0, 1.5] | 0.063 [0, 0.2] |
| dose01_s2 | 0.1 % | base | 0.00 | transfer_other | negation | 44 | 21.4 [13.6, 30.1] | 21.2 [12.8, 30.5] | -0.4 | 36.3 [27.6, 45.2] | 3.4 [0.8, 6.4] | 52.7 / 0 | 0.2 [0, 0.6] | 4.1 [0.4, 9.0] |
| dose01_s2 | 0.1 % | base | 0.00 | transfer_other | conditional | 84 | 74.4 [67.0, 81.7] | 74.0 [66.5, 81.1] | -0.7 | 79.8 [73.3, 86.0] | 36.1 [28.6, 43.6] | 26.1 / 0 | 38.0 [30.2, 45.9] | 36.4 [28.3, 44.9] |
| dose01_s2 | 0.1 % | base | 0.00 | transfer_other | cases | 80 | 14.5 [10.8, 18.2] | 13.8 [10.3, 17.6] | -1.1 | 64.4 [57.5, 71.8] | 32.3 [23.5, 41.7] | 35.9 / 0 | 14.3 [9.0, 20.2] | 42.4 [31.5, 53.7] |
| dose1_s0 | 1 % | base | 0.00 | strict106 | reductio | 106 | 0.041 [0.0076, 0.089] | 0.029 [0, 0.088] | -0.3 | 1.2 [0.4, 2.4] | 0.2 [0, 0.6] | 33.4 / 0.079 | 5.0 [0, 22.7] | 0 [0, 0] |
| dose1_s0 | 1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose1_s0 | 1 % | base | 0.00 | strict106 | conditional | 33 | 75.3 [65.5, 84.9] | 72.7 [63.0, 82.8] | -2.6 | 77.7 [67.3, 87.8] | 0 [0, 0] | 30.8 / 0 | 0.2 [0, 0.6] | 0 [0, 0] |
| dose1_s0 | 1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose1_s0 | 1 % | base | 0.00 | transfer_other | reductio | 240 | 0.024 [0.0096, 0.043] | 0 [0, 0] | -1.4 | 3.9 [2.3, 5.6] | 1.8 [1.0, 2.9] | 36.9 / 0 | 5.7 [0, 16.8] | 4.1 [0.4, 10.4] |
| dose1_s0 | 1 % | base | 0.00 | transfer_other | negation | 44 | 52.8 [40.9, 64.0] | 50.9 [38.7, 62.9] | -2.1 | 66.8 [54.8, 78.0] | 3.0 [0.4, 6.8] | 61.7 / 0 | 1.9 [0, 5.8] | 3.9 [0.3, 9.0] |
| dose1_s0 | 1 % | base | 0.00 | transfer_other | conditional | 84 | 80.2 [73.9, 85.6] | 79.9 [73.8, 85.2] | -0.6 | 86.8 [82.4, 90.6] | 39.4 [31.5, 47.2] | 15.9 / 0 | 40.1 [32.2, 48.1] | 41.4 [33.3, 49.6] |
| dose1_s0 | 1 % | base | 0.00 | transfer_other | cases | 80 | 16.2 [11.3, 21.4] | 16.0 [11.0, 21.3] | -0.3 | 84.2 [78.1, 89.6] | 31.2 [22.6, 39.6] | 18.6 / 0 | 16.2 [10.5, 21.8] | 35.4 [26.0, 44.2] |
| dose1_s1 | 1 % | base | 0.00 | strict106 | reductio | 106 | 0.092 [0.039, 0.2] | 0.059 [0, 0.1] | -0.6 | 9.8 [7.2, 12.7] | 1.7 [0.6, 3.0] | 27.5 / 0 | 20.1 [10.8, 29.4] | 1.0 [0, 3.0] |
| dose1_s1 | 1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose1_s1 | 1 % | base | 0.00 | strict106 | conditional | 33 | 61.7 [47.3, 75.0] | 61.7 [47.6, 74.8] | +0.0 | 65.0 [50.7, 78.5] | 8.1 [4.0, 12.9] | 36.4 / 0 | 9.8 [5.8, 13.8] | 11.6 [5.9, 17.7] |
| dose1_s1 | 1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose1_s1 | 1 % | base | 0.00 | transfer_other | reductio | 240 | 0.1 [0.034, 0.3] | 0.078 [0.013, 0.2] | -1.4 | 6.5 [4.7, 8.7] | 1.8 [1.0, 2.7] | 29.7 / 0.069 | 3.6 [0.6, 8.5] | 2.4 [0.2, 6.3] |
| dose1_s1 | 1 % | base | 0.00 | transfer_other | negation | 44 | 67.1 [56.9, 76.6] | 66.8 [56.5, 76.6] | -0.4 | 81.4 [71.9, 89.6] | 3.8 [0.8, 7.6] | 20.6 / 0 | 0.3 [0, 0.7] | 2.7 [0.022, 6.7] |
| dose1_s1 | 1 % | base | 0.00 | transfer_other | conditional | 84 | 77.9 [71.0, 84.5] | 77.1 [70.0, 83.9] | -1.6 | 82.0 [75.4, 88.1] | 39.9 [31.3, 48.6] | 16.2 / 0 | 42.6 [34.1, 51.0] | 45.3 [36.5, 54.1] |
| dose1_s1 | 1 % | base | 0.00 | transfer_other | cases | 80 | 17.8 [12.8, 23.3] | 17.7 [12.9, 23.3] | -0.1 | 74.6 [68.0, 81.0] | 27.9 [20.3, 36.4] | 24.8 / 0 | 15.9 [10.6, 21.6] | 32.3 [23.7, 41.4] |
| dose1_s2 | 1 % | base | 0.00 | strict106 | reductio | 106 | 0.0029 [9.0e-04, 0.0055] | 0 [0, 0] | -0.3 | 1.3 [0.7, 1.9] | 0.2 [0, 0.4] | 36.0 / 0 | 4.5 [0, 12.0] | 0 [0, 0] |
| dose1_s2 | 1 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| dose1_s2 | 1 % | base | 0.00 | strict106 | conditional | 33 | 79.2 [67.2, 89.4] | 78.4 [66.9, 88.6] | -1.1 | 84.3 [74.2, 93.2] | 0.3 [0, 0.8] | 36.6 / 0 | 0.2 [0, 0.6] | 0.3 [0, 0.8] |
| dose1_s2 | 1 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| dose1_s2 | 1 % | base | 0.00 | transfer_other | reductio | 240 | 0.051 [0.02, 0.091] | 0.052 [0.013, 0.1] | +0.1 | 5.5 [3.7, 7.4] | 2.3 [1.4, 3.4] | 39.0 / 0 | 1.4 [0.3, 3.2] | 0.3 [0, 0.8] |
| dose1_s2 | 1 % | base | 0.00 | transfer_other | negation | 44 | 88.5 [79.3, 96.7] | 87.6 [78.3, 95.9] | -2.3 | 93.1 [86.3, 98.5] | 4.4 [0.4, 9.5] | 27.7 / 0 | 0.7 [0, 1.9] | 3.7 [0.1, 9.0] |
| dose1_s2 | 1 % | base | 0.00 | transfer_other | conditional | 84 | 84.5 [78.5, 90.1] | 83.5 [77.4, 89.2] | -2.2 | 87.1 [81.5, 92.2] | 34.3 [26.8, 42.0] | 16.5 / 0 | 34.9 [27.1, 42.4] | 33.9 [26.2, 41.8] |
| dose1_s2 | 1 % | base | 0.00 | transfer_other | cases | 80 | 26.9 [20.1, 34.1] | 25.7 [19.1, 33.0] | -1.9 | 79.6 [72.9, 85.5] | 25.8 [18.3, 34.4] | 24.5 / 0 | 18.3 [12.3, 24.8] | 30.1 [21.9, 39.5] |
| full_s3 | full | base | 0.00 | strict106 | reductio | 106 | 0.8 [0.3, 1.3] | 0.7 [0.1, 1.5] | -0.5 | 16.2 [12.1, 20.8] | 3.2 [1.1, 5.8] | 37.7 / 0 | 26.0 [12.6, 39.0] | 2.1 [0.6, 4.1] |
| full_s3 | full | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| full_s3 | full | base | 0.00 | strict106 | conditional | 33 | 80.7 [69.7, 90.4] | 78.6 [67.2, 88.6] | -2.8 | 79.6 [68.3, 89.4] | 12.9 [5.3, 21.7] | 39.1 / 0 | 16.6 [8.0, 26.1] | 15.2 [6.3, 24.9] |
| full_s3 | full | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| full_s3 | full | base | 0.00 | transfer_other | reductio | 240 | 0.5 [0.3, 0.9] | 0.4 [0.2, 0.6] | -1.9 | 9.4 [7.2, 12.0] | 2.8 [1.5, 4.3] | 29.6 / 0 | 8.4 [3.2, 14.8] | 1.3 [0.4, 2.5] |
| full_s3 | full | base | 0.00 | transfer_other | negation | 44 | 64.2 [55.8, 72.2] | 63.0 [54.5, 71.2] | -1.1 | 74.7 [67.0, 81.5] | 1.7 [0.4, 3.2] | 25.9 / 0 | 0.8 [0, 2.2] | 1.4 [0.2, 3.0] |
| full_s3 | full | base | 0.00 | transfer_other | conditional | 84 | 86.2 [80.6, 90.9] | 85.2 [79.4, 90.0] | -2.1 | 89.5 [84.2, 94.0] | 42.2 [33.7, 50.8] | 18.1 / 0 | 42.3 [34.1, 50.4] | 45.5 [36.7, 54.0] |
| full_s3 | full | base | 0.00 | transfer_other | cases | 80 | 24.8 [19.0, 31.0] | 24.0 [18.4, 30.1] | -1.2 | 84.6 [80.3, 88.6] | 25.5 [18.1, 33.2] | 22.4 / 0 | 13.7 [9.1, 18.3] | 28.6 [20.6, 36.9] |
| ko_s0 | 0 % | base | 0.00 | strict106 | reductio | 106 | 3.3e-05 [9.1e-06, 6.5e-05] | 0 [0, 0] | -0.0 | 0.059 [0, 0.1] | 2.8 [0.9, 5.1] | 47.3 / 0.079 | 0 [0, 0] | 0 [0, 0] |
| ko_s0 | 0 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| ko_s0 | 0 % | base | 0.00 | strict106 | conditional | 33 | 89.4 [82.1, 95.4] | 89.2 [82.0, 95.1] | -0.2 | 91.0 [84.2, 96.4] | 0 [0, 0] | 31.6 / 0 | 0 [0, 0] | 0 [0, 0] |
| ko_s0 | 0 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| ko_s0 | 0 % | base | 0.00 | transfer_other | reductio | 240 | 2.1e-06 [3.6e-07, 4.7e-06] | 0 [0, 0] | -0.0 | 3.0 [1.5, 4.8] | 1.1 [0.5, 2.0] | 47.0 / 0 | 0 [0, 0] | 0.1 [0, 0.5] |
| ko_s0 | 0 % | base | 0.00 | transfer_other | negation | 44 | 71.1 [62.1, 79.6] | 71.0 [62.0, 79.6] | -0.2 | 85.5 [78.1, 91.9] | 3.4 [0.9, 6.6] | 42.0 / 0 | 1.0 [0.3, 1.9] | 2.1 [0.4, 4.4] |
| ko_s0 | 0 % | base | 0.00 | transfer_other | conditional | 84 | 85.0 [79.7, 90.0] | 83.7 [78.2, 88.8] | -2.7 | 86.5 [81.4, 91.1] | 28.8 [22.3, 35.5] | 22.2 / 0 | 31.3 [24.7, 38.5] | 30.8 [23.8, 37.9] |
| ko_s0 | 0 % | base | 0.00 | transfer_other | cases | 80 | 20.1 [14.6, 25.7] | 19.0 [13.7, 24.3] | -1.7 | 86.8 [82.0, 90.9] | 30.3 [21.6, 39.3] | 27.9 / 0 | 15.8 [11.2, 20.8] | 33.4 [23.9, 42.6] |
| ko_s1 | 0 % | base | 0.00 | strict106 | reductio | 106 | 2.8e-05 [6.4e-06, 5.7e-05] | 0 [0, 0] | -0.0 | 0.9 [0.4, 1.5] | 0.9 [0.2, 1.7] | 70.1 / 0 | 3.4 [0, 14.3] | 0 [0, 0] |
| ko_s1 | 0 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| ko_s1 | 0 % | base | 0.00 | strict106 | conditional | 33 | 75.7 [64.1, 86.3] | 74.3 [62.2, 85.1] | -1.7 | 79.4 [68.0, 89.2] | 0 [0, 0] | 48.2 / 0 | 0.1 [0, 0.4] | 0 [0, 0] |
| ko_s1 | 0 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| ko_s1 | 0 % | base | 0.00 | transfer_other | reductio | 240 | 2.1e-04 [1.2e-05, 5.5e-04] | 0 [0, 0] | -0.1 | 3.8 [1.9, 5.9] | 1.8 [0.8, 2.9] | 63.6 / 0.035 | 0.3 [0, 1.2] | 0.029 [0, 0.1] |
| ko_s1 | 0 % | base | 0.00 | transfer_other | negation | 44 | 45.3 [35.9, 54.3] | 44.2 [34.7, 52.9] | -1.1 | 72.4 [63.8, 79.9] | 3.8 [0.4, 8.3] | 41.3 / 0 | 0.7 [0.094, 1.5] | 2.6 [0.033, 7.5] |
| ko_s1 | 0 % | base | 0.00 | transfer_other | conditional | 84 | 87.6 [82.2, 92.5] | 86.3 [81.1, 91.2] | -3.0 | 89.5 [84.8, 93.9] | 37.4 [29.3, 45.7] | 23.2 / 0 | 40.3 [32.4, 48.6] | 40.0 [31.8, 48.7] |
| ko_s1 | 0 % | base | 0.00 | transfer_other | cases | 80 | 14.3 [9.9, 18.6] | 15.1 [10.4, 20.2] | +1.5 | 76.6 [70.5, 82.6] | 27.8 [20.1, 36.0] | 29.2 / 0 | 14.1 [9.4, 18.8] | 33.1 [24.3, 42.2] |
| ko_s2 | 0 % | base | 0.00 | strict106 | reductio | 106 | 3.4e-06 [1.8e-07, 9.1e-06] | 0 [0, 0] | -0.0 | 0.2 [0.059, 0.3] | 2.1 [0.4, 4.6] | 47.8 / 0 | 0 [0, 0] | 0 [0, 0] |
| ko_s2 | 0 % | base | 0.00 | strict106 | negation | 0 applicable | | | | | | | | |
| ko_s2 | 0 % | base | 0.00 | strict106 | conditional | 33 | 71.7 [59.2, 82.7] | 72.0 [59.8, 82.8] | +0.3 | 74.7 [63.3, 84.8] | 0 [0, 0] | 19.9 / 0 | 0 [0, 0] | 0 [0, 0] |
| ko_s2 | 0 % | base | 0.00 | strict106 | cases | 0 applicable | | | | | | | | |
| ko_s2 | 0 % | base | 0.00 | transfer_other | reductio | 240 | 2.4e-05 [4.7e-07, 6.8e-05] | 0 [0, 0] | -0.0 | 2.8 [1.3, 4.6] | 1.3 [0.6, 2.3] | 46.9 / 0.035 | 1.4 [0, 4.0] | 0.2 [0, 0.7] |
| ko_s2 | 0 % | base | 0.00 | transfer_other | negation | 44 | 27.1 [18.4, 36.3] | 28.5 [19.4, 38.1] | +1.7 | 71.7 [62.6, 79.7] | 9.3 [3.2, 16.5] | 24.1 / 0 | 2.4 [0.3, 5.5] | 7.1 [1.3, 14.7] |
| ko_s2 | 0 % | base | 0.00 | transfer_other | conditional | 84 | 70.1 [62.1, 77.4] | 68.9 [60.9, 76.3] | -2.0 | 77.4 [70.3, 83.8] | 39.7 [31.2, 48.1] | 12.3 / 0 | 45.3 [36.5, 54.0] | 43.6 [34.5, 52.8] |
| ko_s2 | 0 % | base | 0.00 | transfer_other | cases | 80 | 8.4 [5.4, 11.8] | 7.9 [5.1, 11.3] | -1.0 | 84.9 [79.2, 90.3] | 29.7 [21.6, 38.5] | 28.7 / 0.1 | 13.3 [8.8, 18.5] | 31.8 [23.1, 41.2] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | strict106 | reductio | 106 | 3.1e-04 [3.6e-05, 7.1e-04] | 0 [0, 0] | -0.1 | 21.3 [15.7, 27.5] | 2.6 [0.6, 5.5] | 86.4 / 0.7 | 0 [0, 0] | 6.5 [1.6, 13.9] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | strict106 | negation | 0 applicable | | | | | | | | |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | strict106 | conditional | 33 | 93.4 [85.1, 100.0] | 92.7 [83.1, 100.0] | -1.8 | 94.8 [87.1, 100.0] | 0 [0, 0] | 97.2 / 0 | 0 [0, 0] | 0 [0, 0] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | strict106 | cases | 0 applicable | | | | | | | | |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | transfer_other | reductio | 240 | 8.4e-07 [3.2e-07, 1.5e-06] | 0 [0, 0] | -0.0 | 5.7 [3.3, 8.5] | 23.6 [19.4, 28.1] | 49.6 / 1.8 | 0.9 [0, 3.4] | 11.6 [1.2, 23.7] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | transfer_other | negation | 44 | 99.9 [99.9, 99.9] | 99.7 [99.4, 99.9] | -2.5 | 99.8 [99.5, 100.0] | 64.2 [52.6, 76.3] | 29.5 / 0 | 62.3 [50.8, 74.0] | 64.2 [52.6, 76.4] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | transfer_other | conditional | 84 | 99.9 [99.8, 100.0] | 99.6 [99.2, 99.9] | -5.0 | 99.6 [99.3, 99.9] | 89.2 [83.7, 94.1] | 4.5 / 0 | 88.6 [82.7, 93.9] | 89.2 [83.7, 94.2] |
| m_dd01_s0_it60 | 0.1 % | matched | 0.44 | transfer_other | cases | 80 | 49.4 [39.3, 59.7] | 49.0 [38.9, 59.3] | -1.3 | 93.1 [88.0, 97.0] | 71.8 [63.3, 79.7] | 18.3 / 0.2 | 71.8 [63.5, 80.1] | 71.5 [62.9, 79.8] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | strict106 | reductio | 106 | 5.8e-07 [6.7e-09, 1.5e-06] | 0 [0, 0] | -0.0 | 0.8 [0.4, 1.3] | 3.0 [0.8, 5.7] | 89.5 / 1.7 | 0 [0, 0] | 3.6 [0, 11.7] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | strict106 | negation | 0 applicable | | | | | | | | |
| m_dko_s0_it60 | 0 % | matched | 0.42 | strict106 | conditional | 33 | 99.9 [99.8, 100.0] | 99.7 [99.3, 100.0] | -2.6 | 99.7 [99.3, 100.0] | 0 [0, 0] | 95.7 / 0.3 | 0 [0, 0] | 0 [0, 0] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | strict106 | cases | 0 applicable | | | | | | | | |
| m_dko_s0_it60 | 0 % | matched | 0.42 | transfer_other | reductio | 240 | 1.0e-09 [4.2e-12, 2.6e-09] | 0 [0, 0] | -0.0 | 4.1 [1.9, 6.5] | 22.7 [18.3, 27.3] | 50.7 / 1.0 | 4.5 [0, 14.8] | 0 [0, 0] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | transfer_other | negation | 44 | 59.4 [45.1, 73.1] | 59.9 [45.4, 73.7] | +1.5 | 97.0 [91.9, 99.9] | 60.0 [48.3, 72.3] | 31.1 / 0.4 | 78.4 [67.9, 88.0] | 58.9 [47.0, 71.3] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | transfer_other | conditional | 84 | 97.4 [93.8, 99.9] | 97.2 [93.6, 99.7] | -3.2 | 97.2 [93.6, 99.7] | 86.5 [79.3, 92.8] | 10.2 / 0 | 87.8 [80.6, 93.7] | 87.5 [80.4, 93.5] |
| m_dko_s0_it60 | 0 % | matched | 0.42 | transfer_other | cases | 80 | 73.9 [64.8, 82.6] | 74.3 [65.2, 83.0] | +1.2 | 96.4 [92.8, 99.3] | 90.9 [85.5, 95.3] | 6.1 / 0 | 91.8 [86.7, 96.0] | 92.1 [86.9, 96.3] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | strict106 | reductio | 106 | 1.4e-08 [3.3e-11, 4.3e-08] | 0 [0, 0] | -0.0 | 2.4 [1.1, 4.0] | 2.6 [0.6, 5.1] | 79.9 / 0.6 | 0 [0, 0] | 0 [0, 0] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | strict106 | negation | 0 applicable | | | | | | | | |
| m_dko_s1_it70 | 0 % | matched | 0.48 | strict106 | conditional | 33 | 99.7 [99.2, 100.0] | 99.7 [99.3, 100.0] | -0.1 | 99.7 [99.3, 100.0] | 0 [0, 0] | 85.6 / 0 | 0 [0, 0] | 0 [0, 0] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | strict106 | cases | 0 applicable | | | | | | | | |
| m_dko_s1_it70 | 0 % | matched | 0.48 | transfer_other | reductio | 240 | 6.3e-09 [3.4e-12, 1.9e-08] | 0 [0, 0] | -0.0 | 4.2 [2.0, 6.8] | 14.4 [10.7, 18.4] | 44.6 / 0.7 | 0 [0, 0] | 1.7 [0, 5.5] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | transfer_other | negation | 44 | 67.2 [54.2, 79.6] | 67.1 [53.5, 79.7] | -0.1 | 99.9 [99.6, 100.0] | 67.4 [55.5, 78.8] | 21.6 / 0.6 | 71.1 [59.5, 81.7] | 67.4 [55.5, 78.7] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | transfer_other | conditional | 84 | 99.7 [99.2, 100.0] | 99.5 [98.7, 99.9] | -2.6 | 99.6 [99.1, 99.9] | 83.0 [75.6, 89.5] | 8.7 / 0 | 82.5 [75.1, 89.0] | 83.2 [75.8, 89.7] |
| m_dko_s1_it70 | 0 % | matched | 0.48 | transfer_other | cases | 80 | 54.7 [44.7, 65.3] | 54.3 [44.5, 64.9] | -1.1 | 96.4 [92.8, 99.1] | 83.0 [75.5, 90.1] | 9.9 / 0 | 87.4 [80.8, 93.2] | 83.3 [75.8, 90.2] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | strict106 | reductio | 106 | 3.2e-04 [2.4e-08, 9.5e-04] | 0 [0, 0] | -0.1 | 6.0 [3.1, 9.4] | 5.7 [2.0, 10.2] | 86.9 / 5.1 | 0 [0, 0] | 0.082 [0, 0.3] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | strict106 | negation | 0 applicable | | | | | | | | |
| m_dko_s2_it100 | 0 % | matched | 0.71 | strict106 | conditional | 33 | 99.7 [99.0, 100.0] | 99.5 [98.7, 100.0] | -0.7 | 99.5 [98.7, 100.0] | 0 [0, 0] | 92.7 / 0.8 | 0 [0, 0] | 0 [0, 0] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | strict106 | cases | 0 applicable | | | | | | | | |
| m_dko_s2_it100 | 0 % | matched | 0.71 | transfer_other | reductio | 240 | 3.6e-09 [9.3e-13, 1.1e-08] | 0 [0, 0] | -0.0 | 3.5 [1.5, 6.0] | 29.2 [23.9, 34.2] | 45.0 / 5.0 | 0 [0, 0] | 12.0 [0, 33.1] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | transfer_other | negation | 44 | 62.7 [49.6, 76.0] | 62.8 [49.7, 75.9] | +0.4 | 97.7 [93.2, 100.0] | 75.8 [65.0, 85.4] | 17.0 / 1.1 | 83.9 [74.6, 91.7] | 75.2 [64.2, 85.0] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | transfer_other | conditional | 84 | 98.4 [95.9, 99.9] | 98.5 [95.9, 99.9] | +0.3 | 98.5 [96.0, 100.0] | 89.3 [83.1, 94.9] | 6.7 / 0.4 | 90.5 [84.8, 95.8] | 90.2 [84.3, 95.6] |
| m_dko_s2_it100 | 0 % | matched | 0.71 | transfer_other | cases | 80 | 60.4 [50.6, 70.1] | 60.3 [50.5, 69.9] | -0.3 | 94.8 [89.8, 98.6] | 89.8 [83.4, 95.2] | 3.9 / 0.1 | 95.3 [91.2, 98.3] | 89.6 [83.2, 95.2] |

## Unforced success and how it is reached

Share using d: accepted attempts that use d anywhere, over all accepted attempts (an attempt can use several). First move: % of all attempts whose first proof action is d's opening (direct: opens no box).

| checkpoint | set | n | solved | Lean-rejected / refused / capped | share using reductio / negation / conditional / cases | first move reductio / negation / conditional / cases / direct |
|---|---|---|---|---|---|---|
| dd01_s0 | strict106 | 106 | 0 [0, 0] | 2.9 / 95.5 / 1.5 | n/a / n/a / n/a / n/a | 0 / n/a / 31.1 / n/a / 0 |
| dd01_s0 | transfer_other | 240 | 82.5 [77.9, 87.0] | 4.0 / 11.3 / 2.2 | 0.079 / 20.8 / 38.6 / 35.3 | 0 / 18.3 / 34.5 / 13.0 / 9.3 |
| dd01_s1 | strict106 | 106 | 0 [0, 0] | 1.6 / 77.7 / 20.7 | n/a / n/a / n/a / n/a | 0 / n/a / 30.1 / n/a / 8.0 |
| dd01_s1 | transfer_other | 240 | 84.4 [80.3, 88.2] | 4.5 / 6.5 / 4.6 | 0.2 / 21.2 / 36.5 / 34.7 | 0 / 18.3 / 32.6 / 17.1 / 12.4 |
| dd01_s2 | strict106 | 106 | 0 [0, 0] | 2.8 / 85.4 / 11.8 | n/a / n/a / n/a / n/a | 0 / n/a / 30.0 / n/a / 1.2 |
| dd01_s2 | transfer_other | 240 | 79.3 [74.5, 83.6] | 3.5 / 13.6 / 3.5 | 0 / 21.1 / 39.2 / 35.4 | 0 / 11.7 / 34.7 / 18.2 / 8.5 |
| dd1_s0 | strict106 | 106 | 21.8 [14.6, 29.6] | 6.1 / 71.8 / 0.4 | 100.0 / n/a / 53.9 / n/a | 0.6 / n/a / 31.1 / n/a / 0.5 |
| dd1_s0 | transfer_other | 240 | 72.3 [67.3, 77.2] | 5.8 / 21.7 / 0.2 | 3.0 / 15.3 / 43.4 / 36.5 | 0.7 / 18.2 / 34.5 / 12.6 / 9.6 |
| dd1_s1 | strict106 | 106 | 15.7 [9.3, 22.5] | 12.0 / 72.3 / 0 | 100.0 / n/a / 99.8 / n/a | 0.6 / n/a / 29.8 / n/a / 0.9 |
| dd1_s1 | transfer_other | 240 | 71.9 [67.0, 76.8] | 7.1 / 20.9 / 0.1 | 0.9 / 17.3 / 41.4 / 37.5 | 0.2 / 18.3 / 34.8 / 16.9 / 9.1 |
| dd1_s2 | strict106 | 106 | 16.7 [9.9, 23.3] | 7.3 / 74.7 / 1.3 | 100.0 / n/a / 100.0 / n/a | 0 / n/a / 30.5 / n/a / 3.2 |
| dd1_s2 | transfer_other | 240 | 73.8 [68.6, 78.8] | 7.6 / 16.9 / 1.7 | 2.2 / 19.5 / 40.7 / 38.7 | 0 / 18.0 / 34.5 / 18.5 / 9.8 |
| dko_s0 | strict106 | 106 | 0 [0, 0] | 4.2 / 83.3 / 12.4 | n/a / n/a / n/a / n/a | 0 / n/a / 31.1 / n/a / 0 |
| dko_s0 | transfer_other | 240 | 78.0 [72.9, 82.7] | 4.8 / 10.4 / 6.8 | 0 / 21.8 / 37.1 / 36.8 | 0 / 11.5 / 34.2 / 20.7 / 9.5 |
| dko_s1 | strict106 | 106 | 0 [0, 0] | 0.2 / 80.1 / 19.6 | n/a / n/a / n/a / n/a | 0 / n/a / 31.1 / n/a / 6.6 |
| dko_s1 | transfer_other | 240 | 79.2 [74.3, 83.9] | 4.1 / 11.3 / 5.4 | 0 / 20.3 / 38.7 / 37.0 | 0 / 11.2 / 34.6 / 19.6 / 10.5 |
| dko_s2 | strict106 | 106 | 0 [0, 0] | 0.4 / 93.3 / 6.3 | n/a / n/a / n/a / n/a | 0 / n/a / 30.8 / n/a / 0 |
| dko_s2 | transfer_other | 240 | 76.6 [71.6, 81.2] | 3.8 / 16.5 / 3.1 | 0 / 21.2 / 39.0 / 38.5 | 0 / 11.2 / 32.9 / 19.6 / 10.5 |
| dose01_s0 | strict106 | 106 | 0 [0, 0] | 60.6 / 39.4 / 0 | n/a / n/a / n/a / n/a | 0 / n/a / 20.7 / n/a / 8.0 |
| dose01_s0 | transfer_other | 240 | 15.7 [12.4, 19.2] | 60.4 / 23.9 / 0 | 0.083 / 1.3 / 73.0 / 20.0 | 0 / 12.2 / 26.7 / 4.7 / 16.8 |
| dose01_s1 | strict106 | 106 | 0 [0, 0] | 41.0 / 59.0 / 0 | n/a / n/a / n/a / n/a | 0 / n/a / 24.5 / n/a / 6.1 |
| dose01_s1 | transfer_other | 240 | 19.2 [15.4, 23.3] | 45.6 / 35.2 / 0 | 0.6 / 3.7 / 71.4 / 15.8 | 0 / 11.1 / 30.7 / 12.4 / 10.3 |
| dose01_s2 | strict106 | 106 | 0.029 [0, 0.088] | 39.1 / 60.8 / 0 | 100.0 / n/a / 100.0 / n/a | 0 / n/a / 21.3 / n/a / 6.5 |
| dose01_s2 | transfer_other | 240 | 14.0 [11.0, 17.1] | 33.4 / 52.6 / 0 | 0.093 / 0.093 / 75.7 / 21.8 | 0 / 3.9 / 25.9 / 4.6 / 10.7 |
| dose1_s0 | strict106 | 106 | 0.059 [0, 0.1] | 51.7 / 48.3 / 0 | 100.0 / n/a / 100.0 / n/a | 0.029 / n/a / 22.6 / n/a / 4.0 |
| dose1_s0 | transfer_other | 240 | 18.4 [14.9, 22.1] | 45.6 / 36.0 / 0.013 | 1.2 / 1.3 / 66.2 / 24.7 | 0 / 9.3 / 28.0 / 5.3 / 12.2 |
| dose1_s1 | strict106 | 106 | 2.0 [0.9, 3.1] | 40.6 / 57.4 / 0 | 100.0 / n/a / 100.0 / n/a | 0.059 / n/a / 19.2 / n/a / 7.4 |
| dose1_s1 | transfer_other | 240 | 17.9 [14.3, 21.6] | 51.5 / 30.5 / 0 | 1.3 / 0.2 / 68.3 / 22.1 | 0.078 / 12.2 / 27.0 / 5.9 / 15.0 |
| dose1_s2 | strict106 | 106 | 0.059 [0, 0.1] | 40.5 / 59.4 / 0.029 | 100.0 / n/a / 100.0 / n/a | 0 / n/a / 24.4 / n/a / 2.6 |
| dose1_s2 | transfer_other | 240 | 17.4 [14.1, 21.0] | 55.4 / 27.1 / 0 | 0.4 / 0.7 / 61.0 / 27.8 | 0.052 / 16.1 / 29.2 / 8.6 / 8.2 |
| full_s3 | strict106 | 106 | 4.2 [1.8, 7.1] | 43.0 / 52.8 / 0 | 100.0 / n/a / 97.9 / n/a | 0.7 / n/a / 24.5 / n/a / 3.9 |
| full_s3 | transfer_other | 240 | 18.2 [14.6, 22.0] | 52.7 / 29.2 / 0 | 4.4 / 0.6 / 72.9 / 21.2 | 0.4 / 11.5 / 29.8 / 8.0 / 12.3 |
| ko_s0 | strict106 | 106 | 0 [0, 0] | 55.6 / 44.4 / 0.029 | n/a / n/a / n/a / n/a | 0 / n/a / 27.8 / n/a / 5.5 |
| ko_s0 | transfer_other | 240 | 14.8 [11.9, 17.8] | 52.4 / 32.8 / 0 | 0 / 1.1 / 64.3 / 31.1 | 0 / 13.0 / 29.3 / 6.3 / 15.5 |
| ko_s1 | strict106 | 106 | 0.029 [0, 0.088] | 39.7 / 60.3 / 0 | 100.0 / n/a / 100.0 / n/a | 0 / n/a / 23.1 / n/a / 5.9 |
| ko_s1 | transfer_other | 240 | 19.7 [15.9, 23.5] | 42.1 / 38.2 / 0.013 | 0.066 / 0.5 / 64.2 / 18.3 | 0 / 8.1 / 30.2 / 5.0 / 13.1 |
| ko_s2 | strict106 | 106 | 0 [0, 0] | 61.4 / 38.6 / 0 | n/a / n/a / n/a / n/a | 0 / n/a / 22.4 / n/a / 2.4 |
| ko_s2 | transfer_other | 240 | 17.3 [13.9, 20.7] | 54.1 / 28.7 / 0 | 0.2 / 1.8 / 71.2 / 21.9 | 0 / 5.2 / 24.1 / 2.6 / 17.1 |
| m_dd01_s0_it60 | strict106 | 106 | 0 [0, 0] | 4.0 / 96.0 / 0.029 | n/a / n/a / n/a / n/a | 0 / n/a / 28.9 / n/a / 0 |
| m_dd01_s0_it60 | transfer_other | 240 | 67.0 [61.7, 72.4] | 7.6 / 25.3 / 0.065 | 0.078 / 17.0 / 46.1 / 33.2 | 0 / 18.3 / 34.9 / 16.3 / 7.4 |
| m_dko_s0_it60 | strict106 | 106 | 0 [0, 0] | 5.5 / 94.3 / 0.2 | n/a / n/a / n/a / n/a | 0 / n/a / 31.0 / n/a / 0.029 |
| m_dko_s0_it60 | transfer_other | 240 | 73.8 [68.5, 78.7] | 5.7 / 20.3 / 0.2 | 0.2 / 18.9 / 40.5 / 40.0 | 0 / 11.0 / 34.0 / 24.8 / 9.7 |
| m_dko_s1_it70 | strict106 | 106 | 0 [0, 0] | 15.2 / 84.7 / 0.059 | n/a / n/a / n/a / n/a | 0 / n/a / 31.0 / n/a / 1.9 |
| m_dko_s1_it70 | transfer_other | 240 | 71.0 [65.8, 76.3] | 8.4 / 20.6 / 0.013 | 0 / 18.3 / 40.5 / 39.5 | 0 / 12.3 / 34.8 / 18.1 / 6.2 |
| m_dko_s2_it100 | strict106 | 106 | 0 [0, 0] | 3.7 / 95.3 / 0.9 | n/a / n/a / n/a / n/a | 0 / n/a / 31.0 / n/a / 0.029 |
| m_dko_s2_it100 | transfer_other | 240 | 77.8 [72.9, 82.4] | 3.8 / 17.6 / 0.8 | 0 / 19.3 / 40.1 / 38.7 | 0 / 11.5 / 34.5 / 20.1 / 9.2 |

## Secondary: the same rule with propensity = uses d anywhere (U)

| dose | seed | set | direction | P base → end | P ratio | P | C | reading | primary reading |
|---|---|---|---|---|---|---|---|---|---|
| 0 % | 0 | strict106 | reductio | 0.059 → 2.5 | x34.2 [10.6, 231] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 0 | strict106 | conditional | 91.0 → 100.0 | x1.1 [1.04, 1.19] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 0 | transfer_other | reductio | 3.0 → 4.4 | x1.48 [0.888, 2.5] | unresolved | up | unresolved | candidate capability gain |
| 0 % | 0 | transfer_other | negation | 85.5 → 99.4 | x1.16 [1.09, 1.27] | flat | up | candidate capability gain | candidate capability gain |
| 0 % | 0 | transfer_other | conditional | 86.5 → 97.6 | x1.13 [1.06, 1.2] | flat | up | candidate capability gain | candidate capability gain |
| 0 % | 0 | transfer_other | cases | 86.8 → 95.7 | x1.1 [1.05, 1.16] | flat | up | candidate capability gain | mix |
| 0 % | 1 | strict106 | reductio | 0.9 → 4.6 | x5.27 [2.09, 13.8] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 1 | strict106 | conditional | 79.4 → 100.0 | x1.26 [1.12, 1.47] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 1 | transfer_other | reductio | 3.8 → 3.6 | x0.959 [0.686, 1.16] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 1 | transfer_other | negation | 72.4 → 97.7 | x1.35 [1.21, 1.53] | flat | up | candidate capability gain | candidate capability gain |
| 0 % | 1 | transfer_other | conditional | 89.5 → 98.8 | x1.1 [1.05, 1.17] | flat | up | candidate capability gain | candidate capability gain |
| 0 % | 1 | transfer_other | cases | 76.6 → 94.8 | x1.24 [1.14, 1.35] | flat | up | candidate capability gain | mix |
| 0 % | 2 | strict106 | reductio | 0.2 → 0.5 | x2.69 [1, 9.01] | rare | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 2 | strict106 | conditional | 74.7 → 98.9 | x1.32 [1.17, 1.56] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 2 | transfer_other | reductio | 2.8 → 3.3 | x1.2 [0.826, 1.57] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0 % | 2 | transfer_other | negation | 71.7 → 97.7 | x1.36 [1.23, 1.53] | flat | up | candidate capability gain | mix |
| 0 % | 2 | transfer_other | conditional | 77.4 → 97.6 | x1.26 [1.16, 1.38] | flat | up | candidate capability gain | candidate capability gain |
| 0 % | 2 | transfer_other | cases | 84.9 → 93.1 | x1.1 [1.01, 1.2] | flat | up | candidate capability gain | mix |
| 0.1 % | 0 | strict106 | reductio | 0.1 → 4.6 | x28.5 [12.1, 102] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 0 | strict106 | conditional | 74.1 → 100.0 | x1.35 [1.18, 1.61] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 0 | transfer_other | reductio | 3.9 → 4.7 | x1.21 [0.813, 1.73] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 0 | transfer_other | negation | 84.9 → 100.0 | x1.18 [1.09, 1.3] | flat | up | candidate capability gain | mix |
| 0.1 % | 0 | transfer_other | conditional | 82.6 → 98.6 | x1.19 [1.13, 1.28] | flat | up | candidate capability gain | candidate capability gain |
| 0.1 % | 0 | transfer_other | cases | 86.3 → 94.6 | x1.1 [1.04, 1.17] | flat | up | candidate capability gain | mix |
| 0.1 % | 1 | strict106 | reductio | 1.8 → 0.9 | x0.472 [0.235, 0.864] | down | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 1 | strict106 | conditional | 79.6 → 99.7 | x1.25 [1.1, 1.5] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 1 | transfer_other | reductio | 3.8 → 3.9 | x1.05 [0.653, 1.43] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 1 | transfer_other | negation | 66.2 → 100.0 | x1.51 [1.3, 1.8] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 1 | transfer_other | conditional | 89.4 → 99.2 | x1.11 [1.05, 1.18] | flat | up | candidate capability gain | candidate capability gain |
| 0.1 % | 1 | transfer_other | cases | 80.0 → 92.7 | x1.16 [1.05, 1.28] | flat | up | candidate capability gain | candidate capability gain |
| 0.1 % | 2 | strict106 | reductio | 2.5 → 10.4 | x4.18 [2.1, 8.29] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 2 | strict106 | conditional | 74.5 → 96.5 | x1.29 [1.17, 1.49] | flat | rare | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 2 | transfer_other | reductio | 3.5 → 4.5 | x1.29 [0.842, 1.85] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 2 | transfer_other | negation | 36.3 → 100.0 | x2.75 [2.18, 3.56] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 0.1 % | 2 | transfer_other | conditional | 79.8 → 99.6 | x1.25 [1.15, 1.36] | flat | up | candidate capability gain | candidate capability gain |
| 0.1 % | 2 | transfer_other | cases | 64.4 → 94.1 | x1.46 [1.31, 1.66] | up | up | mix | mix |
| 1 % | 0 | strict106 | reductio | 1.2 → 34.2 | x28.7 [13, 96.2] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 1 % | 0 | strict106 | conditional | 77.7 → 99.8 | x1.29 [1.14, 1.5] | flat | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 1 % | 0 | transfer_other | reductio | 3.9 → 8.2 | x2.12 [1.37, 3.33] | up | up | mix | candidate capability gain |
| 1 % | 0 | transfer_other | negation | 66.8 → 99.4 | x1.49 [1.28, 1.81] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 1 % | 0 | transfer_other | conditional | 86.8 → 98.5 | x1.13 [1.09, 1.19] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 0 | transfer_other | cases | 84.2 → 95.2 | x1.13 [1.06, 1.22] | flat | up | candidate capability gain | mix |
| 1 % | 1 | strict106 | reductio | 9.8 → 30.3 | x3.07 [2.35, 4.01] | up | up | mix | candidate capability gain |
| 1 % | 1 | strict106 | conditional | 65.0 → 95.7 | x1.47 [1.24, 1.84] | up | up | mix | mix |
| 1 % | 1 | transfer_other | reductio | 6.5 → 6.4 | x0.986 [0.639, 1.37] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 1 | transfer_other | negation | 81.4 → 99.7 | x1.23 [1.11, 1.4] | flat | up | candidate capability gain | mix |
| 1 % | 1 | transfer_other | conditional | 82.0 → 99.4 | x1.21 [1.13, 1.31] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 1 | transfer_other | cases | 74.6 → 95.2 | x1.28 [1.17, 1.41] | flat | up | candidate capability gain | mix |
| 1 % | 2 | strict106 | reductio | 1.3 → 49.0 | x37.4 [25.7, 61.8] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 1 % | 2 | strict106 | conditional | 84.3 → 99.0 | x1.17 [1.07, 1.32] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 2 | transfer_other | reductio | 5.5 → 12.2 | x2.23 [1.65, 3.15] | up | up | no reading: forced attempts mostly refused or capped | no reading: forced attempts mostly refused or capped |
| 1 % | 2 | transfer_other | negation | 93.1 → 99.5 | x1.07 [1.01, 1.15] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 2 | transfer_other | conditional | 87.1 → 99.6 | x1.14 [1.09, 1.21] | flat | up | candidate capability gain | candidate capability gain |
| 1 % | 2 | transfer_other | cases | 79.6 → 97.6 | x1.23 [1.14, 1.33] | flat | up | candidate capability gain | mix |

## Secondary: capability on the familiar subset (theorems the base itself opens with d at ≥ 1 %)

| dose | seed | set | direction | n | C base → end | C ratio | C state | C state, all theorems |
|---|---|---|---|---|---|---|---|---|
| 0 % | 0 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 0 | strict106 | conditional | 33 | 0 → 0 | x1 [1, 1] | rare | rare |
| 0 % | 0 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 0 | transfer_other | negation | 43 | 3.5 → 73.3 | x20.5 [10.9, 64.8] | up | up |
| 0 % | 0 | transfer_other | conditional | 84 | 28.8 → 84.0 | x2.92 [2.36, 3.73] | up | up |
| 0 % | 0 | transfer_other | cases | 66 | 33.6 → 92.8 | x2.76 [2.14, 3.84] | up | up |
| 0 % | 1 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 1 | strict106 | conditional | 31 | 0 → 0 | x1 [1, 1] | rare | rare |
| 0 % | 1 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 1 | transfer_other | negation | 38 | 0.4 → 73.0 | x133 [59.2, 705] | up | up |
| 0 % | 1 | transfer_other | conditional | 83 | 37.9 → 86.5 | x2.28 [1.89, 2.84] | up | up |
| 0 % | 1 | transfer_other | cases | 60 | 33.2 → 81.8 | x2.46 [1.92, 3.37] | up | up |
| 0 % | 2 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 2 | strict106 | conditional | 31 | 0 → 0 | x1 [1, 1] | rare | rare |
| 0 % | 2 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0 % | 2 | transfer_other | negation | 36 | 2.1 → 77.5 | x35.3 [16.7, 217] | up | up |
| 0 % | 2 | transfer_other | conditional | 78 | 40.0 → 88.4 | x2.21 [1.81, 2.84] | up | up |
| 0 % | 2 | transfer_other | cases | 45 | 43.0 → 91.1 | x2.12 [1.68, 2.9] | up | up |
| 0.1 % | 0 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 0 | strict106 | conditional | 30 | 0 → 0 | x1 [1, 1] | rare | rare |
| 0.1 % | 0 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 0 | transfer_other | negation | 43 | 3.1 → 93.0 | x29.1 [15, 84.8] | up | up |
| 0.1 % | 0 | transfer_other | conditional | 84 | 35.6 → 92.2 | x2.59 [2.12, 3.33] | up | up |
| 0.1 % | 0 | transfer_other | cases | 55 | 33.5 → 91.4 | x2.72 [2.04, 3.82] | up | up |
| 0.1 % | 1 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 1 | strict106 | conditional | 28 | 0 → 0 | x1 [1, 1] | rare | rare |
| 0.1 % | 1 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 1 | transfer_other | negation | 42 | 5.8 → 97.4 | x16.7 [9.22, 47.9] | up | up |
| 0.1 % | 1 | transfer_other | conditional | 80 | 42.1 → 89.7 | x2.13 [1.75, 2.71] | up | up |
| 0.1 % | 1 | transfer_other | cases | 54 | 21.6 → 87.8 | x4.05 [2.89, 6.15] | up | up |
| 0.1 % | 2 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 2 | strict106 | conditional | 29 | 0.6 → 0 | x0.2 [0.0909, 1] | rare | rare |
| 0.1 % | 2 | transfer_other | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 0.1 % | 2 | transfer_other | negation | 31 | 1.6 → 85.8 | x49.2 [19.4, 623] | up | up |
| 0.1 % | 2 | transfer_other | conditional | 76 | 36.5 → 90.7 | x2.48 [2.06, 3.1] | up | up |
| 0.1 % | 2 | transfer_other | cases | 64 | 30.9 → 85.3 | x2.76 [2.11, 3.96] | up | up |
| 1 % | 0 | strict106 | reductio | 3 | | | no reading: n = 3 < 20 | up |
| 1 % | 0 | strict106 | conditional | 31 | 0 → 39.2 | x293 [171, 421] | up | up |
| 1 % | 0 | transfer_other | reductio | 2 | | | no reading: n = 2 < 20 | up |
| 1 % | 0 | transfer_other | negation | 39 | 1.7 → 59.4 | x32.8 [11.3, 601] | up | up |
| 1 % | 0 | transfer_other | conditional | 80 | 39.6 → 91.5 | x2.31 [1.95, 2.84] | up | up |
| 1 % | 0 | transfer_other | cases | 48 | 38.5 → 83.3 | x2.16 [1.68, 3.04] | up | up |
| 1 % | 1 | strict106 | reductio | 3 | | | no reading: n = 3 < 20 | up |
| 1 % | 1 | strict106 | conditional | 26 | 10.3 → 64.1 | x6.17 [4.22, 11] | up | up |
| 1 % | 1 | transfer_other | reductio | 5 | | | no reading: n = 5 < 20 | up |
| 1 % | 1 | transfer_other | negation | 39 | 0.2 → 65.4 | x204 [79, 681] | up | up |
| 1 % | 1 | transfer_other | conditional | 78 | 42.8 → 86.4 | x2.02 [1.7, 2.48] | up | up |
| 1 % | 1 | transfer_other | cases | 54 | 39.4 → 92.0 | x2.33 [1.87, 3.13] | up | up |
| 1 % | 2 | strict106 | reductio | 0 | | | no reading: n = 0 < 20 | up |
| 1 % | 2 | strict106 | conditional | 31 | 0.3 → 57.0 | x142 [52.7, 525] | up | up |
| 1 % | 2 | transfer_other | reductio | 5 | | | no reading: n = 5 < 20 | up |
| 1 % | 2 | transfer_other | negation | 43 | 4.5 → 77.3 | x17 [8.21, 143] | up | up |
| 1 % | 2 | transfer_other | conditional | 81 | 34.4 → 84.9 | x2.47 [2.01, 3.12] | up | up |
| 1 % | 2 | transfer_other | cases | 60 | 27.9 → 84.0 | x3 [2.27, 4.45] | up | up |

## Secondary: router or default shift (exact first-move propensity by class)

Classes present in the result files: helpful, idle, required. Change in log10 odds of P̄ per class; f = the not-needed class's change over the required class's. Router if the interval of Δ(not needed) − 0.5 Δ(required) lies below 0 or the not-needed class is flat, rare or down; default shift if the not-needed class is up and that interval lies above 0. Dan's RL targets are 2.7 % covered by theorems that require the reductio: little direct pressure where it is optional.

| dose | direction | per seed | dose verdict | f per seed |
|---|---|---|---|---|
| 0 % | reductio | no rise to locate / no rise to locate / no rise to locate | **no rise to locate** | n/a |
| 0 % | negation | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 0 % | conditional | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 0 % | cases | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 0.1 % | reductio | no rise to locate / no rise to locate / no rise to locate | **no rise to locate** | n/a |
| 0.1 % | negation | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 0.1 % | conditional | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 0.1 % | cases | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 1 % | reductio | no rise to locate / no rise to locate / no rise to locate | **no rise to locate** | n/a |
| 1 % | negation | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 1 % | conditional | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |
| 1 % | cases | no reading: required n < 20 / no reading: required n < 20 / no reading: required n < 20 | **no reading: required n < 20** | n/a |

| dose | seed | direction | mode | class | n | P base → end | Δlog10 odds [95 % CI] | state | uses anywhere, base → end | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 % | 0 | reductio | classes | required | 136 | 2.7e-05 → 1.6e-08 | -3.21 [-5.63, -2.46] | rare | 2.0 → 3.9 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 1.8e-06 → 2.7e-13 | -4.25 [-4.63, -3.54] | rare | 1.3 → 2.0 | |
| 0 % | 0 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 71.1 → 62.7 | -0.17 [-0.52, +0.23] | unresolved | 85.5 → 99.4 | |
| 0 % | 0 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 86.3 → 98.3 | +0.95 [+0.54, +3.73] | up | 87.8 → 98.3 | |
| 0 % | 0 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 20.1 → 62.0 | +0.81 [+0.59, +1.07] | up | 86.8 → 95.7 | |
| 0 % | 1 | reductio | classes | required | 136 | 2.2e-05 → 1.4e-10 | -5.18 [-5.58, -4.50] | rare | 2.7 → 5.8 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 1.8e-04 → 8.7e-11 | -6.26 [-6.71, -4.91] | rare | 1.8 → 1.4 | |
| 0 % | 1 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 45.3 → 61.4 | +0.28 [-0.06, +0.65] | unresolved | 72.4 → 97.7 | |
| 0 % | 1 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 84.3 → 99.1 | +1.33 [+0.79, +4.82] | up | 86.7 → 99.1 | |
| 0 % | 1 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 14.3 → 60.0 | +0.96 [+0.70, +1.23] | up | 76.6 → 94.8 | |
| 0 % | 2 | reductio | classes | required | 136 | 2.6e-06 → 1.5e-06 | -0.25 [-3.82, -0.18] | rare | 2.2 → 2.6 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 2.9e-05 → 4.5e-10 | -4.81 [-5.92, -3.03] | rare | 1.0 → 1.0 | |
| 0 % | 2 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 27.1 → 61.4 | +0.63 [+0.33, +0.98] | up | 71.7 → 97.7 | |
| 0 % | 2 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 70.5 → 95.4 | +0.94 [+0.66, +1.47] | up | 76.7 → 97.9 | |
| 0 % | 2 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 8.4 → 59.3 | +1.20 [+0.99, +1.45] | up | 84.9 → 93.1 | |
| 0.1 % | 0 | reductio | classes | required | 136 | 5.5e-05 → 1.9e-06 | -1.46 [-2.19, -0.89] | rare | 2.3 → 5.5 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 1.9e-04 → 1.3e-08 | -4.17 [-6.04, -3.62] | rare | 1.6 → 2.2 | |
| 0.1 % | 0 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 67.7 → 100.0 | +4.01 [+3.68, +4.44] | up | 84.9 → 100.0 | |
| 0.1 % | 0 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 74.9 → 98.8 | +1.45 [+1.06, +3.36] | up | 80.2 → 99.0 | |
| 0.1 % | 0 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 14.9 → 39.0 | +0.56 [+0.37, +0.74] | up | 86.3 → 94.6 | |
| 0.1 % | 1 | reductio | classes | required | 136 | 0.0013 → 1.4e-07 | -3.95 [-4.49, -3.15] | rare | 3.9 → 3.7 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 0.0092 → 2.9e-08 | -5.50 [-7.60, -4.86] | rare | 2.0 → 1.2 | |
| 0.1 % | 1 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 60.6 → 100.0 | +3.34 [+3.04, +3.74] | up | 66.2 → 100.0 | |
| 0.1 % | 1 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 86.5 → 94.6 | +0.44 [+0.09, +0.95] | up | 86.7 → 99.3 | |
| 0.1 % | 1 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 36.9 → 51.2 | +0.25 [+0.07, +0.45] | up | 80.0 → 92.7 | |
| 0.1 % | 2 | reductio | classes | required | 136 | 0.0024 → 3.9e-06 | -2.79 [-3.26, -2.30] | rare | 4.1 → 10.3 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 2.3e-04 → 1.0e-07 | -3.35 [-4.66, -2.34] | rare | 1.6 → 2.3 | |
| 0.1 % | 2 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 21.4 → 63.6 | +0.81 [+0.56, +1.08] | up | 36.3 → 100.0 | |
| 0.1 % | 2 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 72.2 → 98.5 | +1.40 [+1.07, +2.04] | up | 78.3 → 98.7 | |
| 0.1 % | 2 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 14.5 → 54.8 | +0.85 [+0.62, +1.10] | up | 64.4 → 94.1 | |
| 1 % | 0 | reductio | classes | required | 136 | 0.034 → 1.0 | +1.46 [+0.94, +2.16] | rare | 1.5 → 30.4 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 0.028 → 0.072 | +0.41 [-0.99, +0.99] | rare | 3.4 → 3.7 | |
| 1 % | 0 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 52.8 → 99.6 | +2.35 [+1.99, +2.96] | up | 66.8 → 99.4 | |
| 1 % | 0 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 78.8 → 99.0 | +1.41 [+0.99, +3.13] | up | 84.2 → 98.9 | |
| 1 % | 0 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 16.2 → 37.8 | +0.50 [+0.28, +0.73] | up | 84.2 → 95.2 | |
| 1 % | 1 | reductio | classes | required | 136 | 0.072 → 0.5 | +0.82 [-0.15, +1.37] | rare | 9.8 → 26.9 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 0.2 → 0.014 | -1.07 [-2.88, -0.27] | rare | 4.3 → 2.8 | |
| 1 % | 1 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 67.1 → 99.9 | +2.66 [+2.27, +3.13] | up | 81.4 → 99.7 | |
| 1 % | 1 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 73.3 → 98.6 | +1.42 [+1.08, +2.18] | up | 77.2 → 98.3 | |
| 1 % | 1 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 17.8 → 51.1 | +0.68 [+0.52, +0.87] | up | 74.6 → 95.2 | |
| 1 % | 2 | reductio | classes | required | 136 | 0.0029 → 0.002 | -0.17 [-1.59, +0.49] | rare | 3.5 → 43.0 | no rise to locate |
| | | | | helpful | 8 | | | | | |
| | | | | idle | 194 | 0.061 → 0.0018 | -1.52 [-3.44, -0.94] | rare | 3.6 → 8.0 | |
| 1 % | 2 | negation | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 44 | 88.5 → 98.3 | +0.88 [+0.33, +2.93] | up | 93.1 → 99.5 | |
| 1 % | 2 | conditional | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 117 | 83.0 → 98.9 | +1.25 [+0.86, +2.06] | up | 86.3 → 99.4 | |
| 1 % | 2 | cases | fallback | required | 0 | | | | | no reading: required n < 20 |
| | | | | optional | 80 | 26.9 → 55.0 | +0.52 [+0.37, +0.69] | up | 79.6 → 97.6 | |

## Matched-compute and secondary ends (never in a dose-level reading)

| checkpoint | kind | dose | seed | x | set | direction | P ratio | P | C ratio | C | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | strict106 | reductio | x0.0177 [10^-3.4, 0.0328] | rare | x1.05 [0.274, 3.8] | unresolved | no reading: forced attempts mostly refused or capped |
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | strict106 | conditional | x1.12 [1.05, 1.22] | flat | x1 [1, 1] | rare | no reading: forced attempts mostly refused or capped |
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | transfer_other | reductio | x10^-3.3 [10^-4.3, 10^-2.7] | rare | x19.5 [11.3, 46.4] | up | no reading: forced attempts mostly refused or capped |
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | transfer_other | negation | x0.836 [0.607, 1.12] | flat | x17.2 [9, 54.1] | up | candidate capability gain |
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | transfer_other | conditional | x1.15 [1.08, 1.22] | flat | x3 [2.46, 3.77] | up | candidate capability gain |
| m_dko_s0_it60 | matched compute | 0 % | 0 | 0.4170962209377143 | transfer_other | cases | x3.68 [2.88, 4.93] | up | x3 [2.33, 4.1] | up | mix |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | strict106 | reductio | x10^-3.3 [10^-5.5, 10^-2.9] | rare | x2.91 [0.806, 10.3] | unresolved | no reading: forced attempts mostly refused or capped |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | strict106 | conditional | x1.32 [1.15, 1.56] | flat | x1 [1, 1] | rare | no reading: forced attempts mostly refused or capped |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | transfer_other | reductio | x10^-4.5 [10^-6.6, 10^-3.1] | rare | x8.05 [5, 17.3] | up | no reading: forced attempts mostly refused or capped |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | transfer_other | negation | x1.48 [1.08, 2.06] | up | x17.4 [8.21, 135] | up | mix |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | transfer_other | conditional | x1.14 [1.08, 1.21] | flat | x2.22 [1.87, 2.78] | up | candidate capability gain |
| m_dko_s1_it70 | matched compute | 0 % | 1 | 0.47931220737254 | transfer_other | cases | x3.84 [2.7, 5.63] | up | x2.98 [2.33, 4.03] | up | mix |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | strict106 | reductio | x94.6 [0.0373, 112] | rare | x2.64 [1.44, 9.55] | up | no reading: forced attempts mostly refused or capped |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | strict106 | conditional | x1.39 [1.2, 1.66] | flat | x1 [1, 1] | rare | no reading: forced attempts mostly refused or capped |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | transfer_other | reductio | x10^-3.8 [10^-5.8, 10^-2.1] | rare | x21.9 [12.8, 47.8] | up | candidate capability gain |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | transfer_other | negation | x2.32 [1.62, 3.46] | up | x8.09 [4.58, 22.4] | up | mix |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | transfer_other | conditional | x1.4 [1.27, 1.57] | flat | x2.25 [1.86, 2.87] | up | candidate capability gain |
| m_dko_s2_it100 | matched compute | 0 % | 2 | 0.7119433823993557 | transfer_other | cases | x7.19 [5.26, 10.6] | up | x3.02 [2.34, 4.11] | up | mix |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | strict106 | reductio | x4.43 [0.551, 15.3] | rare | x2.91 [0.793, 9.57] | unresolved | no reading: forced attempts mostly refused or capped |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | strict106 | conditional | x1.36 [1.21, 1.6] | flat | x1 [1, 1] | rare | no reading: forced attempts mostly refused or capped |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | transfer_other | reductio | x10^-2.3 [10^-2.7, 0.013] | rare | x19.7 [11.3, 46.5] | up | no reading: forced attempts mostly refused or capped |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | transfer_other | negation | x1.47 [1.29, 1.72] | up | x12.8 [6.79, 38.1] | up | mix |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | transfer_other | conditional | x1.29 [1.2, 1.4] | flat | x2.5 [2.05, 3.21] | up | candidate capability gain |
| m_dd01_s0_it60 | matched compute | 0.1 % | 0 | 0.4402610729519408 | transfer_other | cases | x3.31 [2.7, 4.1] | up | x2.48 [1.99, 3.36] | up | mix |

## Exploratory: reductio on the strict 106 by schema (levels, no readings)

| line | schema | n | P base → end | C base → end |
|---|---|---|---|---|
| 0.1|0 | contraposition_conv | 18 | 1.1e-05 → 7.3e-13 | 0 → 0 |
| 0.1|0 | demorgan_nand_to_or | 4 | 1.8e-06 → 1.2e-17 | 0 → 0 |
| 0.1|0 | excluded_middle | 37 | 8.5e-08 → 3.8e-17 | 0.2 → 2.0 |
| 0.1|0 | negated_conditional | 15 | 1.4e-06 → 2.1e-16 | 0 → 0 |
| 0.1|0 | none | 1 | 5.6e-04 → 2.0e-09 | 0 → 0 |
| 0.1|0 | peirce | 12 | 4.4e-08 → 2.6e-15 | 0 → 0 |
| 0.1|0 | peirce_sequent | 19 | 3.5e-04 → 1.4e-05 | 4.4 → 21.9 |
| 0.1|1 | contraposition_conv | 18 | 1.8e-05 → 4.2e-10 | 0 → 0 |
| 0.1|1 | demorgan_nand_to_or | 4 | 0.0048 → 3.4e-17 | 0 → 0 |
| 0.1|1 | excluded_middle | 37 | 2.1e-05 → 1.1e-13 | 0 → 0.2 |
| 0.1|1 | negated_conditional | 15 | 0.0048 → 1.2e-12 | 0 → 0 |
| 0.1|1 | none | 1 | 0.076 → 2.7e-16 | 0 → 0 |
| 0.1|1 | peirce | 12 | 8.9e-13 → 9.4e-22 | 0 → 0 |
| 0.1|1 | peirce_sequent | 19 | 1.7e-05 → 1.0e-06 | 1.8 → 62.7 |
| 0.1|2 | contraposition_conv | 18 | 0.014 → 1.7e-12 | 0 → 0 |
| 0.1|2 | demorgan_nand_to_or | 4 | 3.3e-04 → 1.3e-11 | 0 → 0 |
| 0.1|2 | excluded_middle | 37 | 1.3e-10 → 6.9e-12 | 0 → 0 |
| 0.1|2 | negated_conditional | 15 | 1.6e-06 → 2.1e-06 | 0 → 0 |
| 0.1|2 | none | 1 | 0.014 → 4.9e-11 | 0 → 0 |
| 0.1|2 | peirce | 12 | 2.0e-15 → 5.0e-17 | 0 → 0 |
| 0.1|2 | peirce_sequent | 19 | 0.003 → 2.6e-05 | 1.3 → 25.4 |
| 1|0 | contraposition_conv | 18 | 0.0014 → 8.1e-05 | 0 → 2.3 |
| 1|0 | demorgan_nand_to_or | 4 | 0.4 → 0.0013 | 0 → 0 |
| 1|0 | excluded_middle | 37 | 2.4e-05 → 0.0015 | 0 → 0 |
| 1|0 | negated_conditional | 15 | 0.2 → 4.4 | 0 → 0 |
| 1|0 | none | 1 | 0.06 → 1.3e-06 | 8.3 → 0 |
| 1|0 | peirce | 12 | 4.0e-12 → 4.9e-08 | 0 → 0 |
| 1|0 | peirce_sequent | 19 | 0.0064 → 0.073 | 0.9 → 4.4 |
| 1|1 | contraposition_conv | 18 | 0.031 → 0.0011 | 3.7 → 58.3 |
| 1|1 | demorgan_nand_to_or | 4 | 0.021 → 1.1 | 0 → 0 |
| 1|1 | excluded_middle | 37 | 0.057 → 9.9e-04 | 0 → 0.9 |
| 1|1 | negated_conditional | 15 | 0.4 → 3.4 | 0 → 6.1 |
| 1|1 | none | 1 | 0.8 → 0.085 | 0 → 0 |
| 1|1 | peirce | 12 | 2.5e-12 → 1.5e-16 | 0 → 0 |
| 1|1 | peirce_sequent | 19 | 0.01 → 0.075 | 6.1 → 43.9 |
| 1|2 | contraposition_conv | 18 | 0.0013 → 2.4e-04 | 0 → 28.7 |
| 1|2 | demorgan_nand_to_or | 4 | 0.034 → 5.7e-05 | 0 → 0 |
| 1|2 | excluded_middle | 37 | 1.2e-04 → 1.2e-06 | 0 → 0 |
| 1|2 | negated_conditional | 15 | 1.3e-04 → 9.6e-05 | 0 → 0 |
| 1|2 | none | 1 | 0.093 → 0.0025 | 0 → 100.0 |
| 1|2 | peirce | 12 | 2.1e-09 → 2.7e-15 | 0 → 0 |
| 1|2 | peirce_sequent | 19 | 0.0022 → 0.014 | 0.9 → 65.4 |
| 0|0 | contraposition_conv | 18 | 5.3e-09 → 2.3e-17 | 0 → 0 |
| 0|0 | demorgan_nand_to_or | 4 | 6.9e-08 → 3.4e-26 | 0 → 0 |
| 0|0 | excluded_middle | 37 | 4.6e-09 → 1.4e-16 | 0 → 0.5 |
| 0|0 | negated_conditional | 15 | 4.3e-11 → 1.1e-25 | 0 → 0 |
| 0|0 | none | 1 | 1.6e-04 → 5.0e-19 | 0 → 0 |
| 0|0 | peirce | 12 | 4.4e-09 → 6.9e-17 | 0 → 0 |
| 0|0 | peirce_sequent | 19 | 1.7e-04 → 1.2e-07 | 15.8 → 43.9 |
| 0|1 | contraposition_conv | 18 | 1.0e-06 → 2.6e-20 | 0 → 0 |
| 0|1 | demorgan_nand_to_or | 4 | 1.7e-06 → 3.6e-23 | 0 → 0 |
| 0|1 | excluded_middle | 37 | 7.0e-07 → 4.1e-18 | 0 → 0 |
| 0|1 | negated_conditional | 15 | 7.8e-05 → 1.8e-16 | 0 → 0 |
| 0|1 | none | 1 | 3.6e-05 → 1.8e-20 | 0 → 0 |
| 0|1 | peirce | 12 | 2.0e-14 → 2.6e-21 | 0 → 0 |
| 0|1 | peirce_sequent | 19 | 8.9e-05 → 1.0e-09 | 4.8 → 75.9 |
| 0|2 | contraposition_conv | 18 | 7.3e-09 → 6.1e-23 | 0 → 0 |
| 0|2 | demorgan_nand_to_or | 4 | 6.0e-12 → 1.1e-21 | 0 → 0 |
| 0|2 | excluded_middle | 37 | 1.4e-07 → 2.3e-15 | 0 → 0 |
| 0|2 | negated_conditional | 15 | 5.9e-10 → 4.5e-21 | 0 → 0.6 |
| 0|2 | none | 1 | 3.0e-05 → 1.6e-17 | 0 → 0 |
| 0|2 | peirce | 12 | 4.8e-09 → 1.1e-26 | 0 → 0 |
| 0|2 | peirce_sequent | 19 | 1.7e-05 → 1.1e-05 | 11.8 → 42.5 |

## Inputs

- `dd01_s0.jsonl` sha256 `04a21ddd7879fa79ab81f35351f7410d1c6175472d8b558373c6d71143811915`
- `dd01_s1.jsonl` sha256 `804f104d49fc2ed2f2c313641ce5fedf388f76783fd8ab0ed74ef091d8572d0b`
- `dd01_s2.jsonl` sha256 `84577e63b8a43e8bed664224c3643ea0a9663db2fdc6ec1477d0f4d423fbbb77`
- `dd1_s0.jsonl` sha256 `bc638bae717bc02ab80573a8e6090adf4bd1c256db4aeb54bdfe2595ced8b2f4`
- `dd1_s1.jsonl` sha256 `07703c7bdc868b82c2866ac820581f307c8e5bf15f20520b56892ec48957a68d`
- `dd1_s2.jsonl` sha256 `947ae144bad9e82db9624ea28d1a1261533f6dd35f7d15961e4c07631467bc9e`
- `dko_s0.jsonl` sha256 `5c1c4d13ebb6acffd9826fffdefd5928f73eafd5bf0f24e2f09075c498e6b10b`
- `dko_s1.jsonl` sha256 `d4bd8461d8d301976166c99a6f55ab972b254fda0026e6d75c1315c6884c578e`
- `dko_s2.jsonl` sha256 `5a9d02473f4b626d43bdc0d17176c6661687027e61ca9d970b572d33e7e3f627`
- `dose01_s0.jsonl` sha256 `0d59f78af02a2496934bef533dd5bc5a4aa8ae7bfcf99bddc83400fc8c27be58`
- `dose01_s1.jsonl` sha256 `5d88388eb5862aacebccb4dcd79ede24ad438851f829614687e0866fb84114a6`
- `dose01_s2.jsonl` sha256 `444ba308add674ac09eeb254fdbe3b5d70d7d21a8de8e88fc034ba0ca9cfca43`
- `dose1_s0.jsonl` sha256 `c17e5611f37665d01cee619971a8a47653e580079d5a01f935f2d8fe38e3f2b3`
- `dose1_s1.jsonl` sha256 `3fa7c87b3105c523007f626ec48e99b36c1bdb2cbefd744cb391e93eb995aeaa`
- `dose1_s2.jsonl` sha256 `afd74315a960628205e291a85dceb064b85b7dcd6af0ec3870313452bf486e9b`
- `full_s3.jsonl` sha256 `08399e200bd8e4654bd4af2ca978df2c0c63c9a15372d90f897c0cd92417b464`
- `ko_s0.jsonl` sha256 `fd529e75a1c3663dbeecb27a0670982d38737aef8b25cc9dd758c632029bce30`
- `ko_s1.jsonl` sha256 `a89af7a359e6227c01999921ccce7f090c2f7208237b895e7e997c82923ee35b`
- `ko_s2.jsonl` sha256 `b373c57f415e91311931a2e71351d3c12a0a067b4350a0f18d08ca41f0d4743c`
- `m_dd01_s0_it60.jsonl` sha256 `eea782445e87b5c2a92602fa6b5353bc7f45089fd1cd56c3cb61e9290bee3e74`
- `m_dko_s0_it60.jsonl` sha256 `3f058deeb62e33128017bf09e2d528e23ed005e63024279d8145055149affe64`
- `m_dko_s1_it70.jsonl` sha256 `829a6abf0244e9f90424eba1b64afd68d9b9778466c505ae93420cb357114096`
- `m_dko_s2_it100.jsonl` sha256 `83bfebcca2dfd21dd73802df47d3495f9f220c45e4cfe47d02615685d89365bf`
