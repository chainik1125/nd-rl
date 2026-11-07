Pilot dd01g0_s2: status done, x_dan 3.02, GPU-h 7.2, it 293, crossing None, stop compute cap: x_dan 3.02 ≥ the pilot's 3.

| round (his x) | his measure, pilot / dd01_s2 | draw (first 32) | σ %, pilot / dd01_s2 | pass@1 %, pilot / dd01_s2 | lr ×, pilot / dd01_s2 | iterations with an alarm by type, pilot / dd01_s2 |
|---|---|---|---|---|---|---|
| 1 (0.09) | 1 / 1 | 1 / 1 | 0.07 / 0.01 | 64 / 53 | 1 / 0.5 | grad-norm 1 / canary 1 |
| 2 (0.37) | 1 / 1 | 0 / 0 | 0.04 / 0.05 | 91 / 88 | 1 / 0.25 | grad-norm 1 / canary 1, clip-low 1 |
| 3 (0.69) | 1 / 1 | 0 / 0 | 0.06 / 0.06 | 91 / 88 | 1 / 0.25 | grad-norm 1 / canary 1, clip-low 1 |
| 4 (1.03) | 1 / 1 | 0 / 0 | 0.07 / 0.01 | 91 / 91 | 1 / 0.25 | canary 3, grad-norm 2 / canary 1, clip-low 1 |
| 5 (1.38) | 1 / 1 | 0 / 0 | 0.11 / 0.04 | 91 / 92 | 1 / 0.25 | canary 3, grad-norm 2 / canary 1, clip-low 1 |
| 6 (1.75) | 1 / 1 | 0 / 0 | 0.09 / 0.02 | 93 / 92 | 1 / 0.125 | canary 3, grad-norm 2 / canary 8, clip-low 2 |
| 7 (2.12) | 1 / 1 | 0 / 0 | 0.14 / 0.02 | 93 / 90 | 1 / 0.125 | canary 18, grad-norm 2 / canary 35, clip-low 5 |
| 8 (2.50) | 1 / 1 | 0 / 0 | 0.20 / 0.00 | 91 / 90 | 1 / 0.125 | canary 42, clip-low 1, grad-norm 2 / canary 60, clip-low 10 |
| 9 (2.89) | 1 / 1 | 0 / 0 | 0.10 / 0.01 | 90 / 90 | 1 / 0.125 | canary 60, clip-low 1, grad-norm 2 / canary 83, clip-low 11 |

σ over the pilot's last 50 iterations: 0.16 %.
Collapse (as pre-registered): no.
