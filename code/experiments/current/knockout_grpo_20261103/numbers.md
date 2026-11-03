# knockout_grpo numbers

Bar (median E32 of the full bases, n 256): **26.31** of 106 (24.8 %); Dan-matched single-draw bar 27. E32 = expected number of the 106 solved by a k 32 draw (unbiased pass@32 from 64 attempts per checkpoint, 256 per base). R = (6 trained + 2 sampled tokens) / (6 Stage-1 tokens); c = GRPO iterations (L2 39, then L3). First crossing: the first checkpoint at or above the bar, linear in log10 R from the one before (≤ when it is the first checkpoint). Held: Amendment 1's window (≥ 3 consecutive checkpoints ≥ bar, the last at ≥ 2 × the first's R). σ share: of the distinct Lean-accepted training proofs of an iteration (train_sigma.py), at L2 iteration 1 → the last logged iteration (max).

| line | base E32 (% of 106) | first crossing R (bracket) | GPU-h to it | held window | last checkpoint (c, R, E32) | σ share of accepted training proofs |
|---|---|---|---|---|---|---|
| full_s0 | 22.34 (21.1 %) | ≤ 0.00286 (first checkpoint, c 1) | 0.03 | c 1–c 3, done at R 0.0101 | c 3, R 0.0101, 29.3 |  |
| full_s0_dp | 22.34 (21.1 %) | 0.00881 (c 2–c 3) | 0.08 | c 3–c 7, done at R 0.0212 | c 9, R 0.0268, 30.6 | 0.6 → 1.3 % at L2 it 9 (max 2.2) |
| full_s1 | 26.31 (24.8 %) | at_bar_before_rl |  |  | — |  |
| full_s2 | 29.33 (27.7 %) | at_bar_before_rl |  |  | — |  |
| d1_s0 | 14.39 (13.6 %) | 0.00758 (c 2–c 3) | 0.09 | c 3–c 6, done at R 0.0207 | c 9, R 0.0327, 39.5 |  |
| d1_s1 | 14.39 (13.6 %) | 0.0123 (c 4–c 5) | 0.16 | c 5–c 10, done at R 0.0355 | c 18, R 0.066, 47.8 |  |
| d1_s2 | 32.63 (30.8 %) | at_bar_before_rl |  |  | — |  |
| d01_s0 | 12.29 (11.6 %) | 0.0298 (c 10–c 11) | 0.31 | c 11–c 20, done at R 0.0673 | c 20, R 0.0673, 40.1 |  |
| d01_s1 | 6.67 (6.3 %) | 0.0243 (c 9–c 10) | 0.20 | c 10–c 20, done at R 0.0586 | c 20, R 0.0586, 35.0 |  |
| d01_s2 | 5.21 (4.9 %) | 0.0241 (c 7–c 8) | 0.37 | c 8–c 16, done at R 0.0552 | c 18, R 0.0631, 41.9 |  |
| ko_s0 | 0.73 (0.7 %) | 0.0653 (c 18–c 20) | 0.81 | c 20–c 39, done at R 0.142 | c 40, R 0.146, 36.6 |  |
| ko_s1 | 0.36 (0.3 %) | 0.0403 (c 14–c 16) | 0.56 | c 16–c 32, done at R 0.097 | c 32, R 0.097, 45.9 |  |
| ko_s2 | 1.04 (1.0 %) | 0.0368 (c 11–c 13) | 0.51 | c 13–c 25, done at R 0.0923 | c 25, R 0.0923, 40.8 |  |
| kos_s0 | 0.00 (0.0 %) | not reached by 0.207 (best 21.1); running | 1.84 |  | c 63, R 0.207, 19.6 | 0.3 → 27.4 % at L3 it 6 (max 30.9) |
| kos_s0_dp | 0.00 (0.0 %) | not reached by 0.156 (best 0.5); running | 1.74 |  | c 63, R 0.156, 0.0 | 0.0 → 0.1 % at L3 it 5 (max 0.8) |
| kos_s1 | 0.00 (0.0 %) | not reached by 0.109 (best 0.5); think-ban stop in L3 | 1.37 |  | c 40, R 0.109, 0.0 | 0.0 → 0.0 % at L2 it 35 (max 0.1) |
| kos_s1_dp | 0.00 (0.0 %) | not reached by 0.127 (best 0.8); running | 1.63 |  | c 50, R 0.127, 0.0 | 0.0 → 0.0 % at L2 it 38 (max 0.4) |
| kos_s2 | 0.33 (0.3 %) | not reached by 0.168 (best 7.0); running | 1.83 |  | c 50, R 0.168, 7.0 | 0.0 → 2.1 % at L2 it 36 (max 10.1) |
| kos_s2_dp | 0.33 (0.3 %) | not reached by 0.153 (best 0.9); running | 1.71 |  | c 63, R 0.153, 0.0 | 0.0 → 0.0 % at L3 it 6 (max 0.8) |

## Think-ban guard events (think_ban.py; distinct GRPO iterations)

| line | iterations | think-ban alarms (% of iterations) | rollbacks | alarms with no rollback | stops |
|---|---|---|---|---|---|
| d01_s0 | 21 | 0 (0.0 %) | 0 | 0 | 0 |
| d01_s1 | 21 | 0 (0.0 %) | 0 | 0 | 0 |
| d01_s2 | 18 | 0 (0.0 %) | 0 | 0 | 0 |
| d1_s0 | 9 | 0 (0.0 %) | 0 | 0 | 0 |
| d1_s1 | 18 | 0 (0.0 %) | 0 | 0 | 0 |
| full_s0 | 3 | 0 (0.0 %) | 0 | 0 | 0 |
| full_s0_dp | 9 | 0 (0.0 %) | 0 | 0 | 0 |
| ko_s0 | 41 | 0 (0.0 %) | 0 | 0 | 0 |
| ko_s1 | 33 | 0 (0.0 %) | 0 | 0 | 0 |
| ko_s2 | 26 | 0 (0.0 %) | 0 | 0 | 0 |
| kos_s0 | 57 | 0 (0.0 %) | 0 | 0 | 0 |
| kos_s0_dp | 59 | 0 (0.0 %) | 0 | 0 | 0 |
| kos_s1 | 41 | 19 (46.3 %) | 1 | 17 | 1 |
| kos_s1_dp | 51 | 0 (0.0 %) | 0 | 0 | 0 |
| kos_s2 | 46 | 5 (10.9 %) | 3 | 2 | 0 |
| kos_s2_dp | 60 | 4 (6.7 %) | 1 | 3 | 0 |

## Dan's EI ladders on our axis (dan_axis.py)

His x ÷ our R: round 1 (sampling only; his x counts the re-read state prefill) 4.98–5.41; every later round (fine-tuning and replay added) 1.53–1.90.

| ladder | arm | crossing round | his x | our R | his ÷ ours | not reached: his / our last |
|---|---|---|---|---|---|---|
| la_ko_s0 | knockout | — |  |  | 1.54 | 5.84 / 3.80 |
| la_ko_s1 | knockout | — |  |  | 1.55 | 13.23 / 8.56 |
| la_ko_s2 | knockout | — |  |  | 1.54 | 5.90 / 3.82 |
| la_ko_s3 | knockout | — |  |  | 1.53 | 5.58 / 3.64 |
| la_ko_s4 | knockout | — |  |  | 1.55 | 6.13 / 3.96 |
| la_ko_s5 | knockout | — |  |  | 1.55 | 5.99 / 3.87 |
| la_full_s3 | full | r2 | 0.39 | 0.211 | 1.83 |  |
| la_full_s4 | full | r2 | 0.39 | 0.212 | 1.86 |  |
| la_full_s5 | full | r1 | 0.11 | 0.020 | 5.16 |  |
| la_dose1_s0 | dose1 | r2 | 0.39 | 0.204 | 1.90 |  |
| la_dose1_s1 | dose1 | r2 | 0.41 | 0.214 | 1.89 |  |
| la_dose1_s2 | dose1 | r3 | 0.73 | 0.437 | 1.67 |  |
| la_dose01_s0 | dose01 | r5 | 1.37 | 0.859 | 1.60 |  |
| la_dose01_s1 | dose01 | r8 | 2.58 | 1.656 | 1.56 |  |
| la_dose01_s2 | dose01 | r3 | 0.69 | 0.414 | 1.67 |  |
| old_full_s0 | full | r1 | 0.10 | 0.020 | 5.23 |  |
| old_full_s1 | full | r2 | 0.43 | 0.231 | 1.85 |  |
| old_full_s2 | full | r1 | 0.10 | 0.019 | 5.10 |  |
