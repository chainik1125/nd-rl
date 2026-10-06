# validation-36 (Lean 4.34.0 alone), greedy / pass@32 / pass@256 at T 0.8

| checkpoint | all (36) | <=6 (12) | >6 (24) |
|---|---|---|---|
| stage1_C0_s0 | 6 / 10.19 / 12 | 6 / 9.96 / 11 | 0 / 0.23 / 1 |
| stage1_C0_s1 | 7 / 10.99 / 11 | 7 / 10.99 / 11 | 0 / 0.0 / 0 |
| stage1_S_s0 | 7 / 11.2 / 13 | 7 / 9.26 / 11 | 0 / 1.94 / 2 |
| stage1_S_s1 | 3 / 10.67 / 13 | 3 / 9.34 / 11 | 0 / 1.33 / 2 |
| stage1_SH_s0 | 8 / 11.38 / 13 | 8 / 11.26 / 12 | 0 / 0.12 / 1 |
| stage1_SH_s1 | 9 / 12.55 / 13 | 9 / 11.0 / 11 | 0 / 1.55 / 2 |
| stage1_SN_s0 | 7 / 11.73 / 13 | 7 / 9.53 / 10 | 0 / 2.2 / 3 |
| stage1_SN_s1 | 7 / 10.61 / 13 | 7 / 9.74 / 12 | 0 / 0.87 / 1 |
| T1_C0_s0_r8 | 10 / 11.54 / 13 | 10 / 11.0 / 11 | 0 / 0.54 / 2 |
| T1_C0_s1_r8 | 8 / 11.46 / 12 | 8 / 9.73 / 10 | 0 / 1.73 / 2 |
| T1_S_s0_r8 | 11 / 14.95 / 16 | 7 / 9.96 / 11 | 4 / 5.0 / 5 |
| T1_S_s1_r8 | 10 / 12.97 / 15 | 8 / 9.83 / 11 | 2 / 3.14 / 4 |
| T1_SH_s0_r8 | 9 / 13.62 / 14 | 8 / 10.78 / 11 | 1 / 2.85 / 3 |
| T1_SH_s1_r8 | 12 / 15.46 / 17 | 10 / 11.0 / 11 | 2 / 4.46 / 6 |
| T1_SN_s0_r8 | 13 / 17.01 / 18 | 9 / 10.81 / 11 | 4 / 6.21 / 7 |
| T1_SN_s1_r8 | 11 / 12.54 / 15 | 10 / 10.12 / 11 | 1 / 2.41 / 4 |

## >6-bin theorems solved at pass@256 (union over seeds)

- **stage1_C0** (1): disjunctive_syllogism (7)
- **stage1_S** (2): disjunctive_syllogism (7), export (7)
- **stage1_SH** (2): disjunctive_syllogism (7), export (7)
- **stage1_SN** (3): disjunctive_syllogism (7), export (7), demorgan_nor_to_and (10)
- **T1_C0** (2): contraposition (7), export (7)
- **T1_S** (6): reductio_pair (7), disjunctive_syllogism (7), contraposition (7), export (7), excluded_middle (10), demorgan_nor_to_and (10)
- **T1_SH** (6): disjunctive_syllogism (7), contraposition (7), export (7), negated_conditional_conv (7), demorgan_and_to_nor (10), resolution (14)
- **T1_SN** (7): disjunctive_syllogism (7), contraposition (7), export (7), demorgan_or_to_nand (10), demorgan_nor_to_and (10), dist_or_over_and (14), resolution (14)
