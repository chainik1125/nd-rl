**Conclusion.** Arm D composes short ANDI theorems nearly as well as their parts, but not long ones. Arm C cannot join
even short ones.

- **Short theorems (joined L6-9, one seed each).**
  - D_s101 solves 27.3% of joined theorems against a part product of 34.0%, a ratio of 0.80 [0.67, 0.94].
  - C_s101 solves 4.0% against 27.5%, a ratio of 0.14 [0.10, 0.20].
  - At L6-7, D's joined rate reaches the product: 40.7% vs 40.2% at L6, and 38.5% vs 31.2% at L7.
- **Longer theorems (joined L10-14).**
  - The ratio falls to 0.04 [0.02, 0.07]: D solves 1.1% against a 26.2% product.
  - Even at L10-11, inside D's training range, D solves only 3.8% and 1.5% against 34.1% and 28.8%.
- **Parts alone improve a little:** mean part pass@1 is 58.8% vs 53.0% at L6-9, and 50.4% vs 45.4% at L10-14.
- **Attempts get longer:** among D's joined attempts, 69.5% reach L_true at L6-9 (C: 46.9%), and 13.2% at L10-14
  (C: 6.9%).
- **Failures change shape.** With all seeds pooled at L6-9, D writes the joined goal with an invalid step less often
  (12% vs 21% of failures) and cites the other part less often (18% vs 23%). It more often gets as far as a conjunct but
  writes it with an invalid step (53% vs 33%).
