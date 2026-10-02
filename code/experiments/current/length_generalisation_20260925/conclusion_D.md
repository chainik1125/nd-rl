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
