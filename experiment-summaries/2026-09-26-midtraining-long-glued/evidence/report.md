---
written_on: 2026-09-26
---
# Mid-training on long glued proofs: report

Generated 2026-09-26 07:37 UTC by `analyse.py`. One seed (RL seed 101). k = 32 samples per theorem at T = 0.8, max_new 1,024; pass@k are unbiased estimates averaged over theorems. Checkpoints evaluated: pre039 (039 s0 (no RL)), mt_long (mid-trained long (no RL)), mt_short (mid-trained short (no RL)), mt_more (mid-trained more (no RL)), base (base + RL), long (long + RL), short (short + RL), more (more + RL).

## Conclusion

**No. Mid-training on long glued proofs does not make RL generalise to longer proofs.** After the same RL (one
seed), the long arm is worse than the base arm on the natural long theorems. No RL arm solves any standard theorem
past length 13, and at most 0.2% of the matched constructed theorems at 14.

1. **Natural theorems at L12-14** (never glued; the main evidence).
   - long + RL solves 2.9%, against 5.3% for base (-2.4 points [-3.4, -1.5]), 4.4% for short (-1.5 [-2.4, -0.6])
     and 4.1% for more (-1.2 [-2.2, -0.3]).
   - At L13, long reaches 0.01%, against 1.2% for base.
2. **Chain theorems at L15-18** (a glue type never seen in training): 0% for every checkpoint, before and after RL.
   So are the ANDI theorems at 15-18 and the glued 22-26 set after RL.
3. **In distribution** (L6-11): every mid-trained arm is slightly better than base after RL (long +3.0 points [+1.8,
   +4.4]). Held-out greedy on cap-6 is 0.98 for all four RL arms.
4. **Mid-training does teach something, and RL removes it.**
   - Before RL, the mid-trained long model is the only checkpoint that solves anything at length 15 or more: 1.4%
     of the ANDI theorems at 15-18 and 0.15% of glued 22-26.
   - Before RL, 27-31% of its attempts at 15-18 are at least L_true long (median 12-15 lines), against 0-2% for the
     other pre-RL checkpoints. Its failures on the long set are ordinary invalid steps: premise errors are only 6% of
     failures.
   - After the arm-C RL, which trains only on proofs of at most 11 written lines, those solves are gone. Attempts at
     15-18 are back to a median of 10-11 lines, with at most 1% long enough.
   - On the long set, 49% of failures are again premise-copying errors (73% for base), and the failure hazard at
     line 5 rises from 0.02 to 0.52.
5. **Weight of the glued evidence.** The glued 22-26 test is built exactly like sft_long (3-part ANDI), so even the
   pre-RL solves there are close to in-distribution. They are weak evidence for length generalisation.

**Reading.** In this setup, RL capped at 11-line proofs overwrites whatever long-proof ability supervised
mid-training added, including the ability to copy long premise blocks. Whether RL without the line cap, or with long
targets, would keep it was not tested.

## Caveats

- **test_long is close to in-distribution for the long arm.** It is built exactly like sft_long: glued ANDI joins, all
  of them 3-part, from held-out parts. So the main evidence of generalisation is the natural theorems at L12-14 (never
  glued) and chain 15-18, a glue type absent from mid-training. Those are shown first; glued 22-26 comes last.
- **The 3-part share is tied to length.** sft_short has 3-part joins only at L9-11; sft_long is two thirds 3-part and
  test_long is all 3-part (shares in the data table below). The long-vs-short gap on test_long partly measures
  experience with the 3-part join shape, not only length.
- **Prompts are longer than in run 2** (mean prompt tokens per file and test set in the data table below). Part of
  what the long mix teaches is long prompts.
- **test_long labels** are constructed upper bounds with a proved lower bound of 15; additivity is checked only up to
  17 lines. pass@1 here means that the verifier accepted a proof of the prompted sequent. That does not depend on the
  label. Only the attempt-length share (written lines >= L_true) uses it.
- **No truncation.** Mid-training does not go through `lib/train.load`, so its line cap does not apply. It uses the
  seed recipe's own encode-and-pack path, which never cuts a record: every record is at most 800 tokens, and the
  packer raises an error above 1,024.
- **One seed.** Differences are given with a theorem bootstrap, which does not include seed-to-seed variation. For
  scale, arm C's three seeds in lengthexp spanned 3.1-3.9% pass@1 at L12.

## Headline comparisons (after RL)

pass@1 (a valid proof of the prompted sequent) pooled over the group; difference paired by theorem with a 95% theorem bootstrap. Groups in order of evidential weight: natural L12-14, chain 15-18 (held-out glue type), ANDI 15-18 (2-part joins, test construction), glued 22-26 (the mid-training construction).

| group | comparison | theorems | first | second | difference [95% CI] |
|---|---|---|---|---|---|
| natural L12-14 | long vs base | 758 | 2.89% | 5.31% | -2.42 pts [-3.35, -1.50] |
| natural L12-14 | long vs short | 758 | 2.89% | 4.38% | -1.50 pts [-2.42, -0.62] |
| natural L12-14 | long vs more | 758 | 2.89% | 4.10% | -1.21 pts [-2.19, -0.33] |
| chain 15-18 | long vs base | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| chain 15-18 | long vs short | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| chain 15-18 | long vs more | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| ANDI 15-18 | long vs base | 807 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| ANDI 15-18 | long vs short | 807 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| ANDI 15-18 | long vs more | 807 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| glued 22-26 | long vs base | 450 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| glued 22-26 | long vs short | 450 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| glued 22-26 | long vs more | 450 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| in-distribution L6-11 | long vs base | 1800 | 49.68% | 46.67% | +3.01 pts [+1.78, +4.36] |
| in-distribution L6-11 | long vs short | 1800 | 49.68% | 49.02% | +0.66 pts [-0.69, +1.89] |
| in-distribution L6-11 | long vs more | 1800 | 49.68% | 47.98% | +1.70 pts [+0.49, +2.92] |

## The same before RL (mid-trained checkpoints)

| group | comparison | theorems | first | second | difference [95% CI] |
|---|---|---|---|---|---|
| natural L12-14 | mt_long vs pre039 | 758 | 0.06% | 0.00% | +0.06 pts [+0.01, +0.13] |
| natural L12-14 | mt_long vs mt_short | 758 | 0.06% | 0.00% | +0.06 pts [+0.01, +0.13] |
| natural L12-14 | mt_long vs mt_more | 758 | 0.06% | 0.00% | +0.06 pts [+0.01, +0.13] |
| chain 15-18 | mt_long vs pre039 | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| chain 15-18 | mt_long vs mt_short | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| chain 15-18 | mt_long vs mt_more | 393 | 0.00% | 0.00% | +0.00 pts [+0.00, +0.00] |
| ANDI 15-18 | mt_long vs pre039 | 807 | 1.38% | 0.00% | +1.38 pts [+1.03, +1.74] |
| ANDI 15-18 | mt_long vs mt_short | 807 | 1.38% | 0.00% | +1.38 pts [+1.02, +1.76] |
| ANDI 15-18 | mt_long vs mt_more | 807 | 1.38% | 0.00% | +1.38 pts [+1.04, +1.75] |
| glued 22-26 | mt_long vs pre039 | 450 | 0.15% | 0.00% | +0.15 pts [+0.08, +0.23] |
| glued 22-26 | mt_long vs mt_short | 450 | 0.15% | 0.00% | +0.15 pts [+0.08, +0.24] |
| glued 22-26 | mt_long vs mt_more | 450 | 0.15% | 0.00% | +0.15 pts [+0.08, +0.23] |
| in-distribution L6-11 | mt_long vs pre039 | 1800 | 19.98% | 8.21% | +11.77 pts [+10.50, +13.09] |
| in-distribution L6-11 | mt_long vs mt_short | 1800 | 19.98% | 15.38% | +4.60 pts [+3.40, +5.82] |
| in-distribution L6-11 | mt_long vs mt_more | 1800 | 19.98% | 9.94% | +10.04 pts [+8.72, +11.43] |

## pass@1, std sets

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 38.88% | 10.29% | 0.08% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 44.75% | 31.96% | 17.84% | 16.82% | 6.49% | 2.00% | 0.16% | 0.00% | 0.00% | 0.68% | 1.01% | 0.98% | 1.04% |
| mid-trained short (no RL) | 49.84% | 27.64% | 13.05% | 1.71% | 0.01% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained more (no RL) | 43.26% | 16.09% | 0.26% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| base + RL | 78.79% | 72.12% | 42.57% | 36.38% | 26.62% | 23.53% | 12.19% | 1.23% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| long + RL | 77.56% | 77.92% | 48.17% | 40.23% | 30.86% | 23.34% | 7.28% | 0.01% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| short + RL | 80.93% | 72.72% | 45.61% | 40.71% | 30.21% | 23.94% | 10.01% | 1.06% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| more + RL | 79.93% | 74.40% | 44.95% | 38.31% | 28.16% | 22.14% | 9.97% | 0.39% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## pass@1, matched sets

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 25.41% | 2.57% | 0.01% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 23.08% | 21.58% | 19.71% | 16.18% | 8.81% | 5.19% | 2.93% | 1.57% | 1.02% |
| mid-trained short (no RL) | 49.81% | 48.53% | 39.26% | 27.70% | 11.74% | 3.16% | 0.39% | 0.04% | 0.00% |
| mid-trained more (no RL) | 24.41% | 6.07% | 0.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| base + RL | 33.40% | 37.33% | 34.47% | 29.65% | 22.73% | 12.99% | 5.82% | 0.58% | 0.11% |
| long + RL | 37.34% | 40.93% | 37.30% | 32.80% | 24.26% | 16.81% | 5.48% | 0.61% | 0.03% |
| short + RL | 40.74% | 46.23% | 41.99% | 33.46% | 25.59% | 17.51% | 7.66% | 1.56% | 0.03% |
| more + RL | 33.31% | 35.52% | 37.17% | 28.32% | 19.03% | 15.16% | 5.41% | 0.65% | 0.17% |

## pass@1, long sets

| checkpoint | L22 | L24 | L26 |
|---|---|---|---|
| 039 s0 (no RL) | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 0.21% | 0.12% | 0.10% |
| mid-trained short (no RL) | 0.00% | 0.00% | 0.00% |
| mid-trained more (no RL) | 0.00% | 0.00% | 0.00% |
| base + RL | 0.00% | 0.00% | 0.00% |
| long + RL | 0.00% | 0.00% | 0.00% |
| short + RL | 0.00% | 0.00% | 0.00% |
| more + RL | 0.00% | 0.00% | 0.00% |

## pass@4, std sets

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 49.94% | 14.24% | 0.26% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 63.44% | 44.01% | 28.11% | 25.99% | 12.62% | 4.49% | 0.55% | 0.00% | 0.00% | 2.29% | 3.03% | 2.94% | 3.02% |
| mid-trained short (no RL) | 64.23% | 37.94% | 19.08% | 3.44% | 0.04% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained more (no RL) | 51.24% | 21.66% | 0.60% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| base + RL | 89.99% | 80.49% | 51.26% | 41.71% | 31.77% | 27.89% | 17.90% | 2.39% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| long + RL | 90.15% | 89.07% | 58.07% | 48.41% | 37.63% | 28.13% | 10.49% | 0.04% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| short + RL | 90.95% | 82.02% | 54.31% | 46.95% | 37.23% | 29.03% | 14.75% | 2.04% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| more + RL | 89.44% | 82.93% | 53.00% | 43.70% | 34.12% | 27.88% | 14.85% | 0.60% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## pass@4, matched sets

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 39.46% | 5.97% | 0.04% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 43.49% | 41.03% | 40.84% | 34.91% | 21.74% | 14.60% | 8.84% | 4.81% | 2.95% |
| mid-trained short (no RL) | 69.53% | 69.46% | 61.69% | 45.77% | 22.57% | 7.75% | 1.05% | 0.14% | 0.00% |
| mid-trained more (no RL) | 35.37% | 10.75% | 0.46% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| base + RL | 50.97% | 53.91% | 48.86% | 41.61% | 31.74% | 18.66% | 8.07% | 1.27% | 0.37% |
| long + RL | 58.54% | 60.22% | 55.56% | 47.79% | 35.66% | 24.79% | 9.74% | 1.64% | 0.11% |
| short + RL | 61.62% | 63.95% | 57.19% | 47.94% | 37.19% | 24.82% | 12.34% | 3.34% | 0.12% |
| more + RL | 50.29% | 51.06% | 51.44% | 40.61% | 25.50% | 20.99% | 8.45% | 1.07% | 0.32% |

## pass@4, long sets

| checkpoint | L22 | L24 | L26 |
|---|---|---|---|
| 039 s0 (no RL) | 0.00% | 0.00% | 0.00% |
| mid-trained long (no RL) | 0.82% | 0.49% | 0.39% |
| mid-trained short (no RL) | 0.00% | 0.00% | 0.00% |
| mid-trained more (no RL) | 0.00% | 0.00% | 0.00% |
| base + RL | 0.00% | 0.00% | 0.00% |
| long + RL | 0.00% | 0.00% | 0.00% |
| short + RL | 0.00% | 0.00% | 0.00% |
| more + RL | 0.00% | 0.00% | 0.00% |

## pass@1 / pass@4 by group

| checkpoint | in-distribution L6-11 | natural L12-14 | chain 15-18 | ANDI 15-18 | glued 22-26 |
|---|---|---|---|---|---|
| 039 s0 (no RL) | 8.21% / 10.74% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| mid-trained long (no RL) | 19.98% / 29.78% | 0.06% / 0.22% | 0.00% / 0.00% | 1.38% / 4.19% | 0.15% / 0.57% |
| mid-trained short (no RL) | 15.38% / 20.79% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| mid-trained more (no RL) | 9.94% / 12.25% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| base + RL | 46.67% / 53.85% | 5.31% / 8.03% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| long + RL | 49.68% / 58.58% | 2.89% / 4.17% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| short + RL | 49.02% / 56.75% | 4.38% / 6.64% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| more + RL | 47.98% / 55.18% | 4.10% / 6.12% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |

## Attempt lengths, std sets

Cell: median written lines / share of samples with at least L_true written lines.

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 6 / 80% | 6 / 25% | 6 / 2% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 2% | 7 / 1% | 7 / 0% | 7 / 0% |
| mid-trained long (no RL) | 6 / 86% | 7 / 68% | 7 / 44% | 7 / 36% | 8 / 22% | 8 / 12% | 8 / 6% | 9 / 3% | 6 / 1% | 12 / 30% | 12 / 27% | 14 / 31% | 15 / 29% |
| mid-trained short (no RL) | 6 / 86% | 7 / 60% | 7 / 29% | 7 / 9% | 7 / 3% | 7 / 2% | 7 / 3% | 7 / 0% | 6 / 0% | 9 / 0% | 9 / 0% | 10 / 0% | 10 / 0% |
| mid-trained more (no RL) | 6 / 86% | 6 / 41% | 6 / 3% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 6 / 0% | 7 / 1% | 7 / 1% | 7 / 0% | 7 / 0% |
| base + RL | 6 / 99% | 7 / 92% | 8 / 67% | 9 / 53% | 9 / 40% | 9 / 34% | 9 / 19% | 9 / 4% | 7 / 0% | 10 / 1% | 10 / 1% | 10 / 1% | 11 / 1% |
| long + RL | 6 / 99% | 7 / 93% | 8 / 68% | 9 / 57% | 9 / 45% | 9 / 36% | 9 / 12% | 10 / 0% | 7 / 0% | 10 / 0% | 10 / 0% | 10 / 0% | 10 / 0% |
| short + RL | 6 / 99% | 7 / 94% | 8 / 69% | 9 / 56% | 9 / 42% | 9 / 34% | 9 / 15% | 10 / 2% | 7 / 0% | 10 / 1% | 10 / 0% | 11 / 1% | 11 / 0% |
| more + RL | 6 / 99% | 7 / 90% | 8 / 67% | 9 / 53% | 8 / 40% | 9 / 33% | 9 / 15% | 10 / 2% | 7 / 0% | 10 / 0% | 10 / 0% | 10 / 0% | 10 / 0% |

## Attempt lengths, matched sets

Cell: median written lines / share of samples with at least L_true written lines.

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 6 / 82% | 6 / 37% | 6 / 5% | 6 / 1% | 6 / 2% | 6 / 2% | 6 / 2% | 6 / 1% | 6 / 3% |
| mid-trained long (no RL) | 8 / 95% | 8 / 93% | 9 / 88% | 9 / 76% | 10 / 66% | 11 / 56% | 11 / 46% | 11 / 37% | 11 / 31% |
| mid-trained short (no RL) | 7 / 97% | 7 / 91% | 8 / 81% | 9 / 61% | 9 / 36% | 9 / 18% | 9 / 6% | 9 / 2% | 9 / 1% |
| mid-trained more (no RL) | 6 / 87% | 7 / 55% | 7 / 10% | 7 / 2% | 7 / 3% | 7 / 2% | 7 / 1% | 7 / 2% | 7 / 2% |
| base + RL | 8 / 94% | 8 / 90% | 8 / 85% | 9 / 71% | 10 / 59% | 10 / 40% | 10 / 20% | 10 / 5% | 10 / 1% |
| long + RL | 8 / 96% | 8 / 92% | 8 / 88% | 9 / 74% | 10 / 58% | 10 / 38% | 10 / 17% | 10 / 3% | 10 / 0% |
| short + RL | 7 / 94% | 8 / 91% | 8 / 88% | 9 / 74% | 10 / 62% | 10 / 46% | 10 / 24% | 10 / 6% | 10 / 2% |
| more + RL | 8 / 95% | 8 / 91% | 8 / 86% | 9 / 68% | 10 / 51% | 10 / 35% | 10 / 16% | 10 / 2% | 10 / 0% |

## Attempt lengths, long sets

Cell: median written lines / share of samples with at least L_true written lines.

| checkpoint | L22 | L24 | L26 |
|---|---|---|---|
| 039 s0 (no RL) | 21 / 49% | 21 / 45% | 20 / 39% |
| mid-trained long (no RL) | 21 / 48% | 22 / 28% | 22 / 16% |
| mid-trained short (no RL) | 13 / 1% | 12 / 1% | 13 / 0% |
| mid-trained more (no RL) | 19 / 43% | 18 / 38% | 19 / 34% |
| base + RL | 11 / 4% | 11 / 3% | 11 / 2% |
| long + RL | 11 / 1% | 11 / 1% | 11 / 0% |
| short + RL | 11 / 1% | 11 / 1% | 11 / 1% |
| more + RL | 10 / 3% | 10 / 2% | 10 / 2% |

## Per-line failure hazard, std L15-18

P(line p fails | lines 1..p-1 valid), positions with at least 100 samples at risk.

| checkpoint | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 | p16 | p17 | p18 | p19 | p20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 0.00 | 0.03 | 0.06 | 0.52 | 0.92 | 0.91 | 0.98 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mid-trained long (no RL) | 0.00 | 0.00 | 0.02 | 0.03 | 0.11 | 0.24 | 0.35 | 0.37 | 0.31 | 0.23 | 0.22 | 0.25 | 0.28 | 0.30 | 0.27 | 0.19 | 0.24 | 0.27 | 0.27 | 0.43 |
| mid-trained short (no RL) | 0.00 | 0.00 | 0.00 | 0.04 | 0.21 | 0.52 | 0.58 | 0.53 | 0.54 | 0.73 | 0.75 |  |  |  |  |  |  |  |  |  |
| mid-trained more (no RL) | 0.01 | 0.02 | 0.05 | 0.26 | 0.88 | 0.91 | 0.95 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| base + RL | 0.00 | 0.00 | 0.00 | 0.02 | 0.07 | 0.14 | 0.27 | 0.39 | 0.39 | 0.45 | 0.52 | 0.73 | 0.88 |  |  |  |  |  |  |  |
| long + RL | 0.00 | 0.00 | 0.01 | 0.02 | 0.07 | 0.16 | 0.29 | 0.42 | 0.44 | 0.51 | 0.64 | 0.91 | 0.85 |  |  |  |  |  |  |  |
| short + RL | 0.00 | 0.01 | 0.01 | 0.02 | 0.09 | 0.17 | 0.25 | 0.35 | 0.41 | 0.42 | 0.52 | 0.68 | 0.81 | 0.99 |  |  |  |  |  |  |
| more + RL | 0.00 | 0.00 | 0.01 | 0.03 | 0.08 | 0.17 | 0.33 | 0.43 | 0.48 | 0.54 | 0.62 | 0.81 | 0.97 |  |  |  |  |  |  |  |

## Per-line failure hazard, long L22-26

P(line p fails | lines 1..p-1 valid), positions with at least 100 samples at risk.

| checkpoint | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 | p16 | p17 | p18 | p19 | p20 | p21 | p22 | p23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 039 s0 (no RL) | 0.09 | 0.72 | 0.97 | 0.97 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mid-trained long (no RL) | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.07 | 0.21 | 0.26 | 0.23 | 0.16 | 0.14 | 0.13 | 0.18 | 0.21 | 0.21 | 0.22 | 0.21 | 0.27 | 0.32 | 0.38 | 0.33 | 0.38 | 0.41 |
| mid-trained short (no RL) | 0.00 | 0.02 | 0.02 | 0.02 | 0.09 | 0.34 | 0.74 | 0.49 | 0.37 | 0.44 | 0.62 | 0.46 |  |  |  |  |  |  |  |  |  |  |  |
| mid-trained more (no RL) | 0.13 | 0.71 | 0.82 | 0.89 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| base + RL | 0.06 | 0.14 | 0.13 | 0.38 | 0.77 | 0.13 | 0.21 | 0.48 | 0.42 | 0.29 | 0.54 | 0.86 |  |  |  |  |  |  |  |  |  |  |  |
| long + RL | 0.01 | 0.07 | 0.12 | 0.15 | 0.52 | 0.21 | 0.25 | 0.42 | 0.35 | 0.43 | 0.60 | 0.72 |  |  |  |  |  |  |  |  |  |  |  |
| short + RL | 0.03 | 0.11 | 0.14 | 0.34 | 0.65 | 0.07 | 0.19 | 0.35 | 0.30 | 0.32 | 0.64 | 0.77 |  |  |  |  |  |  |  |  |  |  |  |
| more + RL | 0.07 | 0.15 | 0.18 | 0.45 | 0.67 | 0.19 | 0.27 | 0.47 | 0.19 | 0.29 | 0.43 | 0.72 |  |  |  |  |  |  |  |  |  |  |  |

## First-error reasons of failed samples, std L15-18

Share of failed samples by the reason their first line is rejected: premise = the premise lines are wrong (a premise mis-copied, skipped or out of order); parse = a malformed line; rule = an invalid inference; cite = a citation out of scope.

| checkpoint | premise | parse | rule | cite | other |
|---|---|---|---|---|---|
| 039 s0 (no RL) | 54% | 5% | 36% | 4% | 2% |
| mid-trained long (no RL) | 6% | 1% | 75% | 13% | 4% |
| mid-trained short (no RL) | 2% | 1% | 88% | 7% | 2% |
| mid-trained more (no RL) | 30% | 6% | 56% | 4% | 3% |
| base + RL | 5% | 4% | 73% | 14% | 4% |
| long + RL | 5% | 3% | 73% | 16% | 3% |
| short + RL | 5% | 2% | 73% | 17% | 3% |
| more + RL | 6% | 2% | 76% | 13% | 3% |

## First-error reasons of failed samples, long L22-26

Share of failed samples by the reason their first line is rejected: premise = the premise lines are wrong (a premise mis-copied, skipped or out of order); parse = a malformed line; rule = an invalid inference; cite = a citation out of scope.

| checkpoint | premise | parse | rule | cite | other |
|---|---|---|---|---|---|
| 039 s0 (no RL) | 76% | 24% | 0% | 0% | 0% |
| mid-trained long (no RL) | 6% | 2% | 72% | 15% | 4% |
| mid-trained short (no RL) | 9% | 8% | 74% | 6% | 3% |
| mid-trained more (no RL) | 85% | 15% | 0% | 0% | 0% |
| base + RL | 73% | 17% | 7% | 1% | 1% |
| long + RL | 49% | 26% | 18% | 4% | 2% |
| short + RL | 62% | 23% | 11% | 4% | 1% |
| more + RL | 69% | 20% | 7% | 2% | 2% |

## Mid-training

Same steps, batch, schedule and cap-6 records in all three arms (`midtrain.py`, `ckpt/mt_<arm>.json`).

| checkpoint | records | mix | steps | tokens | wall | loss (first -> last logged step) | held-out greedy (before -> after) |
|---|---|---|---|---|---|---|---|
| mt_long | 10000 | sft_long 2000, cap6 8000 | 234 | 5992948 | 34 s | 0.093 -> 0.047 | 0.978 -> 0.951 |
| mt_short | 10000 | sft_short 2000, cap6 8000 | 234 | 4287061 | 24 s | 0.063 -> 0.047 | 0.978 -> 0.977 |
| mt_more | 10000 | cap6 10000 | 234 | 3250348 | 18 s | 0.054 -> 0.052 | 0.978 -> 0.975 |

## RL runs (arm-C recipe, seed 101, DEFAULT_FLOPS)

| arm | status | rounds | samples | held-out greedy | compute used | wall | pretrain.pt sha256 |
|---|---|---|---|---|---|---|---|
| base | ok | 59 | 747008 | 0.984 | 0.996 | 37.3 min | 08e2644a8560 |
| long | ok | 56 | 711680 | 0.983 | 0.996 | 29.6 min | 295e868872bd |
| short | ok | 57 | 725248 | 0.983 | 0.996 | 36.6 min | 2ed38aa12d3e |
| more | ok | 58 | 742400 | 0.982 | 0.994 | 29.5 min | a36039768584 |

## Data: prompt size and join shape

Mean prompt tokens (one token per symbol), length range (written lines for training files, L_true for test sets) and share of 3-part joins.

| file / set | records | mean prompt tokens | lines | 3-part share |
|---|---|---|---|---|
| sft_long (mid-training) | 2000 | 118 | 16-24 | 67% |
| sft_short (mid-training) | 2000 | 94 | 6-11 | 11% |
| RL pool targets_L6-11 (all arms) | 15738 | 52 | 6-11 | - |
| cap-6 records in the mixes (10,000) | 10000 | 37 | 2-6 | - |
| std natural L6-11 | 1800 | 50 | 6-11 | - |
| std natural L12-14 | 758 | 54 | 12-14 | - |
| std chain L15-18 | 393 | 75 | 15-18 | - |
| std ANDI L15-18 | 807 | 55 | 15-18 | - |
| matched L6-14 | 2700 | 62 | 6-14 | - |
| long L22-26 (glued, 3-part) | 450 | 130 | 22-26 | 100% |

## Figures

![fig_rl.png](fig_rl.png)

![fig_mt.png](fig_mt.png)

![fig_hazard_long.png](fig_hazard_long.png)

