---
written_on: 2026-09-25
---
# ANDI composition diagnosis

Monitor-only analysis for the length-generalisation experiment (`ab3/lengthexp`). Evaluation only, no training. Checkpoints: pretrain, A_s101, B_s101, C_s101 (pretrain and one seed per arm); k = 32 at T = 0.8, max_new 1024. Every number below is in `composition/results.json` (regenerate with `python composition/analyse.py`). Sections 1-5 use the theorem draw of 2026-09-25 00:19 UTC, made from pre-final sealed files (the draws and their evaluations are archived in `testsets_predraw/` and `eval_predraw/`); those theorems are valid, but 37% of them are not in the final sealed files. Section 6 (arm D vs C) uses the final draw.

## Conclusion

**It is a composition failure, not a prompt artefact. From L10 upward it shows first as a refusal to write a proof long enough.**

1. **The parts are solvable; the joined theorem is not.**
   - For the RL arms (one seed each), the two parts alone have a mean pass@1 of 54-56% when the joined length is 6-9, and 41-48% when it is 10-14.
   - If the joined proof were just the two part proofs written one after the other, with independent errors, joined pass@1 would be about the product of the two part rates. That product is 28-31% at joined L6-9 and 16-23% at L10-14.
   - The joined theorem is solved at 4.3% (A), 5.9% (B) and 6.0% (C) at L6-9. That is 15-21% of the product (bootstrap intervals 11-27%). At L10-18 it is 0.00% for every arm, even though the product at L10-14 is 16.1% (A), 23.3% (B) and 22.7% (C).
   - It collapses already at L8. That is inside B's and C's training range. C scores about 19% on the standard L8 set (main report, mean of 3 seeds), but here it gets 0.8% against a product of 26.9%.
   - Pretrain is worse still: 0.3% against a product of 14.2%.
2. **It is not the prompt format.** Changing the prompt, on the same theorems, leaves pass@1 unchanged within about 1 percentage point. Every 95% interval includes zero:
   - (a) premises grouped by part;
   - (b) conjuncts in the order the parts' premises appear;
   - (c) the original atom letters.

   The renamed parts and the original, un-renamed parts are solved at the same rate, within 3 points of the product.
3. **The model will not write a joined proof long enough.**
   - Joined attempts stop at a median of 7 written lines for A at every joined length from 7 to 18, and at 7-9 for B and C (8-9 from L10 on), whatever the required length.
   - As a result, 35-55% of joined attempts are as long as L_true at L6-9, 0.4-5.2% at L10-14 and 0.0% at L15-18.
   - The same checkpoints write the parts alone to their full length: 98-99% of part attempts are long enough for parts of joined L6-9 theorems.
4. **Even when the attempt is long enough, joining fails.**
   - At joined L6-9, attempts that reach L_true are correct 10-13% of the time, against the 28-31% product.
   - The failing line is usually a shortcut: writing a conjunct or the goal with an invalid step, or citing the other part's premises. Across all seeds at L6-9, these account for 29-33%, 17-22% and 22-25% of failures respectively.
   - About three quarters of failures happen before either conjunct has been validly derived. The other part's premises and the conjunction goal disrupt even the first sub-proof.
5. **Premise count makes everything harder but does not explain the gap.**
   - Correction to the brief: the cap-6 pretraining data has 0-3 premises, not 0-1 (41% have 2, 4% have 3). It never has 4 or more, whereas 57% of these joined theorems do.
   - Joined and part pass rates both fall as premises grow.
   - Joined theorems with 3 premises, which are common in RL training, still run at about a quarter of their product: 8-10% against 35-40% at L6-9.
6. **A likely reason.** "Prove A & B from two independent sets of premises" is almost absent from training. Only 0.3% of the RL targets are separable conjunctions (`ab3/data_build/FEASIBILITY.md`). RL on lengths 6-11 hardly changes the joined result: at L6-9, C reaches 6.0% against A's 4.3%, while its parts improve.
7. **Update, arm D (section 6).** Training on constructed composition targets, arm D, largely fixes joining at short
   lengths. At joined L6-9, D reaches 0.80 of the part product, where C reaches 0.14. It does not fix joining at L10 and
   above: the ratio is 0.04 at L10-14. D only rarely solved long constructed training targets, so RL had little to learn
   from there.

**Caveats.**
- Sections 1-4 use one seed per arm (s101). The main evaluation's seed spread was small.
- The product assumes independent errors. It is not a ceiling; the weaker part's rate ("min") is.
- Control (c) is possible for only 18% of theorems (234).
- Failure places are assigned by which part's letters the rejected line uses. This is a heuristic.
- Most L17-18 labels are certified upper bounds (`ub_cert14`), which does not matter at 0%.
- GPU cost: about 23 GPU-minutes in total. The four jobs ran 5-7 minutes each through the lock.

## 1. Joined theorem vs its two parts alone

For each of 100 ANDI theorems per joined length (L6-14 from the matched test draw, L15-18 from the sealed draw; 1,300 in all), both parts were evaluated alone, as they appear in the joined prompt (renamed letters, premises in joined order). "Product" is the mean over theorems of p1 x p2, the rate expected if the joined proof were just the two part proofs written one after the other with independent errors; "min" is the mean of the weaker part, a ceiling. Intervals are 95% theorem-bootstrap. Joined values are the main evaluation's results for the same checkpoints (`eval/<label>/`).

**Joined L6-9.**

| arm | theorems | joined pass@1 | product of parts | min of parts | mean part pass@1 | joined / product | product, original parts | mean original part |
|---|---|---|---|---|---|---|---|---|
| pretrain (max 6) | 400 | 0.27% [0.12%, 0.45%] | 14.25% [11.87%, 16.89%] | 15.6% | 40.9% | 0.02 [0.01, 0.03] | 13.74% | 40.7% |
| A (max 7) | 400 | 4.30% [3.12%, 5.62%] | 28.22% [24.71%, 31.83%] | 29.7% | 54.6% | 0.15 [0.11, 0.20] | 25.48% | 52.7% |
| B (max 9) | 400 | 5.85% [4.20%, 7.59%] | 30.49% [27.00%, 34.14%] | 32.2% | 55.8% | 0.19 [0.14, 0.25] | 29.94% | 56.3% |
| C (max 11) | 400 | 6.02% [4.47%, 7.70%] | 28.88% [25.26%, 32.52%] | 29.9% | 53.8% | 0.21 [0.15, 0.27] | 28.32% | 53.7% |

**Joined L10-14.**

| arm | theorems | joined pass@1 | product of parts | min of parts | mean part pass@1 | joined / product | product, original parts | mean original part |
|---|---|---|---|---|---|---|---|---|
| pretrain (max 6) | 500 | 0.00% [0.00%, 0.00%] | 2.04% [1.28%, 2.96%] | 2.7% | 20.8% | 0.00 [0.00, 0.00] | 2.55% | 21.3% |
| A (max 7) | 500 | 0.00% [0.00%, 0.00%] | 16.07% [13.61%, 18.49%] | 17.6% | 40.8% | 0.00 [0.00, 0.00] | 15.75% | 40.2% |
| B (max 9) | 500 | 0.00% [0.00%, 0.00%] | 23.29% [20.35%, 26.45%] | 24.8% | 47.7% | 0.00 [0.00, 0.00] | 23.32% | 47.8% |
| C (max 11) | 500 | 0.00% [0.00%, 0.00%] | 22.72% [19.90%, 25.65%] | 24.1% | 46.7% | 0.00 [0.00, 0.00] | 22.82% | 47.1% |

**Joined L15-18.**

| arm | theorems | joined pass@1 | product of parts | min of parts | mean part pass@1 | joined / product | product, original parts | mean original part |
|---|---|---|---|---|---|---|---|---|
| pretrain (max 6) | 400 | 0.00% [0.00%, 0.00%] | 0.02% [0.00%, 0.07%] | 0.1% | 6.0% | 0.00 [0.00, 0.00] | 0.00% | 6.0% |
| A (max 7) | 400 | 0.00% [0.00%, 0.00%] | 1.22% [0.44%, 2.15%] | 1.3% | 20.7% | 0.00 [0.00, 0.00] | 1.25% | 22.4% |
| B (max 9) | 400 | 0.00% [0.00%, 0.00%] | 5.97% [3.99%, 8.14%] | 6.2% | 30.7% | 0.00 [0.00, 0.00] | 6.77% | 31.6% |
| C (max 11) | 400 | 0.00% [0.00%, 0.00%] | 9.03% [6.77%, 11.59%] | 9.3% | 32.6% | 0.00 [0.00, 0.00] | 9.05% | 33.7% |

**By joined length** (joined pass@1 / product of the parts):

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | 1.1% / 20.9% | 0.0% / 14.6% | 0.0% / 13.2% | 0.0% / 8.3% | 0.0% / 5.5% | 0.0% / 3.0% | 0.0% / 1.3% | 0.0% / 0.4% | 0.0% / 0.0% | 0.0% / 0.1% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| A | 14.5% / 35.8% | 2.4% / 25.9% | 0.2% / 26.7% | 0.0% / 24.4% | 0.0% / 30.7% | 0.0% / 19.1% | 0.0% / 15.6% | 0.0% / 8.7% | 0.0% / 6.3% | 0.0% / 4.7% | 0.0% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| B | 16.9% / 43.2% | 5.7% / 26.3% | 0.6% / 26.7% | 0.2% / 25.7% | 0.0% / 41.4% | 0.0% / 26.1% | 0.0% / 22.6% | 0.0% / 13.1% | 0.0% / 13.2% | 0.0% / 9.8% | 0.0% / 6.0% | 0.0% / 4.6% | 0.0% / 3.6% |
| C | 18.7% / 43.5% | 4.5% / 24.7% | 0.8% / 26.9% | 0.1% / 20.4% | 0.0% / 30.8% | 0.0% / 26.3% | 0.0% / 22.7% | 0.0% / 17.6% | 0.0% / 16.2% | 0.0% / 10.3% | 0.0% / 7.2% | 0.0% / 8.6% | 0.0% / 10.0% |

## 2. How long the attempts are

Written lines = complete lines in a sample (any sample, correct or not). "Share >= L_true" = fraction of samples with at least as many lines as the theorem's L_true (the joined L for joined theorems and controls, the part's own L for parts).

| arm | joined length | joined attempts: median / p90 / share >= L_true | parts alone: median / p90 / share >= own L_true | control (a): median / p90 / share >= L_true |
|---|---|---|---|---|
| pretrain | L6-9 | 6 / 8 / 35.2% | 5 / 6 / 98.3% | 6 / 8 / 35.6% |
| pretrain | L10-14 | 7 / 8 / 1.5% | 6 / 7 / 67.3% | 7 / 8 / 1.3% |
| pretrain | L15-18 | 7 / 8 / 0.0% | 6 / 7 / 29.2% | 7 / 8 / 0.0% |
| A | L6-9 | 7 / 8 / 42.6% | 5 / 7 / 99.4% | 7 / 8 / 42.2% |
| A | L10-14 | 7 / 8 / 0.4% | 6 / 7 / 73.8% | 7 / 8 / 0.5% |
| A | L15-18 | 7 / 8 / 0.0% | 6 / 7 / 38.4% | 7 / 8 / 0.0% |
| B | L6-9 | 7 / 9 / 54.5% | 5 / 7 / 99.1% | 7 / 9 / 54.3% |
| B | L10-14 | 8 / 10 / 3.6% | 6 / 8 / 79.6% | 8 / 10 / 3.8% |
| B | L15-18 | 9 / 10 / 0.0% | 7 / 9 / 49.6% | 9 / 10 / 0.0% |
| C | L6-9 | 7 / 8 / 47.0% | 5 / 7 / 99.2% | 7 / 8 / 45.7% |
| C | L10-14 | 8 / 11 / 5.2% | 6 / 8 / 81.7% | 8 / 11 / 5.1% |
| C | L15-18 | 9 / 12 / 0.0% | 7 / 9 / 53.9% | 9 / 12 / 0.0% |

Pass rate among attempts that are long enough (written lines >= L_true), pooled over samples:

| arm | joined length | joined attempts long enough | joined pass@1 among those | product of parts (section 1) | part attempts long enough | part pass@1 among those |
|---|---|---|---|---|---|---|
| pretrain | L6-9 | 35.2% | 0.8% | 14.2% | 98.3% | 41.6% |
| pretrain | L10-14 | 1.5% | 0.0% | 2.0% | 67.3% | 30.9% |
| pretrain | L15-18 | 0.0% | 0.0% | 0.0% | 29.2% | 20.5% |
| A | L6-9 | 42.6% | 10.1% | 28.2% | 99.4% | 54.9% |
| A | L10-14 | 0.4% | 0.0% | 16.1% | 73.8% | 55.2% |
| A | L15-18 | 0.0% | 0.0% | 1.2% | 38.4% | 53.8% |
| B | L6-9 | 54.5% | 10.7% | 30.5% | 99.1% | 56.2% |
| B | L10-14 | 3.6% | 0.0% | 23.3% | 79.6% | 59.9% |
| B | L15-18 | 0.0% | 0.0% | 6.0% | 49.6% | 62.0% |
| C | L6-9 | 47.0% | 12.8% | 28.9% | 99.2% | 54.2% |
| C | L10-14 | 5.2% | 0.0% | 22.7% | 81.7% | 57.2% |
| C | L15-18 | 0.0% | 0.0% | 9.0% | 53.9% | 60.4% |

Median written lines of joined attempts by joined length:

| arm | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| A | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| B | 6 | 7 | 7 | 8 | 8 | 9 | 9 | 8 | 8 | 9 | 9 | 9 | 9 |
| C | 6 | 7 | 7 | 7 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 9 |

## 3. Prompt controls on the same theorems

(a) premises grouped by part, the left conjunct's part first; (b) conjuncts ordered as the parts' premises first appear; (c) both parts in their original letters (possible only where the two original parts used disjoint letters: 18% of the theorems). A control that coincides with the joined prompt counts with the joined result. Difference = control minus joined, paired by theorem. At joined L10-18 every control, like the joined theorem, is 0.00% for every arm (`results.json` step2, bands 10-14 and 15-18).

| arm | control | joined length | theorems | joined pass@1 | control pass@1 | difference [95% CI] |
|---|---|---|---|---|---|---|
| pretrain | a | L6-9 | 400 | 0.27% | 0.23% | -0.04 pts [-0.22%, 0.13%] |
| pretrain | a | L6-18 | 1300 | 0.08% | 0.07% | -0.01 pts [-0.07%, 0.04%] |
| pretrain | b | L6-9 | 400 | 0.27% | 0.29% | +0.02 pts [-0.14%, 0.18%] |
| pretrain | b | L6-18 | 1300 | 0.08% | 0.09% | +0.01 pts [-0.05%, 0.06%] |
| pretrain | c | L6-9 | 80 | 0.31% | 0.43% | +0.12 pts [-0.47%, 0.86%] |
| pretrain | c | L6-18 | 234 | 0.11% | 0.15% | +0.04 pts [-0.16%, 0.29%] |
| A | a | L6-9 | 400 | 4.30% | 4.77% | +0.47 pts [-0.79%, 1.82%] |
| A | a | L6-18 | 1300 | 1.32% | 1.47% | +0.14 pts [-0.24%, 0.54%] |
| A | b | L6-9 | 400 | 4.30% | 5.09% | +0.79 pts [-0.03%, 1.63%] |
| A | b | L6-18 | 1300 | 1.32% | 1.56% | +0.24 pts [-0.02%, 0.51%] |
| A | c | L6-9 | 80 | 3.55% | 4.02% | +0.47 pts [-1.49%, 2.62%] |
| A | c | L6-18 | 234 | 1.22% | 1.38% | +0.16 pts [-0.49%, 0.89%] |
| B | a | L6-9 | 400 | 5.85% | 6.05% | +0.20 pts [-1.05%, 1.40%] |
| B | a | L6-18 | 1300 | 1.80% | 1.86% | +0.06 pts [-0.31%, 0.46%] |
| B | b | L6-9 | 400 | 5.85% | 5.44% | -0.41 pts [-1.66%, 0.88%] |
| B | b | L6-18 | 1300 | 1.80% | 1.67% | -0.13 pts [-0.51%, 0.25%] |
| B | c | L6-9 | 80 | 6.13% | 6.37% | +0.23 pts [-1.13%, 1.76%] |
| B | c | L6-18 | 234 | 2.10% | 2.18% | +0.08 pts [-0.43%, 0.59%] |
| C | a | L6-9 | 400 | 6.02% | 4.96% | -1.06 pts [-2.49%, 0.24%] |
| C | a | L6-18 | 1300 | 1.85% | 1.53% | -0.32 pts [-0.74%, 0.10%] |
| C | b | L6-9 | 400 | 6.02% | 5.68% | -0.34 pts [-1.28%, 0.59%] |
| C | b | L6-18 | 1300 | 1.85% | 1.75% | -0.11 pts [-0.41%, 0.17%] |
| C | c | L6-9 | 80 | 7.30% | 8.40% | +1.09 pts [-0.94%, 3.40%] |
| C | c | L6-18 | 234 | 2.50% | 2.87% | +0.37 pts [-0.39%, 1.19%] |

## 4. Premise count and atom count

Training data, for comparison: `train_cap6` (pretraining) has 0 / 1 / 2 / 3 premises in 16.3% / 38.3% / 41.0% / 4.4% of records and never 4 or more; the RL pool `targets_L6-11` has 0-5 premises (4: 6.6%, 5: 0.5%). The joined ANDI theorems here have 2 / 3 / 4 / 5 premises in 5.8% / 37.5% / 47.1% / 9.6%; their parts have 1-3. Atoms: 90% of joined theorems have 4 atoms, as do 47% of RL targets and 18% of cap-6 records.

**pretrain, joined L6-9, by the joined theorem's premises / atoms.**

| stratum | theorems | joined | product of parts | control (a) |
|---|---|---|---|---|
| prem = 2 | 30 | 0.00% | 46.55% | 0.00% |
| prem = 3 | 189 | 0.56% | 16.72% | 0.45% |
| prem = 4 | 153 | 0.00% | 7.22% | 0.04% |
| prem = 5 | 28 | 0.00% | 1.41% | 0.00% |
| atoms = 3 | 59 | 0.16% | 17.78% | 0.11% |
| atoms = 4 | 341 | 0.28% | 13.64% | 0.25% |

**A, joined L6-9, by the joined theorem's premises / atoms.**

| stratum | theorems | joined | product of parts | control (a) |
|---|---|---|---|---|
| prem = 2 | 30 | 0.62% | 63.99% | 1.46% |
| prem = 3 | 189 | 8.18% | 34.69% | 7.77% |
| prem = 4 | 153 | 1.00% | 17.69% | 2.37% |
| prem = 5 | 28 | 0.00% | 3.77% | 1.12% |
| atoms = 3 | 59 | 3.76% | 38.26% | 1.80% |
| atoms = 4 | 341 | 4.39% | 26.48% | 5.28% |

**B, joined L6-9, by the joined theorem's premises / atoms.**

| stratum | theorems | joined | product of parts | control (a) |
|---|---|---|---|---|
| prem = 2 | 30 | 0.10% | 60.00% | 0.31% |
| prem = 3 | 189 | 9.26% | 39.82% | 9.85% |
| prem = 4 | 153 | 3.78% | 17.21% | 3.55% |
| prem = 5 | 28 | 0.33% | 8.44% | 0.22% |
| atoms = 3 | 59 | 6.36% | 37.47% | 4.13% |
| atoms = 4 | 341 | 5.76% | 29.28% | 6.39% |

**C, joined L6-9, by the joined theorem's premises / atoms.**

| stratum | theorems | joined | product of parts | control (a) |
|---|---|---|---|---|
| prem = 2 | 30 | 1.88% | 62.73% | 0.83% |
| prem = 3 | 189 | 10.25% | 38.07% | 7.84% |
| prem = 4 | 153 | 2.57% | 14.57% | 2.74% |
| prem = 5 | 28 | 0.78% | 8.76% | 2.12% |
| atoms = 3 | 59 | 4.18% | 34.12% | 2.75% |
| atoms = 4 | 341 | 6.34% | 27.97% | 5.34% |

**Parts alone (renamed), pass@1 by the part's own premises / atoms, within part-length bands:**

| arm | part length | stratum | pass@1 by value |
|---|---|---|---|
| pretrain | L2-4 | prem | 1: 72.3% (n=241), 2: 24.6% (n=656), 3: 10.2% (n=101) |
| pretrain | L5-7 | prem | 1: 22.6% (n=126), 2: 15.2% (n=591), 3: 6.8% (n=61), 4: 0.0% (n=1) |
| pretrain | L8-10 | prem | 1: 0.0% (n=187), 2: 0.0% (n=445), 3: 0.0% (n=25) |
| pretrain | L2-4 | atoms | 1: 39.9% (n=64), 2: 34.3% (n=934) |
| pretrain | L5-7 | atoms | 1: 30.1% (n=28), 2: 15.2% (n=751) |
| pretrain | L8-10 | atoms | 1: 0.0% (n=14), 2: 0.0% (n=643) |
| A | L2-4 | prem | 1: 80.2% (n=241), 2: 37.8% (n=656), 3: 30.2% (n=101) |
| A | L5-7 | prem | 1: 54.3% (n=126), 2: 51.9% (n=591), 3: 41.8% (n=61), 4: 9.4% (n=1) |
| A | L8-10 | prem | 1: 5.6% (n=187), 2: 0.3% (n=445), 3: 2.1% (n=25) |
| A | L2-4 | atoms | 1: 56.6% (n=64), 2: 46.6% (n=934) |
| A | L5-7 | atoms | 1: 68.4% (n=28), 2: 50.8% (n=751) |
| A | L8-10 | atoms | 1: 6.9% (n=14), 2: 1.8% (n=643) |
| B | L2-4 | prem | 1: 80.6% (n=241), 2: 39.4% (n=656), 3: 31.6% (n=101) |
| B | L5-7 | prem | 1: 58.6% (n=126), 2: 55.7% (n=591), 3: 50.9% (n=61), 4: 15.6% (n=1) |
| B | L8-10 | prem | 1: 20.9% (n=187), 2: 18.0% (n=445), 3: 17.6% (n=25) |
| B | L2-4 | atoms | 1: 55.4% (n=64), 2: 48.1% (n=934) |
| B | L5-7 | atoms | 1: 79.1% (n=28), 2: 54.9% (n=751) |
| B | L8-10 | atoms | 1: 41.7% (n=14), 2: 18.3% (n=643) |
| C | L2-4 | prem | 1: 79.0% (n=241), 2: 38.3% (n=656), 3: 29.1% (n=101) |
| C | L5-7 | prem | 1: 58.2% (n=126), 2: 52.2% (n=591), 3: 38.4% (n=61), 4: 6.2% (n=1) |
| C | L8-10 | prem | 1: 32.1% (n=187), 2: 21.2% (n=445), 3: 19.5% (n=25) |
| C | L2-4 | atoms | 1: 53.7% (n=64), 2: 46.7% (n=934) |
| C | L5-7 | atoms | 1: 82.0% (n=28), 2: 50.9% (n=751) |
| C | L8-10 | atoms | 1: 50.2% (n=14), 2: 23.7% (n=643) |

## 5. Where failed joined samples go wrong

All ANDI theorems of the matched draw (L6-14) and the sealed draw (L15-18), all three seeds per arm, from the saved samples (`composition/failloc.py`). The first rejected line (incremental checker, same verdicts as nd_verify) is placed relative to the two parts by the atoms it uses: fake goal = writes the joined conclusion with an invalid step; fake conjunct = writes one conjunct with an invalid step; cross-part citation = a line in one part's letters citing a premise or line of the other part; inside left/right part = an invalid step within one part; mixes parts = a formula combining both parts' letters; premise block = the premise lines are wrong. "Progress" = which conjuncts had been validly derived before the error.

| arm | joined length | theorems | pass@1 | fake goal | fake conjunct | cross-part citation | inside left part | inside right part | mixes parts | premise block | other (early QED, out of tokens, unattributed) | progress: no conjunct yet | progress: one conjunct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrain | L6-9 | 603 | 0.29% | 8% | 37% | 23% | 4% | 4% | 5% | 19% | 0% | 88% | 11% |
| pretrain | L10-14 | 753 | 0.00% | 3% | 29% | 25% | 4% | 5% | 5% | 29% | 0% | 97% | 3% |
| pretrain | L15-18 | 865 | 0.00% | 1% | 23% | 25% | 5% | 5% | 7% | 34% | 0% | 100% | 0% |
| A | L6-9 | 603 | 4.71% | 17% | 29% | 25% | 8% | 10% | 5% | 6% | 0% | 75% | 24% |
| A | L10-14 | 753 | 0.00% | 7% | 22% | 32% | 14% | 11% | 3% | 11% | 0% | 91% | 9% |
| A | L15-18 | 865 | 0.00% | 2% | 17% | 36% | 17% | 12% | 4% | 12% | 0% | 99% | 1% |
| B | L6-9 | 603 | 4.98% | 19% | 32% | 24% | 7% | 9% | 5% | 3% | 0% | 73% | 25% |
| B | L10-14 | 753 | 0.00% | 10% | 20% | 33% | 13% | 14% | 3% | 6% | 0% | 90% | 10% |
| B | L15-18 | 865 | 0.00% | 5% | 12% | 41% | 16% | 18% | 3% | 6% | 0% | 98% | 2% |
| C | L6-9 | 603 | 5.08% | 22% | 33% | 22% | 6% | 8% | 4% | 4% | 0% | 73% | 26% |
| C | L10-14 | 753 | 0.00% | 12% | 24% | 30% | 10% | 14% | 4% | 6% | 0% | 91% | 9% |
| C | L15-18 | 865 | 0.00% | 5% | 16% | 37% | 10% | 22% | 4% | 6% | 0% | 99% | 1% |

**Five short example failures** (the offending line is marked `<<`):

1. `C_s101`, `ab3_matched_L6-14_454` (L_true 6; left conjunct `( ( Q v Q ) v ( Q > Q ) )`, right conjunct `( R & R )`): Never proves the left conjunct. Writes the right conjunct ( R & R ) twice, then "proves" the goal by ANDI with a premise of the other part, ( ( ~ ( R & S ) ) v S ), in place of the left conjunct.

```
THM ( ( ~ ( R & S ) ) v S ) , ( Q v Q ) , R SEQ ( ( ( Q v Q ) v ( Q > Q ) ) & ( R & R ) ) PRF
  N35 ( ( ~ ( R & S ) ) v S ) : PR ;
  N36 ( Q v Q ) : PR ;
  N37 R : PR ;
  N38 ( R & R ) : ANDI N37 N37 ;
  N39 ( R & R ) : ANDI N37 N37 ;
  N40 ( ( ( Q v Q ) v ( Q > Q ) ) & ( R & R ) ) : ANDI N35 N39 ;   << rule
  QED
```

2. `A_s101`, `ab3_matched_L6-14_375` (L_true 6; left conjunct `( ( S v R ) v ( ( ~ S ) > R ) )`, right conjunct `( P & P )`): Proves the left conjunct correctly (line 4), then writes the goal at once, citing the premise ( S v R ) in place of the right conjunct ( P & P ): it never starts the second part.

```
THM ( S v R ) , ( P v ( Q v ( Q & Q ) ) ) , P SEQ ( ( ( S v R ) v ( ( ~ S ) > R ) ) & ( P & P ) ) PRF
  N12 ( S v R ) : PR ;
  N13 ( P v ( Q v ( Q & Q ) ) ) : PR ;
  N14 P : PR ;
  N15 ( ( S v R ) v ( ( ~ S ) > R ) ) : ORI1 N12 ;
  N16 ( ( ( S v R ) v ( ( ~ S ) > R ) ) & ( P & P ) ) : ANDI N15 N12 ;   << rule
  QED
```

3. `A_s101`, `ab3_matched_L6-14_102` (L_true 7; left conjunct `( S > S )`, right conjunct `( ( ( R v ( ~ Q ) ) & ( R & R ) ) v R )`): Works on the right part but builds ( ( R v ( ~ Q ) ) & ( R & R ) ) by ANDI citing ( ~ P ), a premise of the other part, in place of ( R & R ).

```
THM ( R v ( ~ Q ) ) , ( ( ~ S ) & ( S > S ) ) , R , ( ~ P ) SEQ ( ( S > S ) & ( ( ( R v ( ~ Q ) ) & ( R & R ) ) v R ) ) PRF
  N6 ( R v ( ~ Q ) ) : PR ;
  N7 ( ( ~ S ) & ( S > S ) ) : PR ;
  N8 R : PR ;
  N9 ( ~ P ) : PR ;
  N10 ( ( R v ( ~ Q ) ) & ( R & R ) ) : ANDI N6 N9 ;   << rule
  N11 ( ( ( R v ( ~ Q ) ) & ( R & R ) ) v R ) : ORI1 N10 ;
  QED
```

4. `C_s101`, `ab3_matched_L6-14_2085` (L_true 8; left conjunct `( ( ~ Q ) & ( ~ Q ) )`, right conjunct `( S & S )`): Writes the right conjunct ( S & S ) in one invalid step: IMPE from ( S > ( R > S ) ) and ( ~ Q ), a premise of the other part. Then it writes the goal.

```
THM ( S > ( R > S ) ) , S , ( S v ( ~ R ) ) , ( ~ Q ) , ( Q v P ) SEQ ( ( ( ~ Q ) & ( ~ Q ) ) & ( S & S ) ) PRF
  N7 ( S > ( R > S ) ) : PR ;
  N8 S : PR ;
  N9 ( S v ( ~ R ) ) : PR ;
  N10 ( ~ Q ) : PR ;
  N11 ( Q v P ) : PR ;
  N12 ( S & S ) : IMPE N7 N10 ;   << rule
  N13 ( ( ( ~ Q ) & ( ~ Q ) ) & ( S & S ) ) : ANDI N9 N12 ;
  QED
```

5. `A_s101`, `ab3_matched_L6-14_360` (L_true 8; left conjunct `Q`, right conjunct `( R v S )`): Copies the premise block wrongly: after three correct premises it skips the fourth, ( ( Q v Q ) & P ), and writes R, then a fragment ( Q v Q ) as premises.

```
THM ( ~ ( ~ Q ) ) , ( ( S > S ) > S ) , ( S > S ) , ( ( Q v Q ) & P ) , R SEQ ( Q & ( R v S ) ) PRF
  N8 ( ~ ( ~ Q ) ) : PR ;
  N9 ( ( S > S ) > S ) : PR ;
  N10 ( S > S ) : PR ;
  N11 R : PR ;   << premise
  N12 ( Q v Q ) : PR ;
  N13 S : IMPE N9 N10 ;
  N14 ( R v S ) : ORI1 N13 ;
  N15 ( Q & ( R v S ) ) : ANDI N11 N14 ;
  QED
```

## 6. Arm D (composition-augmented RL pool) vs arm C, final test draw

Arm D = arm C except the RL pool: run 2's targets plus constructed ANDI/chain targets, 30% of each length bin (`ab3/data_comp/`, `ab3/frozen_comp`). The test theorems are part-disjoint from D's targets (0 shared reserve parts among 6,211 test parts and 9,720 training parts). This section uses the **final** matched draw (100 ANDI theorems per length at L6-14 in `composition/final/joined.jsonl`); sections 1-5 used the earlier draw. Parts alone are the renamed parts, k = 32.

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

**Joined vs product of the parts (one seed each).**

| checkpoint | joined length | theorems | joined pass@1 | product of parts | mean part pass@1 | joined / product |
|---|---|---|---|---|---|---|
| C_s101 | L6-9 | 400 | 3.97% [2.76%, 5.22%] | 27.45% [24.03%, 30.98%] | 53.0% | 0.14 [0.10, 0.20] |
| D_s101 | L6-9 | 400 | 27.31% [23.84%, 31.03%] | 33.97% [30.46%, 37.81%] | 58.8% | 0.80 [0.67, 0.94] |
| C_s101 | L10-14 | 500 | 0.00% [0.00%, 0.00%] | 20.52% [17.62%, 23.33%] | 45.4% | 0.00 [0.00, 0.00] |
| D_s101 | L10-14 | 500 | 1.09% [0.51%, 1.77%] | 26.20% [23.04%, 29.37%] | 50.4% | 0.04 [0.02, 0.07] |
| C_s101 | L6-14 | 900 | 1.76% [1.23%, 2.36%] | 23.60% [21.39%, 25.82%] | 48.8% | 0.07 [0.05, 0.10] |
| D_s101 | L6-14 | 900 | 12.74% [10.78%, 14.64%] | 29.66% [27.30%, 32.00%] | 54.1% | 0.43 [0.36, 0.51] |

By joined length (joined / product):

| checkpoint | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| C_s101 | 11.0% / 40.2% | 4.0% / 26.1% | 0.6% / 23.8% | 0.2% / 19.7% | 0.0% / 29.4% | 0.0% / 20.6% | 0.0% / 26.7% | 0.0% / 15.0% | 0.0% / 10.9% |
| D_s101 | 40.7% / 40.2% | 38.5% / 31.2% | 19.5% / 35.7% | 10.6% / 28.7% | 3.8% / 34.1% | 1.5% / 28.8% | 0.2% / 33.1% | 0.0% / 19.7% | 0.0% / 15.3% |

**Joined attempts: written lines** (median / p90 / share >= L_true):

| checkpoint | joined length | median | p90 | share >= L_true |
|---|---|---|---|---|
| C_s101 | L6-9 | 7 | 8 | 46.9% |
| C_s101 | L10-14 | 8 | 11 | 6.9% |
| D_s101 | L6-9 | 7 | 9 | 69.5% |
| D_s101 | L10-14 | 9 | 11 | 13.2% |

**Where failed joined samples go wrong** (all ANDI theorems of the final matched draw, all seeds):

| arm | joined length | checkpoints | pass@1 | fake goal | fake conjunct | cross-part citation | inside left part | inside right part | mixes parts | premise block | no conjunct derived before the error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | L6-9 | C_s101, C_s102, C_s103 | 4.14% | 21% | 33% | 23% | 6% | 8% | 5% | 4% | 74% |
| C | L10-14 | C_s101, C_s102, C_s103 | 0.00% | 12% | 23% | 30% | 10% | 14% | 4% | 6% | 92% |
| D | L6-9 | D_s101, D_s102, D_s103 | 26.50% | 12% | 53% | 18% | 5% | 7% | 4% | 1% | 60% |
| D | L10-14 | D_s101, D_s102, D_s103 | 1.17% | 8% | 34% | 30% | 13% | 12% | 2% | 1% | 74% |

## Files

- `composition/build.py` → `composition/joined.jsonl`, `composition/prompts.jsonl` (selection seed 20260925; every part and control proof checked with nd_verify).
- `composition/run_eval.py` → `composition/eval/<label>/chunk_*.jsonl` (+ samples), logs in `composition/logs/`.
- `composition/failloc.py` → `composition/joined_samples.jsonl`, `composition/failloc_examples.json`.
- `composition/analyse.py` → `composition/results.json` and this file.
- Section 6: `composition/build.py --dir composition/final --lmax 14 --roles part_ren`, `composition/run_eval.py --dir composition/final`, `composition/failloc.py --dir composition/final --labels C_s101,C_s102,C_s103,D_s101,D_s102,D_s103`.

