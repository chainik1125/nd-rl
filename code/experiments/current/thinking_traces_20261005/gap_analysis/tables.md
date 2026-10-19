## Groups

TB = textbook dev58 (grouped by its reference length, as in the eval). Score = mean pass@1 of the four best RL'd checkpoints. TB fail = 7–12 lines and score < 25%; h12 test fail = score < 50%. pool / K12 'gen 7–12' are grouped by the generator proof's length, 'min 7–12' by the shortest proof found. K12 = a random 20,000 of train_k12 (all lengths; shortest proofs only for its 7–12-line part). `*` = some rows lack the feature (no proof found: textbook problems over 16 lines).


### Theorem shape (binary, % of group)

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | TB all (n=58) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 (n=20000) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|---|---|---|
| no premises (a theorem) | 38% (3/8) | 25% (3/12) | 30% (6/20) | 22% (13/58) | 3% | 4% | 0% (0/28) | 6% | 1% | 1% |
| goal is an atom | 25% (2/8) | 17% (2/12) | 20% (4/20) | 16% (9/58) | 6% | 2% | 0% (0/28) | 5% | 7% | 4% |
| goal is a negation | 25% (2/8) | 17% (2/12) | 20% (4/20) | 14% (8/58) | 10% | 9% | 29% (8/28) | 5% | 5% | 4% |
| goal is a negated compound ~(A∘B) | 25% (2/8) | 8% (1/12) | 15% (3/20) | 7% (4/58) | 8% | 8% | 25% (7/28) | 3% | 3% | 3% |
| goal is a disjunction | 25% (2/8) | 0% (0/12) | 10% (2/20) | 17% (10/58) | 9% | 3% | 7% (2/28) | 24% | 18% | 10% |
| goal is a conjunction | 0% (0/8) | 8% (1/12) | 5% (1/20) | 21% (12/58) | 19% | 14% | 4% (1/28) | 21% | 24% | 23% |
| goal is an implication | 25% (2/8) | 58% (7/12) | 45% (9/20) | 28% (16/58) | 55% | 71% | 61% (17/28) | 43% | 44% | 56% |
| goal is A>B with compound A | 12% (1/8) | 50% (6/12) | 35% (7/20) | 17% (10/58) | 44% | 56% | 54% (15/28) | 33% | 34% | 44% |
| goal is (A>B)>C (higher-order) | 0% (0/8) | 25% (3/12) | 15% (3/20) | 9% (5/58) | 4% | 5% | 0% (0/28) | 4% | 3% | 4% |
| some (A>B)>C anywhere | 0% (0/8) | 25% (3/12) | 15% (3/20) | 12% (7/58) | 20% | 23% | 29% (8/28) | 15% | 14% | 19% |
| some negated compound ~(A&B) / ~(AvB) / ~(A>B) | 50% (4/8) | 17% (2/12) | 30% (6/20) | 33% (19/58) | 49% | 46% | 61% (17/28) | 37% | 38% | 35% |
| some double negation ~~A | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (4/58) | 43% | 51% | 61% (17/28) | 26% | 30% | 37% |
| some biconditional (A>B)&(B>A) | 25% (2/8) | 0% (0/12) | 10% (2/20) | 28% (16/58) | 0% | 1% | 4% (1/28) | 0% | 0% | 0% |
| classical-only (not intuitionistically valid, G4ip) | 25% (2/8) | 8% (1/12) | 15% (3/20) | 26% (15/58) | 4% | 5% | 4% (1/28) | 4% | 5% | 8% |
| premises jointly inconsistent | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (4/58) | 33% | 16% | 32% (9/28) | 24% | 29% | 12% |
| F appears in the theorem | 0% (0/8) | 8% (1/12) | 5% (1/20) | 7% (4/58) | 11% | 12% | 11% (3/28) | 24% | 25% | 26% |

### Theorem shape (numeric, group mean)

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | TB all (n=58) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 (n=20000) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|---|---|---|
| premises | 0.88 | 1.25 | 1.10 | 1.24 | 2.24 | 2.17 | 2.29 | 1.82 | 2.16 | 2.15 |
| distinct atoms | 2.38 | 2.50 | 2.45 | 2.59 | 3.12 | 3.22 | 3.07 | 2.92 | 2.96 | 3.04 |
| formula depth (max) | 3.00 | 2.08 | 2.45 | 2.74 | 3.96 | 4.15 | 4.11 | 3.54 | 3.47 | 3.61 |
| goal depth | 2.25 | 1.67 | 1.90 | 1.86 | 3.40 | 3.78 | 3.57 | 2.89 | 2.75 | 3.05 |
| connectives in the theorem | 5.75 | 4.33 | 4.90 | 6.12 | 11.4 | 11.7 | 12.5 | 9.10 | 9.36 | 10.5 |
| ~ count | 1.75 | 1.08 | 1.35 | 1.62 | 3.35 | 3.29 | 4.46 | 2.38 | 2.54 | 2.69 |
| & count | 1.38 | 0.33 | 0.75 | 1.34 | 2.61 | 2.72 | 2.86 | 1.87 | 1.85 | 2.11 |
| v count | 1.12 | 0.83 | 0.95 | 1.07 | 2.23 | 1.97 | 2.57 | 2.32 | 2.48 | 2.47 |
| > count | 1.50 | 2.08 | 1.85 | 2.09 | 3.20 | 3.73 | 2.61 | 2.53 | 2.48 | 3.23 |
| nested-implication depth | 0.75 | 1.25 | 1.05 | 0.97 | 1.76 | 2.05 | 1.39 | 1.44 | 1.35 | 1.66 |
| theorem tokens | 23.4 | 19.0 | 20.8 | 25.6 | 46.7 | 47.9 | 50.1 | 37.7 | 39.2 | 43.6 |
| redundant premises | 0.00 | 0.00 | 0.00 | 0.03 | 1.19 | 1.06 | 1.07 | 0.82 | 1.12 | 0.86 |

### Shortest-proof structure (binary, % of group)

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | TB all (n=58) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 (n=20000) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|---|---|---|
| shortest proof >= 9 lines | 100% (8/8) | 50% (6/12) | 70% (14/20) | 57% (26/46)* | 8% | 18% | 39% (11/28) | – | 4% | 16% |
| <= 7 connectives and shortest proof >= 9 lines | 100% (8/8) | 50% (6/12) | 70% (14/20) | 48% (22/46)* | 1% | 2% | 0% (0/28) | – | 1% | 4% |
| reductio, or a ¬-intro box inside another box | 75% (6/8) | 8% (1/12) | 35% (7/20) | 28% (13/46)* | 2% | 4% | 4% (1/28) | – | 3% | 9% |
| no box at all | 0% (0/8) | 0% (0/12) | 0% (0/20) | 11% (5/46)* | 44% | 13% | 25% (7/28) | – | 51% | 7% |
| box depth >= 2 | 88% (7/8) | 67% (8/12) | 75% (15/20) | 57% (26/46)* | 17% | 36% | 29% (8/28) | – | 9% | 38% |
| box depth >= 3 | 0% (0/8) | 17% (2/12) | 10% (2/20) | 9% (4/46)* | 4% | 9% | 7% (2/28) | – | 0% | 1% |
| >= 3 assumptions (AS lines) | 25% (2/8) | 58% (7/12) | 45% (9/20) | 37% (17/46)* | 5% | 12% | 21% (6/28) | – | 4% | 19% |
| case split (ORE) | 38% (3/8) | 50% (6/12) | 45% (9/20) | 35% (16/46)* | 3% | 7% | 18% (5/28) | – | 10% | 40% |
| reductio (assume ~G, F, NEGI, DN) | 38% (3/8) | 8% (1/12) | 20% (4/20) | 17% (8/46)* | 1% | 2% | 0% (0/28) | – | 2% | 8% |
| plain ¬-intro (NEGI box, no DN) | 75% (6/8) | 25% (3/12) | 45% (9/20) | 35% (16/46)* | 4% | 7% | 25% (7/28) | – | 2% | 4% |
| →-intro box | 25% (2/8) | 58% (7/12) | 45% (9/20) | 41% (19/46)* | 50% | 77% | 50% (14/28) | – | 40% | 62% |
| ex falso | 0% (0/8) | 8% (1/12) | 5% (1/20) | 11% (5/46)* | 42% | 34% | 25% (7/28) | – | 28% | 18% |
| a negation/reductio box nested inside another box | 62% (5/8) | 0% (0/12) | 25% (5/20) | 20% (9/46)* | 1% | 2% | 4% (1/28) | – | 1% | 3% |
| negation box inside a negation box | 38% (3/8) | 0% (0/12) | 15% (3/20) | 11% (5/46)* | 0% | 0% | 0% (0/28) | – | 0% | 0% |
| negation box inside a →-intro box | 25% (2/8) | 0% (0/12) | 10% (2/20) | 7% (3/46)* | 1% | 2% | 4% (1/28) | – | 1% | 3% |
| case split inside a negation/reductio box | 25% (2/8) | 17% (2/12) | 20% (4/20) | 11% (5/46)* | 0% | 1% | 0% (0/28) | – | 0% | 0% |
| negation box inside a case branch | 0% (0/8) | 0% (0/12) | 0% (0/20) | 2% (1/46)* | 0% | 0% | 0% (0/28) | – | 0% | 0% |
| any box inside a case branch | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (3/46)* | 0% | 0% | 0% (0/28) | – | 0% | 1% |
| →-intro box inside a →-intro box | 0% (0/8) | 25% (3/12) | 15% (3/20) | 15% (7/46)* | 15% | 32% | 18% (5/28) | – | 4% | 17% |
| case split on an assumption (not a premise) | 12% (1/8) | 17% (2/12) | 15% (3/20) | 9% (4/46)* | 0% | 0% | 4% (1/28) | – | 0% | 2% |
| case split concluding F | 12% (1/8) | 25% (3/12) | 20% (4/20) | 11% (5/46)* | 0% | 0% | 4% (1/28) | – | 0% | 2% |
| case split concluding a compound | 12% (1/8) | 17% (2/12) | 15% (3/20) | 17% (8/46)* | 1% | 3% | 7% (2/28) | – | 5% | 22% |
| case branches end in Or.inl vs Or.inr | 12% (1/8) | 17% (2/12) | 15% (3/20) | 13% (6/46)* | 0% | 0% | 0% (0/28) | – | 0% | 1% |
| both case branches >= 2 lines | 25% (2/8) | 42% (5/12) | 35% (7/20) | 26% (12/46)* | 1% | 2% | 18% (5/28) | – | 1% | 6% |
| contradiction built by an intro (X built to clash with ~X) | 12% (1/8) | 8% (1/12) | 10% (2/20) | 20% (9/46)* | 16% | 13% | 29% (8/28) | – | 5% | 6% |
| contradiction uses an assumed negation | 38% (3/8) | 17% (2/12) | 25% (5/20) | 24% (11/46)* | 11% | 16% | 14% (4/28) | – | 3% | 6% |
| a ¬-intro result reused by IMPE/NEGE (negation lemma) | 25% (2/8) | 0% (0/12) | 10% (2/20) | 9% (4/46)* | 0% | 1% | 0% (0/28) | – | 0% | 1% |
| a →-intro result applied later (implication lemma) | 0% (0/8) | 8% (1/12) | 5% (1/20) | 7% (3/46)* | 5% | 7% | 4% (1/28) | – | 2% | 5% |
| any box result reused by an elimination | 38% (3/8) | 8% (1/12) | 20% (4/20) | 22% (10/46)* | 7% | 11% | 7% (2/28) | – | 6% | 20% |
| box results combined by ANDI/ORI | 12% (1/8) | 17% (2/12) | 15% (3/20) | 22% (10/46)* | 6% | 11% | 14% (4/28) | – | 7% | 20% |
| IMPE on an assumed implication | 0% (0/8) | 25% (3/12) | 15% (3/20) | 11% (5/46)* | 1% | 2% | 0% (0/28) | – | 0% | 1% |
| first step is forward (not an assumption) | 12% (1/8) | 0% (0/12) | 5% (1/20) | 17% (8/46)* | 46% | 17% | 25% (7/28) | – | 52% | 12% |

### Shortest-proof structure (numeric, group mean)

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | TB all (n=58) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 (n=20000) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|---|---|---|
| shortest proof found (lines) | 10.4 | 8.75 | 9.40 | 9.04* | 6.36 | 7.78 | 7.93 | – | 5.27 | 7.69 |
| box depth | 1.88 | 1.83 | 1.85 | 1.54* | 0.76 | 1.33 | 1.14 | – | 0.59 | 1.33 |
| assumptions (AS lines) | 2.25 | 2.42 | 2.35 | 2.04* | 0.80 | 1.40 | 1.36 | – | 0.69 | 1.74 |
| elimination lines at depth >= 2 | 3.00 | 1.25 | 1.95 | 1.48* | 0.26 | 0.57 | 0.43 | – | 0.08 | 0.35 |
| lines at depth >= 2 | 4.38 | 3.17 | 3.65 | 2.85* | 0.58 | 1.28 | 1.32 | – | 0.25 | 1.07 |
| longest box (lines) | 7.50 | 5.92 | 6.55 | 5.59* | 2.13 | 3.82 | 3.82 | – | 1.42 | 3.48 |
| state-env actions of that proof | 11.4 | 9.75 | 10.4 | 10.0* | 7.36 | 8.78 | 8.93 | – | 6.27 | 8.69 |

### Generator's own proof structure for generator columns (textbook columns: shortest proof)

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | TB all (n=58) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 (n=20000) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|---|---|---|
| shortest proof >= 9 lines | 100% (8/8) | 50% (6/12) | 70% (14/20) | 57% (26/46)* | 69% | 60% | 82% (23/28) | 37% | 67% | 57% |
| <= 7 connectives and shortest proof >= 9 lines | 100% (8/8) | 50% (6/12) | 70% (14/20) | 48% (22/46)* | 12% | 7% | 0% (0/28) | 13% | 23% | 14% |
| reductio, or a ¬-intro box inside another box | 75% (6/8) | 8% (1/12) | 35% (7/20) | 28% (13/46)* | 8% | 9% | 14% (4/28) | 13% | 17% | 16% |
| no box at all | 0% (0/8) | 0% (0/12) | 0% (0/20) | 11% (5/46)* | 3% | 4% | 0% (0/28) | 24% | 2% | 4% |
| box depth >= 2 | 88% (7/8) | 67% (8/12) | 75% (15/20) | 57% (26/46)* | 41% | 52% | 57% (16/28) | 32% | 50% | 57% |
| box depth >= 3 | 0% (0/8) | 17% (2/12) | 10% (2/20) | 9% (4/46)* | 12% | 16% | 11% (3/28) | 0% | 0% | 0% |
| >= 3 assumptions (AS lines) | 25% (2/8) | 58% (7/12) | 45% (9/20) | 37% (17/46)* | 39% | 43% | 43% (12/28) | 29% | 52% | 53% |
| case split (ORE) | 38% (3/8) | 50% (6/12) | 45% (9/20) | 35% (16/46)* | 47% | 30% | 46% (13/28) | 45% | 81% | 65% |
| reductio (assume ~G, F, NEGI, DN) | 38% (3/8) | 8% (1/12) | 20% (4/20) | 17% (8/46)* | 4% | 4% | 4% (1/28) | 11% | 14% | 14% |
| plain ¬-intro (NEGI box, no DN) | 75% (6/8) | 25% (3/12) | 45% (9/20) | 35% (16/46)* | 14% | 15% | 39% (11/28) | 5% | 6% | 6% |
| →-intro box | 25% (2/8) | 58% (7/12) | 45% (9/20) | 41% (19/46)* | 63% | 80% | 61% (17/28) | 43% | 48% | 64% |
| ex falso | 0% (0/8) | 8% (1/12) | 5% (1/20) | 11% (5/46)* | 23% | 10% | 11% (3/28) | 8% | 13% | 5% |
| a negation/reductio box nested inside another box | 62% (5/8) | 0% (0/12) | 25% (5/20) | 20% (9/46)* | 5% | 6% | 14% (4/28) | 7% | 12% | 10% |
| negation box inside a negation box | 38% (3/8) | 0% (0/12) | 15% (3/20) | 11% (5/46)* | 0% | 0% | 0% (0/28) | 0% | 0% | 0% |
| negation box inside a →-intro box | 25% (2/8) | 0% (0/12) | 10% (2/20) | 7% (3/46)* | 4% | 5% | 11% (3/28) | 2% | 2% | 3% |
| case split inside a negation/reductio box | 25% (2/8) | 17% (2/12) | 20% (4/20) | 11% (5/46)* | 2% | 2% | 7% (2/28) | 1% | 1% | 2% |
| negation box inside a case branch | 0% (0/8) | 0% (0/12) | 0% (0/20) | 2% (1/46)* | 1% | 1% | 4% (1/28) | 5% | 10% | 7% |
| any box inside a case branch | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (3/46)* | 2% | 1% | 4% (1/28) | 7% | 12% | 9% |
| →-intro box inside a →-intro box | 0% (0/8) | 25% (3/12) | 15% (3/20) | 15% (7/46)* | 23% | 35% | 25% (7/28) | 8% | 7% | 16% |
| case split on an assumption (not a premise) | 12% (1/8) | 17% (2/12) | 15% (3/20) | 9% (4/46)* | 1% | 1% | 4% (1/28) | 1% | 2% | 2% |
| case split concluding F | 12% (1/8) | 25% (3/12) | 20% (4/20) | 11% (5/46)* | 3% | 2% | 7% (2/28) | 2% | 4% | 3% |
| case split concluding a compound | 12% (1/8) | 17% (2/12) | 15% (3/20) | 17% (8/46)* | 34% | 20% | 29% (8/28) | 33% | 59% | 43% |
| case branches end in Or.inl vs Or.inr | 12% (1/8) | 17% (2/12) | 15% (3/20) | 13% (6/46)* | 1% | 1% | 7% (2/28) | 1% | 2% | 1% |
| both case branches >= 2 lines | 25% (2/8) | 42% (5/12) | 35% (7/20) | 26% (12/46)* | 40% | 21% | 39% (11/28) | 26% | 47% | 24% |
| contradiction built by an intro (X built to clash with ~X) | 12% (1/8) | 8% (1/12) | 10% (2/20) | 20% (9/46)* | 17% | 11% | 29% (8/28) | 3% | 4% | 4% |
| contradiction uses an assumed negation | 38% (3/8) | 17% (2/12) | 25% (5/20) | 24% (11/46)* | 7% | 8% | 11% (3/28) | 2% | 3% | 4% |
| a ¬-intro result reused by IMPE/NEGE (negation lemma) | 25% (2/8) | 0% (0/12) | 10% (2/20) | 9% (4/46)* | 1% | 0% | 0% (0/28) | 0% | 0% | 0% |
| a →-intro result applied later (implication lemma) | 0% (0/8) | 8% (1/12) | 5% (1/20) | 7% (3/46)* | 10% | 9% | 7% (2/28) | 2% | 3% | 4% |
| any box result reused by an elimination | 38% (3/8) | 8% (1/12) | 20% (4/20) | 22% (10/46)* | 21% | 19% | 14% (4/28) | 13% | 23% | 26% |
| box results combined by ANDI/ORI | 12% (1/8) | 17% (2/12) | 15% (3/20) | 22% (10/46)* | 15% | 18% | 14% (4/28) | 12% | 19% | 25% |
| IMPE on an assumed implication | 0% (0/8) | 25% (3/12) | 15% (3/20) | 11% (5/46)* | 2% | 2% | 0% (0/28) | 1% | 1% | 1% |
| first step is forward (not an assumption) | 12% (1/8) | 0% (0/12) | 5% (1/20) | 17% (8/46)* | 59% | 49% | 43% (12/28) | 45% | 37% | 21% |

### Shortest-proof length vs nominal length

| set | n | shortest = nominal | shortest <= 6 lines | shortest 10–12 | mean nominal | mean shortest |
|---|---|---|---|---|---|---|
| TB 7–12 (reference length) | 20 | 95% | 0% | 55% | 9.4 | 9.4 |
| pool (generator length 7–12) | 5000 | 19% | 57% | 3% | 9.7 | 6.4 |
| h12 test | 1000 | 20% | 55% | 4% | 9.6 | 6.4 |
| K12 gen 7–12 | 11000 | 11% | 77% | 1% | 9.5 | 5.3 |

### Box skeletons of the shortest proofs

I(..) →-intro box, N(..) ¬-intro box, C(..) reductio box, O(a|b) case split with branch contents a, b; '-' = no box. Share of each group with that skeleton; the TB skeletons are listed, plus the pool's most common ones.

| skeleton | TB fail (n=8) | TB solved (n=12) | pool (n=5000) | pool min 7–12 (n=2157) | h12 test fail (n=28) | K12 gen 7–12 (n=11000) | K12 min 7–12 (n=2481) |
|---|---|---|---|---|---|---|---|
| `I(O(|))` | 0% (0/8) | 25% (3/12) | 0.8% | 1.8% | 11% (3/28) | 3.7% | 16.0% |
| `O(|)` | 12% (1/8) | 8% (1/12) | 1.8% | 4.0% | 4% (1/28) | 5.9% | 21.6% |
| `C(O(|))` | 12% (1/8) | 8% (1/12) | 0.0% | 0.0% | 0% (0/28) | 0.0% | 0.0% |
| `I(N())` | 25% (2/8) | 0% (0/12) | 0.5% | 1.3% | 0% (0/28) | 0.2% | 0.8% |
| `I(I(I()))` | 0% (0/8) | 17% (2/12) | 3.7% | 8.5% | 0% (0/28) | 0.2% | 0.7% |
| `N(O(|))` | 12% (1/8) | 8% (1/12) | 0.2% | 0.5% | 0% (0/28) | 0.1% | 0.2% |
| `C(N())` | 25% (2/8) | 0% (0/12) | 0.0% | 0.0% | 0% (0/28) | 0.0% | 0.0% |
| `N()N()` | 0% (0/8) | 8% (1/12) | 0.0% | 0.0% | 0% (0/28) | 0.0% | 0.0% |
| `N(N())` | 12% (1/8) | 0% (0/12) | 0.0% | 0.0% | 0% (0/28) | 0.0% | 0.0% |
| `I()` | 0% (0/8) | 8% (1/12) | 32.7% | 40.5% | 21% (6/28) | 30.7% | 24.6% |
| `I(I())` | 0% (0/8) | 8% (1/12) | 11.0% | 22.9% | 7% (2/28) | 4.2% | 16.1% |
| `N()` | 0% (0/8) | 8% (1/12) | 3.1% | 4.4% | 21% (6/28) | 1.4% | 2.4% |
| `-` | 0% (0/8) | 0% (0/12) | 44.5% | 12.7% | 25% (7/28) | 50.9% | 6.7% |

Pool share of each failed TB problem's skeleton: 0.0%, 0.0%, 0.0%, 0.0%, 0.2%, 0.5%, 0.5%, 1.8%. Solved TB problems: 0.0%, 0.0%, 0.2%, 0.8%, 0.8%, 0.8%, 1.8%, 3.1%, 3.7%, 3.7%, 11.0%, 32.7%.


### Within dev58: which features go with failure

Left block: dev58 7–12 (n = 20), mean score with / without the feature, and the failed share with / without (Fisher exact p, two-sided). Right block: all dev58 problems with a shortest proof (<= 16 lines), score residual after removing a per-length-bin mean (bins <=6, 7–9, 10–12, 13–16), with minus without.

| feature | n with (7–12) | score with | score without | failed with | failed without | p | n with (<=16) | Δ residual (pp) |
|---|---|---|---|---|---|---|---|---|
| no premises (a theorem) | 6 | 30% | 40% | 50% | 36% | 0.64 | 11 | -7 |
| goal is an atom | 4 | 23% | 40% | 50% | 38% | 1.00 | 9 | -4 |
| goal is a negation | 4 | 46% | 35% | 50% | 38% | 1.00 | 8 | +8 |
| goal is a negated compound ~(A∘B) | 3 | 31% | 38% | 67% | 35% | 0.54 | 4 | +2 |
| goal is a disjunction | 2 | 1% | 41% | 100% | 33% | 0.15 | 8 | -23 |
| goal is a conjunction | 1 | 38% | 37% | 0% | 42% | 1.00 | 4 | +14 |
| goal is an implication | 9 | 47% | 29% | 22% | 55% | 0.20 | 15 | +7 |
| goal is A>B with compound A | 7 | 46% | 32% | 14% | 54% | 0.16 | 9 | +1 |
| goal is (A>B)>C (higher-order) | 3 | 62% | 33% | 0% | 47% | 0.24 | 4 | +10 |
| some (A>B)>C anywhere | 3 | 62% | 33% | 0% | 47% | 0.24 | 6 | -4 |
| some negated compound ~(A&B) / ~(AvB) / ~(A>B) | 6 | 22% | 44% | 67% | 29% | 0.16 | 13 | -8 |
| some biconditional (A>B)&(B>A) | 2 | 1% | 41% | 100% | 33% | 0.15 | 9 | -15 |
| classical-only (not intuitionistically valid, G4ip) | 3 | 9% | 42% | 67% | 35% | 0.54 | 8 | -12 |
| F appears in the theorem | 1 | 98% | 34% | 0% | 42% | 1.00 | 3 | +13 |
| shortest proof >= 9 lines | 14 | 25% | 65% | 57% | 0% | 0.04 | 26 | -3 |
| <= 7 connectives and shortest proof >= 9 lines | 14 | 25% | 65% | 57% | 0% | 0.04 | 22 | -2 |
| reductio, or a ¬-intro box inside another box | 7 | 5% | 54% | 86% | 15% | 0.00 | 13 | -21 |
| box depth >= 2 | 15 | 33% | 49% | 47% | 20% | 0.60 | 26 | -0 |
| box depth >= 3 | 2 | 69% | 34% | 0% | 44% | 0.49 | 4 | +0 |
| >= 3 assumptions (AS lines) | 9 | 49% | 27% | 22% | 55% | 0.20 | 17 | +8 |
| case split (ORE) | 9 | 41% | 34% | 33% | 45% | 0.67 | 16 | +11 |
| reductio (assume ~G, F, NEGI, DN) | 4 | 7% | 45% | 75% | 31% | 0.26 | 8 | -20 |
| plain ¬-intro (NEGI box, no DN) | 9 | 25% | 47% | 67% | 18% | 0.06 | 16 | -10 |
| →-intro box | 9 | 47% | 29% | 22% | 55% | 0.20 | 19 | +2 |
| ex falso | 1 | 66% | 36% | 0% | 42% | 1.00 | 5 | +0 |
| a negation/reductio box nested inside another box | 5 | 2% | 49% | 100% | 20% | 0.00 | 9 | -22 |
| negation box inside a negation box | 3 | 0% | 44% | 100% | 29% | 0.05 | 5 | -26 |
| negation box inside a →-intro box | 2 | 4% | 41% | 100% | 33% | 0.15 | 3 | -13 |
| case split inside a negation/reductio box | 4 | 30% | 39% | 50% | 38% | 1.00 | 5 | +8 |
| →-intro box inside a →-intro box | 3 | 62% | 33% | 0% | 47% | 0.24 | 7 | +9 |
| case split on an assumption (not a premise) | 3 | 32% | 38% | 33% | 41% | 1.00 | 4 | +3 |
| case split concluding F | 4 | 54% | 33% | 25% | 44% | 0.62 | 5 | +21 |
| case split concluding a compound | 3 | 29% | 38% | 33% | 41% | 1.00 | 8 | +3 |
| case branches end in Or.inl vs Or.inr | 3 | 29% | 38% | 33% | 41% | 1.00 | 6 | -3 |
| both case branches >= 2 lines | 7 | 43% | 34% | 29% | 46% | 0.64 | 12 | +12 |
| contradiction built by an intro (X built to clash with ~X) | 2 | 19% | 39% | 50% | 39% | 1.00 | 9 | -15 |
| contradiction uses an assumed negation | 5 | 36% | 37% | 60% | 33% | 0.35 | 11 | -6 |
| a ¬-intro result reused by IMPE/NEGE (negation lemma) | 2 | 0% | 41% | 100% | 33% | 0.15 | 4 | -13 |
| a →-intro result applied later (implication lemma) | 1 | 45% | 37% | 0% | 42% | 1.00 | 3 | -25 |
| any box result reused by an elimination | 4 | 12% | 43% | 75% | 31% | 0.26 | 10 | -17 |
| box results combined by ANDI/ORI | 3 | 27% | 39% | 33% | 41% | 1.00 | 10 | -11 |
| IMPE on an assumed implication | 3 | 62% | 33% | 0% | 47% | 0.24 | 5 | +6 |
| first step is forward (not an assumption) | 1 | 2% | 39% | 100% | 37% | 0.40 | 8 | -2 |

Spearman correlation of numeric features with the score:

| feature | dev58 7–12 (n=20) | dev58 <= 16 lines (n=46) | dev58 <= 16, length residual |
|---|---|---|---|
| premises | +0.19 | +0.10 | +0.15 |
| distinct atoms | -0.11 | -0.14 | -0.17 |
| formula depth (max) | -0.45 | -0.29 | -0.32 |
| goal depth | -0.19 | -0.19 | -0.05 |
| connectives in the theorem | -0.57 | -0.48 | -0.30 |
| ~ count | -0.26 | -0.19 | -0.00 |
| & count | -0.42 | -0.37 | -0.20 |
| v count | -0.11 | -0.29 | -0.19 |
| > count | +0.20 | +0.06 | -0.06 |
| nested-implication depth | +0.22 | +0.09 | -0.08 |
| theorem tokens | -0.55 | -0.47 | -0.31 |
| redundant premises | +nan | +0.13 | +0.04 |
| shortest proof found (lines) | -0.61 | -0.73 | -0.09 |
| box depth | -0.04 | -0.41 | -0.07 |
| assumptions (AS lines) | +0.18 | -0.40 | +0.08 |
| elimination lines at depth >= 2 | -0.31 | -0.52 | -0.10 |
| lines at depth >= 2 | -0.07 | -0.45 | -0.03 |
| longest box (lines) | -0.42 | -0.59 | -0.15 |
| state-env actions of that proof | -0.61 | -0.73 | -0.09 |

### Scores by group (pass@1 / pass@5, % of group)

| group | n | best-4 mean pass@1 | best-4 mean pass@5 | Stage 1 mean pass@1 |
|---|---|---|---|---|
| TB 7–12 | 20 | 37.0% | 54.1% | 14.4% |
| TB fail | 8 | 2.8% | 9.5% | 0.3% |
| TB solved | 12 | 59.9% | 83.8% | 23.8% |
| TB all | 58 | 32.4% | 45.1% | 16.5% |
| train14 (not used) | 14 | 33.7% | 42.8% | 28.4% |
| h12 test | 1000 | 88.9% | 97.8% | – |
| h12 test fail | 28 | 36.4% | 66.3% | – |

### How attempts end (four best checkpoints pooled; % of attempts)

| group | attempts | ok | finished, Lean rejects (wrong inference) | type/shape (Or.elim on non-or, box on non-arrow, binder or goal mismatch) | scope (unbound name, exact not last) | syntax/format (formula, assign, term, semicolon, box head, premise position ...) | caps (step cap, context, truncated) |
|---|---|---|---|---|---|---|---|
| TB fail | 2048 | 3% | 38% | 17% | 23% | 19% | 0% |
| TB solved | 3072 | 60% | 18% | 6% | 8% | 7% | 0% |
| TB all | 14848 | 32% | 25% | 14% | 15% | 14% | 0% |
| h12 test | 64000 | 89% | 7% | 2% | 1% | 1% | 0% |
| h12 test fail | 1792 | 36% | 44% | 7% | 8% | 5% | 0% |

Most common failure statuses (% of failed attempts):

- **TB fail:** `done` 39%, `fail:S:unbound` 22%, `fail:S:Or.elim on non-or` 9%, `fail:S:term` 7%, `fail:S:binder mismatch` 6%, `fail:S:no semicolon` 3%, `fail:S:formula` 3%, `fail:S:box on non-arrow` 2%
- **TB solved:** `done` 45%, `fail:S:unbound` 18%, `fail:S:Or.elim on non-or` 8%, `fail:S:term` 7%, `fail:S:binder mismatch` 5%, `fail:S:formula` 4%, `fail:S:no semicolon` 3%, `fail:S:box on non-arrow` 3%
- **TB all:** `done` 37%, `fail:S:unbound` 20%, `fail:S:Or.elim on non-or` 10%, `fail:S:term` 7%, `fail:S:binder mismatch` 6%, `fail:S:no semicolon` 3%, `fail:S:box on non-arrow` 3%, `fail:S:formula` 3%
- **h12 test:** `done` 63%, `fail:S:unbound` 10%, `fail:S:binder mismatch` 9%, `fail:S:term` 5%, `fail:S:Or.elim on non-or` 3%, `fail:S:no semicolon` 2%, `fail:S:box on non-arrow` 2%, `fail:S:exact goal mismatch` 2%
- **h12 test fail:** `done` 69%, `fail:S:unbound` 11%, `fail:S:term` 4%, `fail:S:binder mismatch` 4%, `fail:S:Or.elim on non-or` 4%, `fail:S:exact not last` 2%, `fail:S:no semicolon` 2%, `fail:S:exact goal mismatch` 2%

Where the failing attempt stopped, as accepted actions / actions of the shortest proof (state-env actions):

| group | failed attempts | stopped in first 25% | 25–75% | past 75% | ran past the shortest proof's length |
|---|---|---|---|---|---|
| TB fail | 1991 | 15% | 25% | 60% | 37% |
| TB solved | 1233 | 11% | 15% | 75% | 51% |
| h12 test | 7109 | 2% | 17% | 82% | 63% |

Successful attempts, actions used / actions of the shortest proof found:

| group | successful attempts | mean ratio | share at the shortest length | share >= 1.5x |
|---|---|---|---|---|
| TB 7–12 | 1896 | 1.14 | 63% | 12% |
| h12 test | 56891 | 1.30 | 43% | 26% |

### Per-checkpoint pass@1 / pass@5 (% of set)

| checkpoint | TB 7–12 (n=20) pass@1 | pass@5 | TB fail (n=8) pass@1 | h12 test (n=1000) pass@1 | pass@5 |
|---|---|---|---|---|---|
| T1v1g4e_it78 (GRPO, think, B256) | 35.9% | 51.2% | 4.5% | 93.6% | 98.2% |
| grpoT0_it78 (GRPO, no think) | 39.7% | 53.4% | 0.0% | 92.8% | 96.6% |
| eiT1v1e_r2 (EI, think, B256) | 32.8% | 55.4% | 2.0% | 86.5% | 98.6% |
| eiT0_r2 (EI, no think) | 39.8% | 56.4% | 4.7% | 82.7% | 97.7% |
| T0_s0 (Stage 1, no think) | 14.8% | 28.2% | 0.0% | – | – |
| T1v1_s0 (Stage 1, think B256) | 14.0% | 37.7% | 0.5% | – | – |

### Within the hard12 test split: score with / without each structure (shortest proof)

| feature | n with | best-4 pass@1 with | without | n with, at shortest 7–12 | with | without |
|---|---|---|---|---|---|---|
| no premises (a theorem) | 25 | 93% | 89% | 16 | 93% | 87% |
| goal is an atom | 43 | 88% | 89% | 5 | 84% | 87% |
| goal is a negation | 102 | 85% | 89% | 44 | 82% | 88% |
| goal is a negated compound ~(A∘B) | 72 | 84% | 89% | 35 | 79% | 88% |
| goal is a disjunction | 86 | 89% | 89% | 9 | 87% | 87% |
| goal is a conjunction | 192 | 91% | 88% | 61 | 88% | 87% |
| goal is an implication | 574 | 89% | 89% | 330 | 88% | 86% |
| goal is A>B with compound A | 463 | 89% | 89% | 263 | 87% | 87% |
| goal is (A>B)>C (higher-order) | 41 | 90% | 89% | 21 | 89% | 87% |
| some (A>B)>C anywhere | 195 | 89% | 89% | 94 | 87% | 87% |
| some negated compound ~(A&B) / ~(AvB) / ~(A>B) | 491 | 88% | 89% | 230 | 86% | 88% |
| some double negation ~~A | 431 | 89% | 89% | 229 | 88% | 86% |
| some biconditional (A>B)&(B>A) | 9 | 89% | 89% | 3 | 97% | 87% |
| classical-only (not intuitionistically valid, G4ip) | 35 | 93% | 89% | 26 | 93% | 87% |
| premises jointly inconsistent | 326 | 90% | 88% | 66 | 87% | 87% |
| F appears in the theorem | 108 | 88% | 89% | 52 | 86% | 87% |
| shortest proof >= 9 lines | 75 | 77% | 90% | 75 | 77% | 89% |
| <= 7 connectives and shortest proof >= 9 lines | 5 | 86% | 89% | 5 | 86% | 87% |
| reductio, or a ¬-intro box inside another box | 17 | 85% | 89% | 16 | 85% | 87% |
| no box at all | 451 | 90% | 88% | 60 | 90% | 87% |
| box depth >= 2 | 158 | 87% | 89% | 146 | 86% | 88% |
| box depth >= 3 | 42 | 86% | 89% | 42 | 86% | 87% |
| >= 3 assumptions (AS lines) | 53 | 81% | 89% | 53 | 81% | 88% |
| case split (ORE) | 35 | 74% | 89% | 34 | 73% | 88% |
| reductio (assume ~G, F, NEGI, DN) | 12 | 90% | 89% | 11 | 91% | 87% |
| plain ¬-intro (NEGI box, no DN) | 45 | 77% | 89% | 35 | 75% | 88% |
| →-intro box | 477 | 89% | 89% | 330 | 88% | 85% |
| ex falso | 427 | 91% | 88% | 155 | 89% | 86% |
| a negation/reductio box nested inside another box | 6 | 73% | 89% | 6 | 73% | 87% |
| negation box inside a →-intro box | 6 | 73% | 89% | 6 | 73% | 87% |
| case split inside a negation/reductio box | 3 | 60% | 89% | 3 | 60% | 87% |
| →-intro box inside a →-intro box | 144 | 89% | 89% | 133 | 88% | 87% |
| case split concluding F | 6 | 68% | 89% | 6 | 68% | 87% |
| case split concluding a compound | 16 | 78% | 89% | 16 | 78% | 87% |
| both case branches >= 2 lines | 14 | 63% | 89% | 14 | 63% | 88% |
| contradiction built by an intro (X built to clash with ~X) | 143 | 88% | 89% | 57 | 83% | 88% |
| contradiction uses an assumed negation | 117 | 89% | 89% | 69 | 87% | 87% |
| a ¬-intro result reused by IMPE/NEGE (negation lemma) | 5 | 82% | 89% | 5 | 82% | 87% |
| a →-intro result applied later (implication lemma) | 29 | 91% | 89% | 14 | 90% | 87% |
| any box result reused by an elimination | 59 | 87% | 89% | 44 | 85% | 87% |
| box results combined by ANDI/ORI | 44 | 83% | 89% | 40 | 82% | 87% |
| IMPE on an assumed implication | 13 | 85% | 89% | 13 | 85% | 87% |
| first step is forward (not an assumption) | 462 | 90% | 88% | 71 | 88% | 87% |

### Pool: how often a generator-proof structure survives in the shortest proof

| structure | generator proofs with it | of those, shortest proof keeps it |
|---|---|---|
| case split | 47% (2357) | 6% |
| reductio | 4% (192) | 22% |
| plain ¬-intro | 14% (686) | 31% |
| →-intro box | 63% (3167) | 78% |
| ex falso | 23% (1130) | 82% |
| box depth >= 2 | 41% (2028) | 39% |
| nested negation box | 5% (274) | 14% |
| box inside a case branch | 2% (84) | 1% |
| case split concluding a compound | 34% (1714) | 3% |

### Compact theorem, long proof

Share of each group whose theorem has <= 7 connectives and whose shortest proof has >= 9 lines; and shortest-proof lines per theorem connective (mean).

| group | n | <= 7 connectives and shortest >= 9 lines | lines per connective |
|---|---|---|---|
| TB 7–12 | 20 | 70% | 2.01 |
| TB fail | 8 | 100% | 1.87 |
| pool | 5000 | 1% | 0.64 |
| pool min 7–12 | 2157 | 2% | 0.75 |
| K12 gen 7–12 | 11000 | 1% | 0.67 |
| K12 min 7–12 | 2481 | 4% | 0.87 |

### Is the structure necessary? Restricted exact search

For each problem whose shortest proof uses the structure, the shortest proof *without* it (search from L to L + 3 lines, 120 s). 'needed' = no proof without it within L + 3 lines; 'same length' = an equally short proof without it exists.

| structure forbidden | group | n | needed (none within L+3) | +1 to +3 lines | same length | timeout |
|---|---|---|---|---|---|---|
| negation box inside a box | TB fail | 5 | 100% (5/5) | 0% (0/5) | 0% (0/5) | 0% (0/5) |
| negation box inside a box | pool (random <= 300) | 41 | 0% | 80% | 20% | 0% |
| DN (reductio) | TB fail | 3 | 67% (2/3) | 33% (1/3) | 0% (0/3) | 0% (0/3) |
| DN (reductio) | TB solved | 1 | 100% (1/1) | 0% (0/1) | 0% (0/1) | 0% (0/1) |
| DN (reductio) | pool (random <= 300) | 300 | 46% | 43% | 11% | 0% |
| case split | TB fail | 3 | 100% (3/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) |
| case split | TB solved | 6 | 100% (6/6) | 0% (0/6) | 0% (0/6) | 0% (0/6) |
| case split | pool (random <= 300) | 150 | 92% | 7% | 1% | 0% |
| box depth >= 2 | TB fail | 7 | 100% (7/7) | 0% (0/7) | 0% (0/7) | 0% (0/7) |
| box depth >= 2 | TB solved | 8 | 88% (7/8) | 12% (1/8) | 0% (0/8) | 0% (0/8) |
| box depth >= 2 | pool (random <= 300) | 300 | 27% | 62% | 12% | 0% |

### Pool by generator config: shortest-proof length

| config | n | shortest <= 6 | shortest 10–12 | nested negation box | reductio | case split (shortest proof) |
|---|---|---|---|---|---|---|
| strict_long | 1674 | 61% | 3% | 1% | 0% | 2% |
| fwd_takehome_d2 | 1245 | 55% | 3% | 1% | 0% | 3% |
| fwd_strict_d2 | 701 | 57% | 2% | 1% | 0% | 2% |
| takehome | 679 | 52% | 5% | 0% | 3% | 8% |
| fwd_strict_d1 | 459 | 62% | 0% | 0% | 0% | 2% |
| goal_takehome | 242 | 45% | 9% | 2% | 8% | 0% |

### train14 (reported separately; not used for any conclusion)

| feature | train14 |
|---|---|
| no premises (a theorem) | 21% (3/14) |
| goal is an atom | 0% (0/14) |
| goal is a negation | 14% (2/14) |
| goal is a negated compound ~(A∘B) | 7% (1/14) |
| goal is a disjunction | 29% (4/14) |
| goal is a conjunction | 14% (2/14) |
| goal is an implication | 36% (5/14) |
| goal is A>B with compound A | 14% (2/14) |
| goal is (A>B)>C (higher-order) | 7% (1/14) |
| some (A>B)>C anywhere | 14% (2/14) |
| some negated compound ~(A&B) / ~(AvB) / ~(A>B) | 29% (4/14) |
| some double negation ~~A | 14% (2/14) |
| some biconditional (A>B)&(B>A) | 29% (4/14) |
| classical-only (not intuitionistically valid, G4ip) | 29% (4/14) |
| premises jointly inconsistent | 21% (3/14) |
| F appears in the theorem | 7% (1/14) |
| shortest proof >= 9 lines | 62% (8/13) |
| <= 7 connectives and shortest proof >= 9 lines | 38% (5/13) |
| reductio, or a ¬-intro box inside another box | 54% (7/13) |
| no box at all | 8% (1/13) |
| box depth >= 2 | 69% (9/13) |
| box depth >= 3 | 15% (2/13) |
| >= 3 assumptions (AS lines) | 31% (4/13) |
| case split (ORE) | 8% (1/13) |
| reductio (assume ~G, F, NEGI, DN) | 23% (3/13) |
| plain ¬-intro (NEGI box, no DN) | 46% (6/13) |
| →-intro box | 54% (7/13) |
| ex falso | 15% (2/13) |
| a negation/reductio box nested inside another box | 54% (7/13) |
| negation box inside a negation box | 31% (4/13) |
| negation box inside a →-intro box | 23% (3/13) |
| case split inside a negation/reductio box | 0% (0/13) |
| negation box inside a case branch | 0% (0/13) |
| any box inside a case branch | 0% (0/13) |
| →-intro box inside a →-intro box | 23% (3/13) |
| case split on an assumption (not a premise) | 8% (1/13) |
| case split concluding F | 0% (0/13) |
| case split concluding a compound | 0% (0/13) |
| case branches end in Or.inl vs Or.inr | 0% (0/13) |
| both case branches >= 2 lines | 0% (0/13) |
| contradiction built by an intro (X built to clash with ~X) | 38% (5/13) |
| contradiction uses an assumed negation | 38% (5/13) |
| a ¬-intro result reused by IMPE/NEGE (negation lemma) | 15% (2/13) |
| a →-intro result applied later (implication lemma) | 15% (2/13) |
| any box result reused by an elimination | 38% (5/13) |
| box results combined by ANDI/ORI | 38% (5/13) |
| IMPE on an assumed implication | 15% (2/13) |
| first step is forward (not an assumption) | 8% (1/13) |

train14 shortest-proof lengths: {2: 1, 4: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 3, 12: 2, 14: 1, 99: 1} (99 = none found).
train14 best-4 mean pass@1: 33.7%.

