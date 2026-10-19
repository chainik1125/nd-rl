---
written_on: 2026-10-08
written_by: agent:claude
---

# Gap analysis: which 7–12-line problems the generator does not make, and the RL'd models fail

> **dev58 was used for design decisions here.** This note used textbook dev58 to choose generator targets: its
> per-problem pass rates and the structure of its shortest proofs. dev58 is therefore **no longer a clean held-out
> set for claims about a generator designed from this note.** A gain on dev58 after training on such a generator is
> partly by design. Only aggregate structural features were used: no textbook problem, renaming or template is
> copied here or into any generator. train14 is reported apart and was not used for any conclusion.

**Question.** At 7–12 reference lines, the best RL'd checkpoints reach 33–40% pass@1 on textbook dev58 problems
(n = 20) but 83–94% on generated hard12 test problems of the same nominal length. Which kinds of problems does the
generator rarely or never produce, that textbook problems contain and that the models fail on? The answer is meant to
guide a new generator of harder 7–12-line training problems.

**How this was done.** CPU only (≤ 8 threads, `nice`). No GPU, training or eval was run.

- **Inputs:** existing eval rows (k = 64 on textbook72, k = 16 on hard12_test_1k).
- **Shortest proofs:** a fresh exact search for every textbook problem up to 16 lines, every pool and test problem,
  and the 7–12-line part of a random 20k K12 sample.
- **Theorem features:** for every problem.
- **Necessity check:** a restricted re-search on the structures that matter.

Scripts and regeneration commands are under [Files](#files-and-regeneration).

## Bottom line

Textbook 7–12-line problems are **small statements with long, necessary proofs**. The models fail on the ones whose
proof does negation work below the top level. The generator's "7–12-line" problems are big statements whose proofs
mostly collapse to 3–6 lines; their difficulty comes from generator artifacts that textbook problems lack.

**Observed.** The top separating features, failed textbook 7–12s (TB fail, n = 8) vs the RL pool (n = 5,000):

1. **The shortest proof really is long, on a small statement.**
   - **Shortest proof:** ≥ 9 lines in 100% (8/8) of failed textbook problems and 70% (14/20) of all textbook 7–12s,
     vs **8%** of the pool and 4% of K12 at 7–12 generator lines.
   - **With ≤ 7 connectives in the theorem:** 100% (8/8) and 70% (14/20) vs **1%** of the pool.
   - **Pool by nominal length:** 57% of the pool's "7–12-line" problems have a proof of ≤ 6 lines, and only 3%
     need 10–12 (textbook 7–12s: 0% and 55%).
   - **Proof lines per theorem connective:** 2.0 on textbook vs 0.64 in the pool.
   - **Within dev58 7–12:** every one of the 6 problems with a shortest proof under 9 lines is solved.
     8 of the 14 with ≥ 9 lines fail (p = 0.04).
2. **Reductio, or a ¬-intro box nested inside another box** (shortest-proof skeletons such as I(N()), N(N()),
   C(N()), C(O(|))).
   - **Prevalence:** 75% (6/8) of failed vs 8% (1/12) of solved textbook 7–12s (Fisher p = 0.004), vs **2%** of the
     pool, 3% of K12 7–12 and 4% of the pool at shortest length 7–12.
   - **The nested ¬-box alone:** 62% (5/8) vs 0% (0/12) (p = 0.004), vs **1%** of the pool.
   - **Necessity:** in all 5 failed textbook problems no proof without the nested box exists within L + 3 lines. In
     the pool, every sampled nested ¬-box (41/41) can be avoided within 3 extra lines.
3. **Negation introduction at all.**
   - **Plain ¬-intro:** 75% (6/8) of failed vs 25% (3/12) of solved (p = 0.06), vs **4%** of the pool's shortest
     proofs.
   - **Reductio:** 38% (3/8) vs 8% (1/12), vs **1%**.
   - **Classical-only theorems:** 25% (2/8) of failed and 26% (15/58) of dev58, vs **4%** of the pool and K12.
4. **Box depth ≥ 2, with elimination work at depth 2.**
   - **Box depth ≥ 2:** 88% (7/8) vs **17%** of the pool's shortest proofs.
   - **Elimination lines at depth ≥ 2:** mean 3.0 vs **0.26**.
   - **Necessity:** depth 2 is needed in 14/15 textbook 7–12s that have it, vs 27% of the pool rows that have it.
5. **Statement shape.**
   - **No premises:** 30% (6/20) of textbook 7–12s vs 3% of the pool.
   - **Biconditional sub-formulas:** 28% (16/58) of dev58 vs 0.4% of the pool.
   - **Atomic, negated or disjunctive goals:** 75% (6/8) of failed problems.
   - **Implication goals:** 58% (7/12) of solved problems.

**Also observed.**

- **Necessary case splits** are common in textbook 7–12s and rare in the generator: 45% (9/20), every one
  necessary, vs 3% of the pool's shortest proofs. 47% of the pool's generator proofs contain a case split, but 94% of
  those are removable. Case splits do not separate failed from solved textbook problems (33% vs 45% failed).
- **The pool carries artifacts that textbook 7–12s lack:**
  - double negation in the statement: 43% of the pool vs 0%;
  - jointly inconsistent premises: 33% vs 0%;
  - redundant premises: 1.19 per theorem vs 0;
  - ex falso in the shortest proof: 42% vs 5%.
- **On textbook problems the models break the action grammar more often.**
  - **State-env rejections:** 61% of failed attempts on failed textbook problems (scope 24%, mostly unbound names;
    type/shape 18%; syntax 19%), vs 36% on hard12 test.
  - **Finished proofs that Lean rejects:** 39% vs 63%.
  - **Step cap:** never binds (≤ 0.4%).
  - **Early stops:** failing attempts on failed textbook problems stop in the first quarter of the shortest proof
    15% of the time, vs 2% on hard12 test.

**Interpretation, not tested causally.** Most proofs in the RL pool shorten to 3–6 lines. Successful hard12 attempts
use 1.30× the shortest proof's actions, so the models seldom have to carry a necessary 9–12-line argument. The pool
rarely asks for what textbook failures need: deriving ⊥ inside a nested box, closing a ¬-intro or reductio there, and
reusing its result.
The generator mechanism behind this is described [below](#why-the-generator-misses-these-interpretation). The fix is
a **proof-first generator gated on the shortest proof**; [targets below](#generator-targets).

## 1. The gap being explained

Pass@1 / pass@5, unbiased estimator, % of set. Each checkpoint is shown at its eval budget (think arms B256, no-think
B0).

| checkpoint | TB 7–12 (n=20) pass@1 | pass@5 | TB fail (n=8) pass@1 | hard12 test (n=1,000) pass@1 | pass@5 |
|---|---|---|---|---|---|
| T1v1g4e_it78 (GRPO, think, B256) | 35.9% | 51.2% | 4.5% | 93.6% | 98.2% |
| grpoT0_it78 (GRPO, no think) | 39.7% | 53.4% | 0.0% | 92.8% | 96.6% |
| eiT1v1e_r2 (EI, think, B256) | 32.8% | 55.4% | 2.0% | 86.5% | 98.6% |
| eiT0_r2 (EI, no think) | 39.8% | 56.4% | 4.7% | 82.7% | 97.7% |
| T0_s0 (Stage 1, no think) | 14.8% | 28.2% | 0.0% | – | – |
| T1v1_s0 (Stage 1, think, B256) | 14.0% | 37.7% | 0.5% | – | – |

**Groups used below.**

- **TB 7–12:** dev58 with reference length 7–12 (n = 20).
  - The brief's "n = 20 across dev + train" is in fact all dev58. train14 has no reference length.
  - The exact search reproduced the reference length for 19 / 20; one is a line shorter.
- **Score:** a problem's mean pass@1 over the four RL'd checkpoints.
- **TB fail:** score < 25% (n = 8: per-problem scores 0–12%). **TB solved:** the other 12 (27–98%). The 25% threshold
  was fixed before looking at features; the continuous-score analyses in §4 do not depend on it.
- **pool:** `rl_hard12.jsonl`, all 5,000.
- **h12 test:** `hard12_test_1k`. **h12 test fail:** score < 50% (n = 28; only 6 are under 25%).
- **K12:** a random 20,000 of `train_k12.jsonl` (seed 20261008).
  - "K12 gen 7–12": its 11,000 problems with 7–12 generator lines.
  - "min 7–12" columns: rows whose shortest proof is 7–12 lines.

## 2. Nominal length vs necessary length

| set | n | shortest = nominal | shortest ≤ 6 lines | shortest 10–12 | mean nominal | mean shortest |
|---|---|---|---|---|---|---|
| TB 7–12 (reference length) | 20 | 95% | 0% | 55% | 9.4 | 9.4 |
| pool (generator length 7–12) | 5,000 | 19% | 57% | 3% | 9.7 | 6.4 |
| hard12 test | 1,000 | 20% | 55% | 4% | 9.6 | 6.4 |
| K12, generator length 7–12 | 11,000 | 11% | 77% | 1% | 9.5 | 5.3 |

**Compact theorem, long proof.** Share with ≤ 7 connectives in the theorem and a shortest proof of ≥ 9 lines:

- TB 7–12: 70% (14/20); TB fail: 100% (8/8);
- pool: 0.9%; pool at shortest 7–12: 2.1%;
- K12 gen 7–12: 0.9%; K12 at shortest 7–12: 4.0%.

The pool's generator configs differ little in how far their proofs shorten. Shortest ≤ 6 lines: 45% (`goal_takehome`)
to 62% (`fwd_strict_d1`). Shortest 10–12 lines: 0–9%. Full table in [`tables.md`](tables.md).

## 3. Feature prevalence

% of group, with k/n for the small groups. Shortest-proof features come from the exact search's proof. An asterisk
marks a group where some rows have no proof found: the dev58 problems over 16 lines.

**Theorem shape.**

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | dev58 (n=58) | pool (n=5,000) | pool, shortest 7–12 (n=2,157) | h12 test fail (n=28) | K12 (n=20,000) |
|---|---|---|---|---|---|---|---|---|
| no premises | 38% (3/8) | 25% (3/12) | 30% (6/20) | 22% (13/58) | 3% | 4% | 0% | 6% |
| goal is an atom | 25% (2/8) | 17% (2/12) | 20% (4/20) | 16% (9/58) | 6% | 2% | 0% | 5% |
| goal is a negation | 25% (2/8) | 17% (2/12) | 20% (4/20) | 14% (8/58) | 10% | 9% | 29% | 5% |
| goal is a disjunction | 25% (2/8) | 0% (0/12) | 10% (2/20) | 17% (10/58) | 9% | 3% | 7% | 24% |
| goal is an implication | 25% (2/8) | 58% (7/12) | 45% (9/20) | 28% (16/58) | 55% | 71% | 61% | 43% |
| biconditional (A>B)&(B>A) inside | 25% (2/8) | 0% (0/12) | 10% (2/20) | 28% (16/58) | 0% | 1% | 4% | 0% |
| negated compound ~(A∘B) inside | 50% (4/8) | 17% (2/12) | 30% (6/20) | 33% (19/58) | 49% | 46% | 61% | 37% |
| double negation ~~A inside | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (4/58) | 43% | 51% | 61% | 26% |
| classical-only (G4ip) | 25% (2/8) | 8% (1/12) | 15% (3/20) | 26% (15/58) | 4% | 5% | 4% | 4% |
| premises jointly inconsistent | 0% (0/8) | 0% (0/12) | 0% (0/20) | 7% (4/58) | 33% | 16% | 32% | 24% |
| **mean** connectives | 5.8 | 4.3 | 4.9 | 6.1 | 11.4 | 11.7 | 12.5 | 9.1 |
| **mean** premises | 0.88 | 1.25 | 1.10 | 1.24 | 2.24 | 2.17 | 2.29 | 1.82 |
| **mean** distinct atoms | 2.4 | 2.5 | 2.5 | 2.6 | 3.1 | 3.2 | 3.1 | 2.9 |
| **mean** formula depth | 3.0 | 2.1 | 2.5 | 2.7 | 4.0 | 4.2 | 4.1 | 3.5 |
| **mean** nested-implication depth | 0.75 | 1.25 | 1.05 | 0.97 | 1.76 | 2.05 | 1.39 | 1.44 |
| **mean** redundant premises | 0 | 0 | 0 | 0.03 | 1.19 | 1.06 | 1.07 | 0.82 |

**Shortest-proof structure.** K12 here is its 7–12-generator-line part (n = 11,000). The last column is the
**generator's own proof** of the pool rows, for comparison.

| feature | TB fail (n=8) | TB solved (n=12) | TB 7–12 (n=20) | dev58 ≤16 lines (n=46) | pool | pool, shortest 7–12 | h12 test fail | K12 gen 7–12 | pool, generator proof |
|---|---|---|---|---|---|---|---|---|---|
| shortest proof ≥ 9 lines | 100% (8/8) | 50% (6/12) | 70% (14/20) | 57% | 8% | 18% | 39% | 4% | (69%) |
| reductio, or a ¬-intro box inside another box | **75% (6/8)** | 8% (1/12) | 35% (7/20) | 28% | **2%** | 4% | 4% | 3% | 8% |
| ¬-intro / reductio box nested inside another box | **62% (5/8)** | 0% (0/12) | 25% (5/20) | 20% | **1%** | 2% | 4% | 1% | 5% |
| … inside a ¬-intro / reductio box | 38% (3/8) | 0% (0/12) | 15% (3/20) | 11% | 0% | 0% | 0% | 0% | 0% |
| … inside a →-intro box | 25% (2/8) | 0% (0/12) | 10% (2/20) | 7% | 1% | 2% | 4% | 1% | 4% |
| plain ¬-intro (NEGI box, no DN) | 75% (6/8) | 25% (3/12) | 45% (9/20) | 35% | 4% | 7% | 25% | 2% | 14% |
| reductio (assume ~G, ⊥, NEGI, DN) | 38% (3/8) | 8% (1/12) | 20% (4/20) | 17% | 1% | 2% | 0% | 2% | 4% |
| case split (ORE) | 38% (3/8) | 50% (6/12) | 45% (9/20) | 35% | 3% | 7% | 18% | 10% | 47% |
| case split inside a ¬-intro / reductio box | 25% (2/8) | 17% (2/12) | 20% (4/20) | 11% | 0% | 1% | 0% | 0% | 2% |
| both case branches ≥ 2 lines | 25% (2/8) | 42% (5/12) | 35% (7/20) | 26% | 1% | 2% | 18% | 1% | 40% |
| case split on an assumption | 12% (1/8) | 17% (2/12) | 15% (3/20) | 9% | 0% | 0% | 4% | 0% | 1% |
| box depth ≥ 2 | 88% (7/8) | 67% (8/12) | 75% (15/20) | 57% | 17% | 36% | 29% | 9% | 41% |
| no box at all | 0% (0/8) | 0% (0/12) | 0% (0/20) | 11% | 44% | 13% | 25% | 51% | 3% |
| →-intro box | 25% (2/8) | 58% (7/12) | 45% (9/20) | 41% | 50% | 77% | 50% | 40% | 63% |
| ex falso | 0% (0/8) | 8% (1/12) | 5% (1/20) | 11% | 42% | 34% | 25% | 28% | 23% |
| a ¬-intro result reused by IMPE / NEGE | 25% (2/8) | 0% (0/12) | 10% (2/20) | 9% | 0% | 1% | 0% | 0% | 1% |
| a box result reused by an elimination | 38% (3/8) | 8% (1/12) | 20% (4/20) | 22% | 7% | 11% | 7% | 6% | 21% |
| first step forward (not an assumption) | 12% (1/8) | 0% (0/12) | 5% (1/20) | 17% | 46% | 17% | 25% | 52% | – |
| **mean** elimination lines at depth ≥ 2 | 3.0 | 1.25 | 1.95 | 1.48 | 0.26 | 0.57 | 0.43 | 0.08 | – |
| **mean** longest box (lines) | 7.5 | 5.9 | 6.6 | 5.6 | 2.1 | 3.8 | 3.8 | 1.4 | – |

**Box skeletons of the shortest proofs.**

- **Notation:** I = →-intro box, N = ¬-intro box, C = reductio box, O(a|b) = case split with branch contents a and b;
  "-" = no box.
- **TB fail:** I(N()) ×2, C(N()) ×2, N(N()), C(O(|)), N(O(|)), O(|).
- **TB solved:** I(O(|)) ×3, I(I(I())) ×2, and one each of O(|), C(O(|)), N(O(|)), N()N(), I(), I(I()), N().
- **Pool shares of the failed problems' skeletons:** 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.5 and 1.8%.
  - C(N()) and C(O(|)) occur **0 times** in the 5,000-problem pool and in the 11,000 K12 7–12s.
  - N(N()) occurs 0 times in the pool and once in K12; I(N()) is 0.5% of the pool (27 problems).
- **The pool's own mix:** 44% no box, 33% I() and 11% I(I()).

## 4. Within dev58: what goes with failure

dev58 7–12 (n = 20): the mean score with and without each feature, and the failed share with and without (Fisher
exact, two-sided). The right-hand columns use dev58 rows ≤ 16 lines (n = 46). There the score is first residualised on
length bins (≤ 6, 7–9, 10–12, 13–16), and the column gives the residual with the feature minus without.

| feature | n with (7–12) | score with | score without | failed with | failed without | p | n with (≤16) | Δ residual |
|---|---|---|---|---|---|---|---|---|
| reductio, or a ¬-intro box inside another box | 7 | 5% | 54% | 86% | 15% | 0.004 | 13 | −21 pp |
| ¬-intro / reductio box nested inside another box | 5 | 2% | 49% | 100% | 20% | 0.004 | 9 | −22 pp |
| ¬-intro box inside a ¬-intro box | 3 | 0% | 44% | 100% | 29% | 0.05 | 5 | −26 pp |
| shortest proof ≥ 9 lines | 14 | 25% | 65% | 57% | 0% | 0.04 | 26 | (length) |
| plain ¬-intro | 9 | 25% | 47% | 67% | 18% | 0.06 | 16 | −10 pp |
| reductio | 4 | 7% | 45% | 75% | 31% | 0.26 | 8 | −20 pp |
| classical-only | 3 | 9% | 42% | 67% | 35% | 0.54 | 8 | −12 pp |
| goal is a disjunction | 2 | 1% | 41% | 100% | 33% | 0.15 | 8 | −23 pp |
| biconditional inside | 2 | 1% | 41% | 100% | 33% | 0.15 | 9 | −15 pp |
| goal is an implication | 9 | 47% | 29% | 22% | 55% | 0.20 | 15 | +7 pp |
| case split | 9 | 41% | 34% | 33% | 45% | 0.67 | 16 | +11 pp |

**Spearman correlations with the score** (dev58 7–12):

- shortest-proof length −0.61;
- theorem connectives −0.57;
- formula depth −0.45;
- longest box −0.42.

On the dev58 ≤ 16 residuals, all of these shrink to between −0.09 and −0.32. Every within-dev result is exploratory:
about 45 features were tested on 20 problems. With a Bonferroni correction none passes 0.05; the two nested-negation
features come closest (p = 0.004 each).

**Within the generator too.** On the hard12 test split, the structures that mark textbook failures are also harder,
though rare (best-4 mean pass@1, with vs without):

- nested ¬-box: 73% vs 89% (n = 6);
- both case branches ≥ 2 lines: 63% vs 89% (n = 14);
- case split: 74% vs 89% (n = 35);
- plain ¬-intro: 77% vs 89% (n = 45);
- shortest ≥ 9 lines: 77% vs 90% (n = 75).

## 5. Are these structures necessary? (restricted exact search)

**Method.** The problems are those whose shortest proof (length L) uses the structure. For each, the search was rerun
with the structure forbidden, from L to L + 3 lines, at 120 s per problem.

- **needed:** no proof without the structure exists within L + 3 lines.
- **same length:** an equally short proof without it exists.
- **Pool rows:** a random ≤ 300 per structure, drawn from rows whose shortest proof uses it.
- **Timeouts:** none.

| structure forbidden | group | n | needed | +1 to +3 lines | same length |
|---|---|---|---|---|---|
| ¬-intro / reductio box inside a box | TB fail | 5 | 100% (5/5) | 0% | 0% |
| ¬-intro / reductio box inside a box | pool | 41 | 0% | 80% | 20% |
| DN (reductio) | TB 7–12 | 4 | 75% (3/4) | 25% (1/4) | 0% |
| DN (reductio) | pool | 300 | 46% | 43% | 11% |
| case split | TB 7–12 | 9 | 100% (9/9) | 0% | 0% |
| case split | pool | 150 | 92% | 7% | 1% |
| box depth ≥ 2 | TB 7–12 | 15 | 93% (14/15) | 7% (1/15) | 0% |
| box depth ≥ 2 | pool | 300 | 27% | 62% | 12% |

**What the generator's proofs keep (pool).** The second column is the share of generator proofs that use the
structure; the third, of those, the share whose shortest proof still uses it.

| structure | share of generator proofs | kept by the shortest proof |
|---|---|---|
| case split | 47% | 6% |
| case split concluding a compound | 34% | 3% |
| box depth ≥ 2 | 41% | 39% |
| nested ¬-box | 5% | 14% |
| plain ¬-intro | 14% | 31% |
| reductio | 4% | 22% |
| →-intro box | 63% | 78% |
| ex falso | 23% | 82% |

## 6. How attempts end

The four best checkpoints pooled; % of attempts. These are state-env rejection reasons; "Lean rejects" means the
proof finished and Lean refused it.

| group | attempts | ok | Lean rejects | type/shape | scope | syntax/format | caps |
|---|---|---|---|---|---|---|---|
| TB fail | 2,048 | 3% | 38% | 17% | 23% | 19% | 0% |
| TB solved | 3,072 | 60% | 18% | 6% | 8% | 7% | 0% |
| dev58 | 14,848 | 32% | 25% | 14% | 15% | 14% | 0% |
| h12 test | 64,000 | 89% | 7% | 2% | 1% | 1% | 0% |
| h12 test fail | 1,792 | 36% | 44% | 7% | 8% | 5% | 0% |

**Column key.**

- type/shape: Or.elim on a non-disjunction, box on a non-arrow, binder or goal mismatch.
- scope: unbound name, exact not last.
- syntax/format: formula, assign, term, semicolon, box head, premise position.
- caps: step cap, context, truncated.

**Most common reasons among failed attempts.**

- **TB fail:** finished but Lean-rejected 39%, `unbound` 22%, `Or.elim on non-or` 9%, `term` 7%, `binder mismatch` 6%.
- **h12 test:** finished but Lean-rejected 63%, `unbound` 10%, `binder mismatch` 9%, `term` 5%.

On textbook problems, failed attempts hit state-env rejections (scope, type, syntax) more often than on the
generator's problems: 61% vs 36%. Solved textbook problems show a similar mix (54%). The step cap (96) never binds.

**Where failing attempts stop**, measured as the accepted actions as a share of the shortest proof's actions:

| group | in the first 25% | 25–75% | past 75% |
|---|---|---|---|
| TB fail | 15% | 25% | 60% |
| h12 test | 2% | 17% | 82% |

**Successful attempts** use 1.14× the shortest proof's actions on TB 7–12 (63% exactly at the shortest length). On
h12 test they use 1.30× (43% exactly; 26% use ≥ 1.5×).

## Why the generator misses these (interpretation)

The generator is the fork's `gen.Gen` (`nd-takehome/gen.py`). hard12 uses it unchanged, through the knobs in
`pools/hard12/gen_steer.py`.

**Its knobs.**

- **Rule set:** the 14 K12 rules (PR / AS / R, ANDI / ANDE1-2, ORI1-2 / ORE, IMPI / IMPE, NEGI / NEGE, DN, BOTE).
- **Random formulas:** `rf` with depth 2 over atoms P / Q / R / S. A leaf has p 0.45; an internal node is ¬ with
  p 0.28 and otherwise &, v, > uniformly. F is an atom with p 0.02 (take-home mode; 0 in strict mode).
- **Caps:** at most 3 premises (`max_prem`); box nesting `max_depth` 1–3 depending on the config.
- **hard12 gates:** 7–12 *generator* lines, theorem formula depth ≤ 5, prompt ≤ 156 tokens. Each kept problem was then
  band-selected on T0_s0's pass rate.
- **Forward mode:** random rule applications. Boxes open on random hypotheses and close at random (IMPI, or NEGI when
  the box ends in ⊥). Missing inputs become **lazy premises**.
- **Goal mode:** `reach()` decomposes a random goal by its intro rule. Otherwise it uses an available line, ex
  falso, DN or reductio (offered as an option half the time), or adds a lazy premise (Z > G).
- **Case splits:** ORE branch 1 takes 0–`ore_steps` local steps with no inner boxes, unless `ore_boxes` is on.
  Branch 2 is completed by `reach(G, 2)`.
- **Contradictions:** `reach(⊥)` closes with an available X / ¬X pair or a lazy ¬X premise. It never builds X by a
  chain.
- **Pruning:** `finish()` removes uncited lines and premises, but not unnecessary structure.

**Consequences (consistent with §2–5, not tested causally).**

- The length gate counts generator lines, so most "7–12-line" problems shorten to 3–6.
- Branch 2 of a case split usually derives G without its hypothesis, so 94% of case splits are removable.
- ¬-intro and reductio boxes appear mainly at the top, where the goal is a negation. Inside a box, ⊥ comes from a
  ready-made pair or a lazy premise, so negation work below the top level is almost never necessary.
- Lazy premises inflate the statement: 11 connectives, 1.2 redundant premises, 33% inconsistent premise sets, 43%
  double negations. The pool's T0-hardness comes largely from these
  ([hard12 README](../pools/hard12/README.md) §1), and textbook problems have none of them.

## Generator targets

Each target is a structural recipe for a proof-first generator: build the proof skeleton first, fill it, read the
theorem off the proof, then gate on the exact search. These are not formulas. Shares are suggested proportions of a
new 7–12-line set; the evidence columns repeat the numbers above.

| # | target | recipe | textbook 7–12 (TB fail) | pool now | suggested share |
|---|---|---|---|---|---|
| T1 | **Necessary length** | Keep a problem only if its *shortest* proof (`minproofs.py`, < 1 s per problem at ≤ 12 lines) is 7–12 lines and within 1 line of the constructed proof. Weight 9–12 lines. | 70% ≥ 9 lines (100%) | 8% ≥ 9; 57% ≤ 6 | ≥ 60% at 9–12 |
| T2 | **Compact statement** | ≤ 7 connectives, ≤ 2 premises (25–35% with none), 2–3 atoms, formula depth ≤ 3. Pair this with T1, so proof lines per connective ≥ 1.5. | 70% compact + long (100%) | 1% | ≥ 60% |
| T3 | **Negation work below the top level** | A ¬-intro box nested inside a →-intro, ¬-intro or reductio box (skeletons I(N()), N(N()), C(N())). The inner ⊥ comes from ≥ 2 eliminations that use the *outer* hypothesis, and in about half the problems the inner ¬ result is reused by IMPE / NEGE. Gate: `necessity.py` finds no proof without the nested box within L + 2 lines. | 25% (62%) | 1%, never necessary | 25–30% |
| T4 | **Reductio with work inside** | Atomic or disjunctive goals proved by reductio. The body holds a nested box (C(N()), C(O(\|))) or ≥ 3 eliminations ending in a clash with the assumed ¬G. Include classical-only theorems. | 20% reductio (38%); 15% classical-only (dev58 26%) | 1%; 4% | 20% reductio; 15–20% classical-only |
| T5 | **Necessary case splits with work in both branches** | A case split inside a →-intro, ¬-intro or reductio box (I(O), N(O), C(O)); some split on an assumption. Both branches ≥ 2 lines. The conclusion is ⊥ (inside N / C), or a compound routed by Or.inl in one branch and Or.inr in the other. Gate: no proof without the split. | 45% split (all needed); 35% both branches ≥ 2 | 3%; 1% | 35–40% |
| T6 | **Elimination chains at depth 2** | ≥ 2 elimination lines (ANDE / IMPE / NEGE) at box depth ≥ 2. Box depth 2 is the norm; depth 3 is rare. | mean 1.95 (3.0); depth ≥ 2 in 75% | mean 0.26; 17% | ≥ 2 on average; depth ≥ 2 in 70% |
| T7 | **Drop generator artifacts in this set** | Consistent premises, no redundant premise, no ~~ in the statement, ex falso ≤ 10%, no lazy (Z > G) premises. | 0% / 0 / 0% / 5% | 33% / 1.19 / 43% / 42% | ≤ 5–10% each |
| T8 | **Biconditional and higher-order statements** | (A>B)&(B>A) as goal or premise; (A>B)>C goals. | 10% / 15% (dev58: 28% / 9%) | 0% / 4% | 10% each |

**How to build and check it.**

- **Construction:** choose the skeleton and box nesting first, then the goal shape. Fill boxes with short
  elimination chains over the box hypotheses and ≤ 2 premises.
- **Gates:** `minproofs.py` (shortest length = constructed length ± 1); `necessity.py` (the target structure is
  needed); and the existing hard12 shape gates (state ≤ 386 tokens).
- **Exclusions:** the textbook72 renaming classes and premise-order keys, as in `pools/hard12/exclude.py`. Never seed
  from textbook formulas.
- **Calibration:** use T0 pass rates as before. Expect the band filter to be less necessary: the within-generator
  rows in §4 show these structures are already harder.

## Caveats

- **Small n.**
  - Twenty textbook problems, eight of them failed. Wilson 95% intervals: 5/8 = 31–86%, 6/8 = 41–93%,
    14/20 = 48–85%, 7/20 = 18–57%.
  - The textbook-vs-pool differences stay large even at the intervals' lower ends, because the pool's rates are 1–8%
    at n = 5,000.
  - The *within-dev* predictors are exploratory (§4).
- **One shortest proof per problem, in a restricted space.**
  - The search ranges over the theorem's subformulas, their negations and double negations; generator rows also get
    their own proof's formulas.
  - The proof found is one shortest proof; others may differ in structure. §5 checks four structures; the rest
    (e.g. lemma reuse, branch routing) are unchecked.
- **The failure label pools four checkpoints** (two think, two no-think) at different budgets. Per checkpoint, TB-fail
  pass@1 is 0–4.7%, so the label is stable across them.
- **The h12 test "fail" threshold** is 50%, not 25%, so that the group is not near-empty.
- **dev58 is now a design set.** Use a fresh textbook-like set, not dev58 or train14, to test a generator built from
  this note.

## train14 (reported separately; not used for conclusions)

- **Scores:** best-4 mean pass@1 33.7%.
- **Lengths:** the exact search finds shortest proofs of 7–12 lines for 9 of 14. One has no proof within 16 lines.
- **Shapes:**
  - classical-only 29% (4/14), biconditional 29%, no premises 21%;
  - nested ¬-box 54% (7/13), reductio 23% (3/13), case split 8% (1/13).

These broadly agree with dev58. Full table in [`tables.md`](tables.md).

## Files and regeneration

The scripts are in this folder, uncommitted, on `leon/thinking` at `0acf62a`.

| file | does |
|---|---|
| [`common.py`](common.py) | parsing; theorem features (G4ip from the fork's `intuit.py`); proof features and box skeletons |
| [`minproofs.py`](minproofs.py) | shortest proofs via `test_21_30/fastlen.fast_minlen`, 8 workers, 3.5 GB cap each |
| [`features.py`](features.py) | one feature row per problem (shortest proof, generator proof, state-env action counts) |
| [`necessity.py`](necessity.py) | restricted re-search (no nested ¬-box / no DN / no case split / depth ≤ 1) |
| [`analyze.py`](analyze.py) | joins the eval rows and writes `tables.md` and `results.json` |
| [`tables.md`](tables.md), [`results.json`](results.json) | every table and number (aggregates only) |
| [`test_gap_features.py`](test_gap_features.py) | regression tests on synthetic theorems |

**Outputs** are in `/scratch/cowifornia/nd-rl/think/runs/gap_analysis/`. The shortest-proof files and
`features.jsonl` are there. **`minproofs_textbook.jsonl` holds proofs of the textbook problems: it is eval-only and
must never enter training data.**

```bash
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
G=code/experiments/current/thinking_traces_20261005/gap_analysis
for s in textbook pool h12test k12; do nice -n 15 code/.venv/bin/python $G/minproofs.py --set $s --workers 8; done
nice -n 15 code/.venv/bin/python $G/features.py --workers 8
nice -n 15 code/.venv/bin/python $G/analyze.py
nice -n 15 code/.venv/bin/python $G/necessity.py --workers 8 && code/.venv/bin/python $G/analyze.py
```

**Run times (2026-10-08, CPU).**

- Textbook search: 6 min, most of it one train14 problem with no proof within 16 lines.
- Pool, test and K12 searches: under 1 min each, with no timeouts. 4 of the 17,000 generator rows had no proof in the
  search space within their generator length, and fall back to their generator proof.
- Features: about 3 min. Necessity check: 2 min.

**Checks run.**

- `test_gap_features.py`: 4 passed. Its fixtures are nd_verify-accepted and are not textbook renamings.
- Every shortest proof was re-verified by nd_verify inside the search.
- A spot check of the skeleton and feature code on the 20 textbook 7–12 proofs, by eye. No proof is printed here.
- No model was run, and no existing file was modified.
