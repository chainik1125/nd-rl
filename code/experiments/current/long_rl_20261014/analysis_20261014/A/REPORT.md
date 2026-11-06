<!-- agent A's final report, saved by the lead from its returned message (2026-10-14). Paths to scratchpad/probe/A/ now live in this directory. -->

## Summary

1. **Biggest failure: a missing case-split branch.** When the first branch of a case split contains another case split, the model finishes the inner split, then writes `; exact … <eos>` and never writes the outer second branch.
   - **Never written:** none of the ~150k Lean-accepted proofs found in RL has this shape, in all four runs (both arms, both seeds).
   - **Same length, nested split first vs second:** 6/151 vs 147/148 (s family, 7–14 lines); 3/259 vs 77/197 (s, 15–19); 0/1,529 vs 92/375 (b, 20–24).
   - **Swap test:** swapping the outer premise's disjuncts (`S v U` → `U v S`) gives the same theorem at the same length. On 30 RL-unsolved targets it goes from 0/480 samples to 456/480, and from 0/30 targets to 30/30.
   - **Likelihood:** in all 30 gold proofs the least likely token is the `(` that opens the outer second branch (log p −12.7; the model's top choice is `;`).
   - **Scale:** this shape covers 60% of unsolved 10–14-line targets, 66% at 20–24 and 82% at 25–34.
   - **Why it persists:** nested case splits never appear in ≤ 6-line pretraining, and expert iteration only trains on its own successes, so it never sees one.
   - **Cheap fix:** randomly swap disjunct order in prompts, or seed the pool with gold proofs of this shape.
2. **It won't open a box for a formula that isn't in the goal.** With a premise `~(A & B)` and only A in hand, it writes `⟨a, a⟩` (the same line twice) instead of "assume B, derive a contradiction, conclude ~B".
   - At the same length, these targets solve at 15% against 42%, and only 1/127 targets needing such a box are solved.
   - Rewriting `~(A&B)` as `A>~B` (equivalent) takes 30 targets from 3/30 to 25/30 (5/480 → 289/480 samples). The rewrite also shortens proofs by about 2 lines, so length is a partial confound.
3. **Classical proof by contradiction is essentially absent.** Excluded middle is 0/19 on old dev in all four runs, and Peirce and the classical De Morgan / negated-conditional schemata cluster in the failures. In the solved instances I read, the theorem happened to have a non-classical shortcut.
4. **It only case-splits toward the current goal.** Of 75 solved targets whose gold proof splits cases to derive a contradiction, it never does so, and detours at +2.8 lines. It fails when no detour exists (¬A∨¬B ⊢ ¬(A∧B): 0/18 on old dev).
5. **Family c is unexplained:** 3/2,127 RL, 1/135 dev. Removing every conjunction-building step (currying `(A&B)>C` to `A>(B>C)`) still gives 0/480.
6. **The dominant error is circular citation:** stuck, the model cites the very `have` it is proving (86/98 of unbound-name errors).

Prompt length, number of atoms, premises and citation distance don't predict failure at fixed length. The RL pools contain no distractor premises.

---

# Report: what untied-ei s1 can't solve, with length held fixed

Model: `long-rl-20261014-v2_untied-ei_s1`. Labels:
- RL pool: a target counts as solved if it appears in `found_targets.jsonl`.
- dev: k = 64.
- old dev: k = 16.

Features come from the gold proof (or Dan's `gen_proof` for ladder and old-dev generator theorems) and from the statement.

## Length is confounded with family; family matters a lot at fixed length

| body lines | s / s2 | d | ladder | c | b / b2 / b3 |
|---|---|---|---|---|---|
| 9 | 1.00 | 0.33 | 0.67 | – | – |
| 12 | 0.93 | 0.06 | 0.12 | – | – |
| 13–14 | 0.80–0.86 | 0.15–0.17 | 0.00 | 0.00 | – |
| 15–19 | 0.20–0.43 | 0.00 | – | 0.00 (n = 1,423) | – |
| 20–22 | – | – | – | 0.00 | 0.03–0.13 |

## Taxonomy

"Strata" below means within (family × body length), using strata fixed effects or stratified rates.

| # | Class | Count | Effect at fixed length | Hypothesis |
|---|---|---|---|---|
| 1 | **Missing outer branch after a nested case split** (s, b, d, e families) | Unsolved RL targets with this shape: 432 (10–14), 397 (15–19), 1,561 (20–24), 2,036 (25–34), 2,703 (35+) | Nested first vs second: s 7–14: 6% vs 99%; s 15–19: 1% vs 39%; b 20–24: 0% vs 25%; b 25–34: 1/1,973 vs 20/237 | The token after a nested first branch was never shown in pretraining or EI (p ≈ 3e-6). Fix: disjunct-swap augmentation, or ~500 gold proofs of this shape. **Tested by the swap: 0/30 → 30/30.** |
| 2 | **Box for a formula not in the goal** (`~(A&B)` premise; ¬-intro whose result isn't part of the goal) | 83 of the 104 short d failures | 15% vs 42% (strata OR 0.13, p 4e-22); non-goal ¬-intro boxes: 0.6% vs 19%; dev: 6% vs 43% | The model only opens boxes it can read off the goal, and fakes the missing conjunct with `⟨a, a⟩`. Non-goal ¬-intro boxes are in 0.025% of pretraining proofs. **Tested by the rewrite: 3/30 → 25/30.** |
| 3 | **Classical reductio** to a non-negated goal (excluded middle, Peirce, ¬(A∧B) ⊢ ¬A∨¬B, ¬(A→B) ⊢ A∧¬B) | Old-dev failures: EM 19/19, Peirce 16/18, ¬∧→∨ 15/19, ¬→ 15/18 (65 of the 121 textbook failures). Short RL failures: EM 39/40, Peirce 34/40 | Schema-level | Reductio via double negation is in pretraining (6.8%), but only in a 4-line shape. The missing step is re-deriving the goal from a lemma proved inside the reductio. Prediction: the solved instances are exactly the ones with a non-classical proof. |
| 4 | **Case split that isn't aimed at the current goal** (split to derive a contradiction under ¬-intro, or to an intermediate result) | ¬A∨¬B ⊢ ¬(A∧B): 0/18 on old dev; dist-or-over-and-conv: 8/16 unsolved; 0 of 75 solved targets use this shape | Strata OR 0.55 (p 0.07); when a detour exists it costs +2.8 lines | When the split has to sit under ¬-intro, the model applies the disjunction as if it were a function. |
| 5 | **Family c** ("gate" DAGs: goal is a 1–3-deep implication chain, conjunctive-antecedent premises, long-range reuse) | 2,124/2,127 RL, 134/135 dev | 0% at every length from 13 to 30 | Not the conjunction-building steps (currying gives 0/480). Errors: circular citation 32%, ill-typed modus ponens 41% after currying, stall loops 22%. Open guess: a ¬-intro box or case split needed inside a branch under deep implication boxes. |
| 6 | **Two case splits in general** (s family) | 12–14 lines: 98% with one split vs 51% with two | OR 0.20 per SD of case-split count | Almost entirely class #1: splits nested only in the second branch solve like no nesting. |
| 7 | **Residual generator theorems**, 6–11 lines | 58 old-dev gen failures: 26 solved by all three other runs (sampling noise), 14 by none | Per SD: ¬-intro count OR 0.49–0.60, case-split count OR 0.51–0.59, goal size OR 2.4–2.6 (big goals decompose easily) | Same mechanisms as #2–#4, plus some degenerate theorems with huge repeated subformulas. |

## Examples

- **#1, rl_2761 (13 lines):** `S v U, S>(W v Z), S>(W>Q), U>Q, S>(Y>Q), Z>Y ⊢ Q`. The model writes the S branch, including its inner split, then `… exact n27 ) ; exact n20 <eos>` without the one-line U branch. On 12 such targets, 173/192 samples fail this way. Two more: rl_2313, and `S>(R v X), S v W, S>(X>T), S>(R>T), W>T ⊢ T` (0/16, all with the same parse error).
- **#2, rl_13308 (9 lines):** `~Q>R, ~(~S&Q), ~S v T, T>R ⊢ R`. The model writes `⟨n32, n32⟩ : (¬S ∧ Q)`. Same pattern in rl_13684 (`⟨n6, n6⟩ : S ∧ W`) and rl_926.
- **#3:** `⊢ (~S v Q) v ~(~S v Q)` gets 0/16. On `((~Q > ~(R>S)) > ~Q) ⊢ ~Q` it builds the lemma before opening the goal's `Q` box, where the lemma isn't provable.
- **#4:** `(~(~R&(R&Q))) v (~((QvR)&R)) ⊢ ~(…)`: the model writes `n13 n17`, applying the disjunction as a function. Detour example: rl_8442, gold 11 lines, model 13.
- **#5, rl_10958:** `U>~T, S>R, T>~R, S v U ⊢ P>(Q>~T)`. It applies `U>~T` to S instead of opening the ¬-intro box. In rl_7120 it repeats `have : S→Q` about 20 times.

## Detours on solved targets

Model's shortest proof minus the gold minimum, on targets with an exact minimum:

| Excess lines | Targets |
|---|---|
| 0 | 1,878 |
| +1 | 558 |
| +2–3 | 669 |
| > 3 | 162 |

- Mean excess by family: s 0.8–0.95, d 1.2.
- Splits to a contradiction: never reproduced (+2.8 lines).
- Conjunction-against-`~(A&B)` targets: same route as gold 65/66 times. So class #2 is all-or-nothing, not a detour.
- Reductio: the model uses it in only 10 of 168 solved targets whose gold proof does, taking a non-classical route when one exists.

## Experiments (CPU, k = 16, Lean-judged)

| Experiment | Before | After |
|---|---|---|
| Swap outer disjuncts, 30 s targets | 0/480 | **456/480** |
| `~(A&B)` → `A>~B`, 30 d targets | 5/480 | **289/480** |
| Curry `(A&B)>C`, 30 c targets | 0/480 | 0/480 |

Likelihood probes:
- **Nested-split gold proofs:** in 30/30 the worst token is the `(` that opens the second branch. Log p is −14.7 before RL and −12.7 after.
- **Excluded-middle proof:** total log p is −91 before RL and −46 after. The worst token is `Or.inr` (log p −11), where the model prefers another `Classical.byContradiction`.

Caveats:
- **RL labels depend on exposure.** At 15–19 lines the s family solves about 30% in RL but 60–100% on dev at k = 64. Within-band comparisons share one sampling policy and are fair; cross-band ones aren't.
- **Gold features describe one proof.** The `gen_proof`s are not minimal.
- **Branch order is fixed.** `Or.elim` takes branches in the premise's disjunct order. So in the generator families a nested-first gold proof means any proof without an extra lemma is nested-first.
- **The error classifier only localizes.** `errs.py` names the first suspicious line of a sample Lean already rejected; Lean is the only judge.

## Next on a GPU

1. **Fix #1 and measure the frontier.** Add disjunct-swap augmentation, or seed EI with gold nested-first proofs.
   - Predict that the 20–34 band rises from 23/5,000 toward 8–25%, and that the dev frontier moves past 26 lines.
   - The looped arm also has zero proofs of this shape, so check it too.
2. **#2 and #4:** add ≤ 6-line pretraining proofs that open a ¬-intro box for a non-goal formula, and that split cases under ¬-intro. Predict d at 10–14 lines rises from 28% toward the s-family rate.
3. **#3:** compare solved instances against an intuitionistic prover over the same formula closure. If they match, add short reductio-to-a-disjunction proofs.
4. **#5:** run a length-matched rewrite that removes the ¬-intro box inside a branch (give `S>~T` directly) to see whether c is really class #2 in disguise.

Files are in the work dir:
- **Features:** `feat.py`, `pre.py`, `addflags.py` → `feats.jsonl`, `feats2.jsonl`.
- **Analyses:** `an*.py`.
- **Interventions:** `mkint.py`, `mkswap.py`, `samp.py` → `int_*.json`, `res_*.json`, `log_*.txt`.
- **Error classifiers:** `errs.py`, `unb2.py`.
- **Likelihood probes:** `nest_lp.py`, `em_lp.py`, `em_show2.py`.