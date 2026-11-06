<!-- agent C's final report, saved by the lead from its returned message (2026-10-14). Paths to scratchpad/probe/C/ now live in this directory. -->

## Summary (untied-ei seed 0, before RL vs after RL)

**What I ran.**
- Scored the gold proofs of all 843 dev targets and 900 RL targets per band (7,080 proofs) under both checkpoints, then mapped every token to its line and step in the proof (rule, box opening, citations).
- A "decision point" is a token where the gold choice has probability below 0.2.
- To separate free choices from real errors, I took 428 decision points in bands 7-19. At each I forced the model's preferred token and, separately, the gold token, then sampled 3-4 completions and had Lean judge them.

**Findings.**
1. **Most surprise is noise.** Choosing which elimination step or goal case split comes next is surprised 25-57% of the time, equally on solved and unsolved targets. Forcing the model's choice works about as well as gold (0.92 vs 0.88 Lean-valid completions; 0.82 vs 0.79 for goal splits).
2. **Bookkeeping surprises are fatal and pick out unsolved targets.** These are rule heads, cited names, box hypotheses, box-closing names and opening the second branch of a case split. The model's alternative was never Lean-valid (0 of about 1,000 samples). They are surprised 0.3-1.6% of the time on solved targets and 9-34% on unsolved ones.
3. **The wall: a nested box inside the first branch of a case split.** 53% of targets with a gold proof have this shape. On those, the model wants `;` where branch 2 opens (`(`) 100% of the time, before RL and after. With a flat first branch RL brought that from 27-84% down to 0%.
   - Within the same band and family (s/s2 at 10-14), solve rates are 97% flat, 96% nested elsewhere, and 0% with this shape.
   - At 15+, 0 of 7,133 such targets were solved.
   - The RL-found proofs contain 0 of 37,462 with this shape, so expert iteration never trains on it.
   - This agrees with D.
4. **Ex falso is a strategic error.** Where gold derives the goal from False, the model's own choice is 0 of 80 Lean-valid against 46 of 80 for gold. This agrees with B, and overlaps with item 3.
5. **The c family (premise gate DAGs) fails regardless of structure** (0.1-0.6% solved at 15-19). It is over-surprised at building a conjunction (71% vs 0% for s2) and at citing implications (21% vs 4%). This is a hypothesis only; I didn't run the forced-token test on it.
6. **What RL learned:** citations dropped from 65-89% to 3-9% surprised, case-split hypotheses 70% to 19%, branch closes 58% to 10%, the final step 40% to 0.4%. **What it didn't:** nested first branches stay at 100%. Case splits that prove an intermediate lemma got worse (90% to 99.5%), but the model routes around them. Reductio (42%) and classical contradiction (81%) also stayed high.
7. **RL teaches padding.** Lines that restate something already in scope rose from 3.4% to 21-23% of the body lines in RL-found proofs; gold has 0%. Expert iteration trains on unpruned proofs.

**Caveats.** One seed. Before-vs-after comparisons use only the first 300 proof tokens, because the pre-RL model is also off-distribution on length.

---

# Full report: where in a gold proof is the model surprised, and what did RL teach it?

## Method
1. **Teacher forcing** (`score.py`, about 5 min per checkpoint on CPU).
   - 7,080 gold-bearing proofs; ladder targets are excluded because they have no gold proof.
   - `annot.py` mirrors the tokenizer's `proof_tokens` and reproduces `encode_proof` exactly (3,000 of 3,000 checked).
   - `events.py` groups tokens into decision events: which formula to assert next (`WHAT`), rule and citations (`HOW`), box hypothesis (`HYP`), the box's closing `exact nE` (`CLOSE`), the final step (`FINAL`), and the `(` opening branch 2 of a case split (`BOX2_OPEN`).
   - An event is a decision point if the gold token has p < 0.2.
2. **Ordering test** (`order.py`, `order2.py`).
   - At 2,192 surprised `WHAT` events (after RL) and 1,427 (before RL), I forced the model's top token and greedy-decoded the rest of the line.
   - Each alternative line is classed as a later gold line, a restatement of something in scope, a jump to the box goal, out of scope, or novel.
3. **Free vs strategic test with Lean** (`run_alt.py`). 428 events (203 solved, 225 unsolved). From each point I sampled 3-4 completions after forcing the model's token and after forcing the gold token, and Lean judged the full proofs.
4. **Structure census** (`q8.py`) of the gold proofs of 14,789 targets and of 37k RL-found proofs per run.

## Taxonomy

| Class | Surprise after RL | Solved vs unsolved | Model's choice vs gold (Lean-valid, solved targets) | Verdict |
|---|---|---|---|---|
| A. Which elimination step next (implication, and-elimination) | 25-57% | no difference (19/18%, 55/54%, 31/34%) | 0.92 vs 0.88, 0.72 vs 0.81, 0.46 vs 0.74 | noise: ordering |
| B. Case split on the goal | 31% | 63% vs 54% | 0.82 vs 0.79 | noise |
| C. Nested box in the first branch of a case split | `(` opening branch 2: 100% (1,299/1,299) | 0% solved at 15+ (0/7,133) | 0 valid alternatives | the wall |
| D. Bookkeeping after an inner box | box-closing name 25-66% vs 0-2% in flat boxes; later-branch hypothesis 39-41% at depth 2 | 1-2% vs 25-41% | 0 of about 200 | fatal token errors |
| E. Ex falso and reductio | ex-falso citation 0.2% (solved) vs 20% (unsolved) | partly class C | 0.00 vs 0.58 | strategic |
| F. Gate assembly (c family) | building a conjunction 71% (s2: 0%), citing implications 21% (s2: 4%) | c solved 0.1-0.6% | not tested | hypothesis |

**Class C solve rates by structure** (same band and family):

| Band, family | Flat | Nested elsewhere | Nested in branch 1 |
|---|---|---|---|
| 10-14, s2 | 96.7% | 96.9% | 0% (93) |
| 10-14, s | 97.1% | 96.1% | 0% (70) |
| 15-19, s2 | 50% | 37% | 0% (154) |
| 15-19, s | 47% | 44% | 0% (115) |
| 20-24, b2 | 8% | 24% | 0% (838) |

- On the opening `(` of branch 2, surprise stays at 100% before and after RL when branch 1 contains a box. With a flat branch 1 it went from 27-84% to 0%.
- RL-found proofs with this shape: 0 for untied s0, 0 for s1, 97 for looped s0.

Examples (`[[gold p=… want X]]` marks p < 0.2; `want X` is the model's top choice):
- `dev_724` (body 9, unsolved): `have n5 : S := Or.elim n3 ( fun ( n6 : U ) => by have n7 : ( ¬ W ) := ( fun ( n8 : W ) => … ) ; have n11 : S := n1 n7 ; exact n11 ) [[( p=0.00 want ;]] fun ( n12 : T ) => …`
- `rl_12542` (body 9): `exact [[n7 p=0.07 want n5]] ) [[( p=0.00 want ;]]`. The model wants to cite `n5`, the outer step that is still being written and not in scope.
- `rl_11589`: the nested box in branch 1 is itself a case split, with the same `(`-vs-`;` failure, and then it picks the wrong disjunct for branch 2.

**Hypothesis C.** In the Lean text the model writes, a box's conclusion is stated before the box. The model treats an inner box's closing `)` as the end of the whole case split. It has never seen otherwise: pretraining has case splits in 73 of 5,000 proofs and this shape in none, and RL produced none. **Fix:** add gold proofs with this shape to RL replay, or move box conclusions after their boxes. **Prediction:** branch-2 opening surprise drops below 10% and solves in s/s2 at 10-14 approach 96%.

**Class E.** The model's alternative to ex falso failed in 0 of 80 samples against 46 of 80 for gold. A related shortcut: after writing `have n7 : Y :=` the model applies an implication whose antecedent isn't available (`n2 n1`), and Lean rejects it.

## What RL learned (before → after, first 300 tokens)

**Learned:**
- citations: 65-89% → 3-9%
- case-split hypotheses: 70% → 19%
- branch closes: 58% → 10%
- final step: 40% → 0.4%
- asserting negation-elimination results: 97% → 2%
- opening branch 2 when branch 1 is flat: → 0%
- closing implication boxes: 86% → 20%

**Not learned:**
- nested-first-branch case split: 100% → 100%
- case split that proves an intermediate lemma: 90% → 99.5% (but solves are unaffected: 91% vs 83%)
- reductio: 35% → 42% (on solved targets the model's own route beats gold, 0.62 vs 0.02)
- classical contradiction: 94% → 81%
- closing reductio boxes: 83% → 41%

Before RL, surprise also rises with position (2.7% → 14% → 26% by token 200+), so these comparisons use only the first 300 tokens.

## Side finding: padding
- Restated lines in RL-found proofs rose by RL round: 3.4% → 14% → 18% → 21-23%. Gold has 0%.
- At a decision point, the model's preferred line restates something already in scope 28% of the time after RL, against 3% before.
- 59% of found proofs are longer than their pruned form, by +1.0 line on average.
- Fix: prune proofs before expert-iteration training.

## Next runs on a GPU
1. Replay 2-5k gold proofs with a nested first branch during RL.
2. Change the text format so box conclusions follow their boxes.
3. An RL arm that trains on pruned proofs.
4. A forced-ex-falso rescue probe at scale.
5. The free-vs-strategic test on the c family.

## Files
All in `/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe/C/`:
- token annotation: `annot.py`, `events.py`
- scoring: `score.py` → `scores.pkl`
- event table: `analyze.py` → `events_df.pkl`
- ordering test: `order.py`, `order2.py`
- Lean free-vs-strategic test: `alt.py`, `run_alt.py` → `alt_final_*.pkl`
- solved vs unsolved at matched band and family: `q4.py`
- structure census: `q8.py`
- surprise by structure: `q6b.py`, `q6p.py`, `q11.py`
- lemma vs goal case splits: `q10.py`
- padding: `redund.py`
- proof printer with surprises marked: `show.py`