---
written_on: 2026-10-14
written_by: agent:claude
---

# What long_rl_20261014's models can't solve, and what the GRPO wall was

Four analysis agents (A, C on Opus; B, D on Fable), one model each, plus the lead's GRPO-wall analysis.
Models: v2 untied-ei s1 (A), looped-ei s0 (B), untied-ei s0 pretrain vs final (C), all four v2 EI runs (D), v3 GRPO s0/s1 (lead).
Reports: B/REPORT.md, D/REPORT.md; A and C returned theirs inline (summarised here; scripts in A/, C/).
Every solve count is Lean-judged. Proof length = body lines.

## One-line answer

The blockers are specific proof *shapes* that are absent or rare in the ≤ 6-line pretraining data, not proof length.
When a shape is rare (depth-3 boxes, 3.8% of pretraining proofs) RL eventually breaks through; when it is absent
(a nested case split in an Or.elim's first branch, 0 of 2,277 pretraining case splits) RL never samples it, because
the needed token has p ≈ 3e-6 and expert iteration only trains on its own successes.

## The GRPO wall (v3 s0, oracle calls 84-102, +1,771 targets)

- 1,657 of the 1,771 newly solved targets are Dan's ladder *generator* theorems, 7-9 body lines, implication goals.
- Wall proofs: max box depth 3 in 1,582/1,771 (pre-wall proofs: mostly depth 2).
- Ladder targets whose goal nests 3 implications (`A → (B → (C → D))`, 1,805 targets):

| | solved |
|---|---|
| GRPO s0 before call 84 | 270 (15%) |
| GRPO s0 by the end | 1,743 (97%) |
| GRPO s1 (no wall) by the end | 235 (13%) |
| EI s1 | 1,789 (99%), ramped over calls ~5-113 |

- Skill: open a third nested assumption box (often for junk antecedents like `~~~P`) and close all three. Depth 3 is
  in 3.8% of pretraining proofs (depth 4: 0%), always with a trivial interior because of the 6-line cap.
- Once a few depth-3 proofs land, every depth-3 target is solvable (the interior is 1-2 known steps); the ~15-round
  tail is the sampler sweeping 1,500 of 18,439 targets per round.
- v3 GRPO s0 final: dev 73/843 (EI 232-247), old dev 593 (EI 852-929), RL 4,545 (EI 7,020-7,395).

## Taxonomy of what EI can't solve

| # | class | scale | evidence | found by |
|---|---|---|---|---|
| 1 | **Nested case split in the FIRST branch of an Or.elim** | 60% of unsolved 10-14-line targets, 66% at 20-24, 82% at 25-34; 0/7,133 solved at 15+ | Same band and family: 0% vs 96-97% flat. Swap the outer disjuncts (same theorem, same length): 0/480 → 456/480 samples, 0/30 → 30/30 targets (A); 0/80 → 77/80 untied, 0/80 → 73/80 looped, reverse swap 77 → 2 (D). Worst gold token in 30/30 (A) and 143/146 (D) proofs is the `(` opening branch 2, log p ≈ −12.7; argmax `;`. 100% surprised before and after RL (C). | A, C, D |
| 2 | **Box for a formula not in the goal** (`~(A&B)` premise with only A: assume B, derive ⊥, conclude ~B; or a →/¬-intro box proving an intermediate lemma) | d family: 83 of 104 short failures; c family: 499 of 535 lemma-box targets | Same band and family: 15% vs 42% solved (A). Rewrite `~(A&B)` → `A > ~B`: 3/30 → 25/30 (confounded: ~2 lines shorter). Synthetic 9-line family: box proves the goal 68/80 vs a lemma 0/80, looped 72 vs 0 (D). The model forges `⟨a, a⟩` instead. | A, D |
| 3 | **Classical reductio** (excluded middle, Peirce, ¬(A∧B) ⊢ ¬A∨¬B, ¬(A→B) ⊢ A∧¬B, any DN step) | Old dev: EM 19/19 unsolved (all four runs), Peirce 15-16 of 18-20; pool: 262 of 271 DN targets never solved by any run | Synthetic EM 0/80, Peirce 0/80 vs a constructive nested-¬-intro control 66/80 (D). Solved instances have a non-classical route (A). An ex-falso (⊥-elim) line costs 10.9 nats on unsolved targets vs 0.3 on solved ones (B). | A, B, C, D |
| 4 | **Case split not aimed at the current goal** (split to derive ⊥ inside a ¬-intro) | ¬A∨¬B ⊢ ¬(A∧B): 0/18 old dev, 1/40 RL; 0 of 75 solved targets reproduce gold's split-to-⊥ | Detours of +2.8 lines when a detour exists | A |
| 5 | **Family c** (gate DAGs, 13-30 lines) | 3/2,127 RL, 1/135 dev | Currying conjunctive antecedents: 0/480 → 0/480 (A). Over-surprised at building conjunctions (71% vs 0% for s2) (C). Unexplained. | A, C |
| 6 | **Long** (≥ 20 lines, outside 1-5) | 97% never | Diffuse: no single bad token | D |

At 7-14 lines, classes 1-3 cover 605 of 695 never-solved targets with gold proofs (D).
No effect at matched length: premise count, atoms, distractors (the RL pools have none), prompt length, citation
distance, lemma reuse.

**Untied vs looped on class 1.** Untied runs never wrote a nested first branch: 0 of 72,852 RL-found proofs. Looped
runs wrote a few, always a ¬-intro or →-intro box, never a nested Or.elim:
- s0: 99 proofs on 31 targets
- s1: 73 proofs on 28 targets

(Lead's count, reconciling A's "zero" with C's "97".)

## How the model fails (B, looped s0, 65 targets × 16 samples × 1/2/4/8 loops)

- **Circular box (~240 samples, flat in loops).** Opens a box, can't derive what it owes, closes it by citing the
  `have` it is defining. 86/98 unbound-name errors are this (A).
- **Bluffing.** In 74% of ill-typed steps the fact Lean wanted was never derived anywhere. 93% of rejected samples
  still write the goal on the last line.
- **Errors loops reduce:**
  - ill-typed modus ponens: 133 → 61
  - hallucinated names: 68 → 26
  - wrong disjunct: 52 → 35
- **Errors loops increase** (the sample gets further before dying):
  - ramble to cap: 109 → 155
  - one-armed Or.elim: 66 → 102
  - scope leak: 62 → 103
- **Loops: 1 → 2 is the whole gain** (12 → 14 targets; 4 and 8 loops reach 15). The RL-retry advantage is mostly
  k = 24 plus 1 loop being a weak setting for this model.

## What RL taught (C, untied s0, pretrain vs final, first 300 tokens)

- **Learned:**
  - citations: 65-89% → 3-9% surprised
  - case-split hypotheses: 70% → 19%
  - branch closes: 58% → 10%
  - final step: 40% → 0.4%
  - opening branch 2 after a flat branch 1: → 0%
- **Not learned:**
  - opening branch 2 after a nested branch 1: 100% → 100%
  - reductio: 35% → 42%
  - classical contradiction: 94% → 81%
- **Most surprise is noise.** On "which step next" decisions, the model's choice completes to a Lean-valid proof as
  often as gold's (0.92 vs 0.88). On bookkeeping and ex-falso decisions its choice is never valid (0 of ~1,000).
- **Padding.** Restated lines in RL-found proofs grow from 3.4% to 21-23% of body lines (gold: 0%), because EI
  trains on unpruned proofs.

## Fixes, cheapest first

1. **Class 1.** Pick one:
   - randomly swap disjunct order in prompts;
   - seed RL replay with a few hundred to a few thousand nested-first gold proofs;
   - change the Lean text so a box's conclusion follows the box.

   Prediction (A, C): s/s2 at 10-14 lines → ~96%, and the 20-34 band from 23/5,000 toward 8-25%.
2. **Classes 2 and 4.** Add ≤ 6-line pretraining proofs with non-goal ¬-intro boxes and with a case split under a
   ¬-intro.
3. **Class 3.** Short classical-reductio curriculum (EM, Peirce instances).
4. **Padding.** Prune found proofs before EI training.
5. **Loops.** Equal-compute test: 2 loops × 24 samples vs 8 loops × 6. B predicts more samples wins.
