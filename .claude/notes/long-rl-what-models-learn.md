---
written_on: 2026-10-14
---

# What the long-RL models learned, and what they can't solve (2026-10-14)

Experiment: `code/experiments/current/long_rl_20261014/`. Pretraining saw only proofs of ≤ 6 body lines. RL then
targeted proofs of 7–80 body lines, judged by Lean only.

Analysis: `analysis_20261014/` in that directory.
- `SYNTHESIS.md` holds the full tables.
- Agents A–D each have a directory with a `REPORT.md` and the analysis code.
- `wall/` holds the GRPO-wall scripts.
- `ndmodel.py` loads any checkpoint on CPU, scores gold proofs and samples Lean-judged proofs.
- Large derived files (> 5 MB) were not kept; the scripts regenerate them.

Four analysis agents each looked at one model:
- A: untied-ei s1
- B: looped-ei s0
- C: untied-ei s0, before and after RL
- D: minimal pairs across all four v2 EI runs

Robbie decided not to start new runs on these findings (2026-10-14).

## Headline

**The blockers are proof shapes, not length.** Each one is rare or absent in the ≤ 6-line pretraining data.
- **Rare shapes, RL breaks through.** Depth-3 boxes are in 3.8% of pretraining proofs, and RL eventually learns
  them.
- **Absent shapes, RL never does.** No pretraining proof has a case split nested inside the first branch of
  another case split (0 of 2,277 case splits). The token RL would have to sample has p ≈ 3e-6, and expert
  iteration (EI) trains only on its own successes.

At matched length, these don't predict failure: premise count, atoms, distractors, prompt length, citation distance.

## What was learned

- **GRPO's "wall" (v3 s0, oracle calls 84–102, +1,771 targets) was learning depth-3 nested assumption boxes.**
  - The targets were Dan's ladder generator theorems with goals like `A → (B → (C → D))`. Solve rate on the 1,805
    such targets:

    | run | solved |
    | --- | --- |
    | GRPO s0, before the wall | 15% |
    | GRPO s0, after | 97% |
    | GRPO s1 (never crossed) | 13% |
    | EI s1 (gradual, oracle calls ~5–113) | 99% |

  - It looked like a cliff because two things combined. One rare skill unlocked a whole family whose interior is
    1–2 known steps. The sampler then swept 1,500 of the 18,439 targets per round.
- **What EI learned versus pretraining** (C). The metric is the share of gold-proof decision tokens with p < 0.2,
  before RL → after RL.
  - Citations: 65–89% → 3–9%.
  - Case-split hypotheses: 70% → 19%.
  - Branch closes: 58% → 10%.
  - Final step: 40% → 0.4%.
  - Opening the second branch after a flat first branch: → 0%.
  - Not learned: a nested first branch (100% → 100%), reductio (35% → 42%), classical contradiction (94% → 81%).
- **RL changed how the model fails** (B). Pretrained-only models fail on local type errors. RL'd models fail by
  planning: rambling to the token cap, one-armed case splits, reusing facts across branches.
- **EI teaches padding** (C). Restated lines in found proofs rose from 3.4% to 21–23% of body lines; gold has 0%.
  The cause is that EI trains on unpruned proofs.
- **Gold-proof −log p on dev falls under EI** (1.10 → 0.35 for v4 s0). GRPO's fall is smaller and varies by seed.
  The per-token average hides single blocking tokens at log p ≈ −12.

## What can't be solved (taxonomy)

1. **Nested case split in the first branch of a case split.**
   - Scale: 60% of unsolved 10–14-line targets, 66% at 20–24, 82% at 25–34.
   - At matched band and family: 0% solved with this shape, against 96–97% for flat proofs.
   - Swap test: swapping the outer disjuncts gives the same theorem at the same length, and takes 0/30 → 30/30
     targets (A) and 0/80 → 77/80 samples (D).
   - The worst gold token is the `(` that opens branch 2 (log p ≈ −12.7; argmax `;`). It looks like a learned
     bracket-counting habit of the Lean-text format.
   - Untied runs never wrote this shape (0 of 72,852 found proofs). Looped runs wrote ¬-intro boxes there on about
     30 targets per seed, and never a case split.
2. **A box for a formula that isn't in the goal.**
   - Example: given `~(A&B)` and A, assume B to derive ~B. The model forges `⟨a, a⟩` instead.
   - Synthetic: 68/80 when the box concludes the goal, 0/80 when it concludes a lemma.
3. **Classical reductio.** Excluded middle is 0/19 on old dev in all runs. Synthetic excluded middle and Peirce are
   each 0/80. A ⊥-elimination line costs 10.9 nats on unsolved targets against 0.3 on solved ones.
4. **Case split aimed at ⊥ rather than the goal.** `¬A∨¬B ⊢ ¬(A∧B)` is 0/18.
5. **Family c** (gate DAGs, 3/2,127 solved). Unexplained: currying the conjunctions doesn't help.
6. **≥ 20 lines, outside classes 1–5.** 97% are never solved, with diffuse failures.

The commonest failure is a **circular box**: the model closes a box by citing the `have` it is defining.

Loops: going from 1 to 2 loops is nearly the whole gain. The looped model's RL-retry advantage comes mostly from
k = 24 plus 1 loop being a weak setting.

## Background needed to read these numbers

- **Ladder targets** are 4,493 of the RL pool. They come from Dan's ladder-A build (`$ND_FORK/data/ladder/POOLS.md`)
  and have no gold proofs. Their `body` = certified minimal length − premises.
  - 3,735 are generator theorems; EI solves 99% of them.
  - 760 are textbook schema instances from Dan's `textbook_pool.py` (19 schemata × 40, random subformulas). The
    unsolved ones are concentrated in excluded middle, Peirce, negated conditional and two De Morgan directions.
- **Old dev** is the sibling transfer pool from the same build.
- **Seed spread** of one 3 h EI run is about ±30 dev theorems: v4 s1 got 186 and v2 s1 got 247 with the same
  recipe.

## Fixes proposed (not run)

1. Disjunct-swap augmentation, or replay of nested-first gold proofs. A and C predict s/s2 at 10–14 lines → ~96%,
   and the 20–34 band from 23/5,000 → 8–25%.
2. Short pretraining proofs containing non-goal ¬-intro boxes and case splits under ¬-intro.
3. A classical-reductio curriculum.
4. Prune found proofs before EI trains on them.
5. Equal-compute test: 2 loops × k = 24 against 8 loops × k = 6.

## Mechanism of the GRPO wall (model diffing, 2026-10-15)

[experiment-summaries/2026-10-14-grpo-wall-model-diff](../../experiment-summaries/2026-10-14-grpo-wall-model-diff/README.md).
- The depth-3 blocker is closing, not opening. Every pretrained seed closes a run of boxes with the shortcut "depth after
  the r-th consecutive close = max(0, 2 − r)", which pops one level too far on depth-3 proofs. That is the first error in
  93-95% of failed depth-3 samples.
- GRPO s0 and EI fixed it with a distributed change to blocks 0-2. GRPO s1, v4 s0 and v4 s1 never did.
- Crosscoders find no new features. A diff-SAE's top 20 latents ("first close of a run that started at depth ≥ 2")
  transfer most of the fix.
