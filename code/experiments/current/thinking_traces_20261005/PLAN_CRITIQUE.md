---
written_on: 2026-10-08
written_by: agent:claude
papers:
  - yueDoesReinforcementLearning2025
  - poesia2024learningformalmathematicsintrinsic
  - zomboriFindingLongerProofs2021
  - liuProRLProlongedReinforcement2025
---

# Critique of the planned changes: honest-search traces, effort control, harder problems, curriculum

Leon asked for this review on 2026-10-08. It audits the six planned changes listed in the brief:
1. honest-search traces;
2. the budget ladder;
3. model-decided effort;
4. harder 7–12-line problems;
5. the curriculum;
6. traces on the harder problems.

It builds on [LIT_CRITIQUE.md](LIT_CRITIQUE.md) and [RM_CRITIQUE.md](RM_CRITIQUE.md) and does not repeat them.

**What I read.**
- Notes: [RL_RESULTS.md](RL_RESULTS.md), [matched_eval/final_tables.md](matched_eval/final_tables.md),
  [TRACE_AUDIT.md](TRACE_AUDIT.md), [graded_traces/README.md](graded_traces/README.md), [ADAPTIVITY.md](ADAPTIVITY.md),
  [GRPO_COLLAPSE_AUDIT.md](GRPO_COLLAPSE_AUDIT.md), [pools/hard12r/README.md](pools/hard12r/README.md),
  [PROPOSAL.md](PROPOSAL.md), the two earlier critiques, [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md), and the two
  earlier literature reviews under `docs/literature/`.
- Code: `code/thinking/progtrace.py` and `code/thinking/grpo.py` (the `Curriculum` class).
- Branch `leon/thinking`, worktree `/scratch/cowifornia/nd-rl/think/nd-rl`, HEAD `0acf62a` when I started.
- When I started, `collapse_audit/` and `gap_analysis/` held scripts only, and `code/thinking/searchtrace.py` did not
  exist.
- By the end of this review, three more items had appeared, and I read them:
  - [gap_analysis/README.md](gap_analysis/README.md), committed in `864ee17`;
  - the scripts of a proof-first generator in `hard712/`;
  - `code/thinking/searchtrace.py`, whose docstring I read.
- §3.1 and §3.4 take these into account. Otherwise this is a critique of the plan, not of the new code.

**What I ran.** CPU analysis of stored eval rows, GRPO metrics and EI found-attempt files only; method in §7. No
training, eval or GPU. I edited nothing except this file.

**Labels.** Observed: computed here from stored files. Reported: what a paper states (§6 gives read depth).
Interpretation: my reading, which may be wrong.

## 1. Verdict

The plan puts most of its effort into what the think block says (honest search) and how long it is (ladder, auto
effort). The stored evals point at three other bottlenecks:
- **invalid single steps;**
- **a training pool with no signal left;**
- **GRPO narrowing coverage.**

Five recommendations (evidence in §2, details in §3):
1. **Diagnose step validity before redesigning the thinking.**
   - *Observed:* on textbook 7–12 and 13–20, one invalid step ends 60–92% of all attempts, which is essentially every
     failure:
     - either the environment rejects a step (21–67% of attempts);
     - or Lean rejects a proof the environment accepted (6–54%), almost always for a step-local type error.
   - No attempt in these cells fails by reaching the step cap. The only exception is seed 0's runaway actions.
   - *Interpretation:* thinking-as-search addresses the choice of reduction. The data show the model failing to
     *execute* a step: copying long formulas, citing names, typing terms.
2. **Drop model-decided effort (change 3).**
   - *Observed:* a per-problem oracle that picks the best level from held-out attempts beats the best fixed level by
     only +0.3 to +2.2 pp pass@1 in 8 of 9 cells (−1.4 pp in the ninth), though with 30–65% fewer tokens.
   - A free cascade with Lean as the verifier (e0, then e64, then e256) already matches or beats the best fixed level
     at matched tokens per problem from 2,000 tokens on, and matches the oracle there too.
   - Only at about one attempt per problem (1,000 tokens) does oracle allocation beat the cascade, by 2–4 pp. A learned
     controller would get only part of that.
   - The model already chooses its own length at B256: fewer than 0.5% of blocks are force-closed.
3. **Fix the pool before the sampler.**
   - *Observed:* the GRPO learnability weights prefer an in-band target only about 3.4× over a solved one.
   - With 4 budgets, about half of each late iteration's draws are pairs never drawn before.
   - Only 5.6% of hard12 test targets are still in the (0.1, 0.7] band after GRPO.
   - 13–15-line problems from the existing composition pipeline (the one that built the sealed 13–20 and 21–30 sets)
     are in band 2–4× as often as fresh 7–12 candidates. Growing proof length with that pipeline is the cheapest
     supply.
4. **Base the new generator on structural evidence, not on which textbook problems fail, and find a fresh
   textbook-like test.**
   - *Observed:* textbook72's 7–12-line band is 20 problems; 5–11 of them are unsolved in any given checkpoint and 2 in
     all 44 cells. Two GRPO seeds read 39.7% and 22.8%.
   - The gap analysis (now committed) found its strongest results without failure labels. In the pool, 57% of the
     "7–12-line" problems have a shortest proof of 6 lines or fewer, against 0% for textbook problems, and nested
     negation work is needed in textbook problems but avoidable in the pool.
   - The failure-label contrasts it also used, about 45 features on 20 problems, survive no multiple-comparison
     correction. They add tuning without adding information.
   - It flags dev58 itself as a design set now. A generator built from it needs a test set other than textbook72.
5. **Gate the honest-search generator on CPU checks and give it a matched no-think control.**
   - In propositional logic a truth-table check certifies any dead goal immediately, so a search with that check
     rarely backtracks. The new generator does exactly that: it follows the first candidate whose holes are all valid.
   - Non-ex-falso dead ends occur in about 12% of decision blocks of the existing K12 traces.
   - "To Backtrack or Not" finds that, for shallow search, trace training loses to parallel sampling of direct
     solutions at matched compute.
   - The found proofs change the actions, so a T0 trained on the same found actions is required. Otherwise any
     result confounds thinking with proof style.

**Single experiment to run first:** a logical-arm read of the existing it78/r2 checkpoints on 13–20 and textbook72 at
matched tokens, about 3 GPU-h, eval only (§4.5). It decides whether the bottleneck is executing steps or choosing them,
which sets how the search traces should be designed.

## 2. What the stored data say about the bottleneck (observed, new)

### 2.1 Out-of-distribution failures are invalid single steps, not failed searches

Share of all attempts, %. Plain arm, temperature 0.8. Stored rows under `runs/eval/matched/`. The 13–20 rows are the
1,000-item subsample, which includes the 14 items RL_RESULTS excludes.

| set | checkpoint | solved | env rejects a step | env accepts, Lean rejects | action never ends (> 512 tokens) |
|---|---|---|---|---|---|
| hard12 test | GRPO no-think s0 it78 | 92.8 | 2.7 | 4.5 | 0.0 |
| textbook dev, 7–12 lines (20 problems) | GRPO no-think s0 it78 | 39.7 | 41.2 | 19.0 | 0.1 |
| same | GRPO no-think s1 it78 | 22.8 | 23.1 | 54.1 | 0.0 |
| same | EI no-think r2 | 39.8 | ~21.5 | 38.6 | ~0.1 |
| same | GRPO thinking s1 it78, B256 | 33.6 | 51.7 | 13.2 | 1.5 |
| 13–20 | GRPO no-think s0 it78 | 10.6 | 65.2 | 24.2 | 0.0 |
| 13–20 | GRPO thinking s1 it78, B256 | 10.1 | 66.5 | 22.4 | 1.0 |
| 13–20 | GRPO thinking s0 it78, B256 | 6.2 | 41.2 | 6.4 | **46.1** |
| 13–20 | EI no-think r2 | 7.7 | 47.7 | 44.5 | 0.0 |

**What the rejected steps are** (% of all attempts):
- **13–20, no-think s0:**
  - `binder mismatch` 32.9: a box or case-split binder whose formula is not the antecedent or disjunct, i.e. a
    formula-copy error;
  - `unbound` name 14.5;
  - `Or.elim` on a non-disjunction 5.0;
  - malformed term 4.9.
- **13–20, thinking s1:** unbound 20.4, malformed formula 16.6, malformed `have` header (`assign`) 11.3.
- **Textbook 7–12, no-think s0:** unbound 13.6, `Or.elim` on a non-disjunction 13.4, binder mismatch 5.9.

**Why Lean rejects proofs the environment accepted.** The Lean gate's prefilter classes (`lean_gate.leanrej.jsonl`)
are step-local type errors:
- on 13–20, no-think s0: 1,091 of 1,891 rejections (58%) are `app-arg`, an application whose argument has the wrong
  type;
- the rest: `function-expected`, `reiterate`, `and-elim`, `by-contradiction`, `or-intro-target`.

**Where in the proof attempts fail.** Hazard = failures at step s ÷ attempts reaching step s. Steps include premise
restatements.
- **13–20:** from step 4 on, the per-step hazard is 3–8% and roughly flat:
  - no-think s0: 6.9% at step 4, 3.4–4.9% at steps 8–16, 6–7% at 18–22;
  - thinking s1: 7–8% from step 8;
  - EI: 6.3% at step 4, then 2–3%.
- About a quarter of failures fall in each third of the reference length. 18–28% come after the attempt has already
  run past the reference length.
- **hard12 test:** the hazard is ≤ 0.5% up to step 10 and rises to 6.3% at step 16. 46% of failures come after the
  attempt passed its reference length, i.e. on detours.
- **No cell above has any attempt ended by the step cap.**

*Interpretation:*
- Out of distribution, the policy dies by compounding per-step invalidity: copying long formulas, citing names that
  are not in scope, and writing ill-typed terms. This is LIT_CRITIQUE C5's s^L argument, now with the failure classes
  measured.
- A wrong high-level choice can surface as an invalid step later, so these rows cannot fully separate "no viable
  plan" from "botched execution". The logical-arm read in §4.5 separates them.
- Search traces mainly teach which reduction to take. They do not target what is failing.

### 2.2 Seed 0's 13–20 collapse shows up as runaway actions

- *Observed:* the seed-0 thinking GRPO model at it78 writes an action that never reaches `<eos>` within 512 tokens:

  | set | budget | share of attempts |
  |---|---|---|
  | 13–20 | B256 | 46.1% |
  | 13–20 | B0 | 13.7% |
  | textbook72 | B256 | 33.4% |
  | hard12 test | B256 | 1.6% |

  - Seed 1 at it78: 1.0% on 13–20.
  - Graded + LCPO at it78: 12.4% on 13–20 at B256.
  - One budget at it78: 4.7% on 13–20.
- `collapse_audit/` is investigating this; its scripts include `colon_loops.py`. I do not diagnose it.
- *Interpretation:* an in-distribution metric shows 1.6% while OOD shows 46%. Any planned change that lengthens
  generations needs a fixed OOD probe logged during training, with failure classes broken out (§4.1). That applies to
  search traces, e1024 and auto levels.

### 2.3 Headroom for adaptive effort is small, and Lean already gets most of it for free

Per-problem headroom, from stored B0/B64/B256 rows (`oracle_effort.py`, §7):
- **Split-half oracle:** pick each problem's level on its even-numbered attempts, maximising pass rate, and score it
  on the odd ones. This is an approximate upper bound for any controller that reads only the state.
- **Full oracle:** choose and score on all attempts, so it is optimistic.

pass@1 % / tokens per attempt:

| set | checkpoint | best fixed level | split-half oracle | full oracle |
|---|---|---|---|---|
| hard12 test | thinking s1 it78 | 94.4 / 256 (B256) | 94.7 / 180 | 96.5 / 184 |
| hard12 test | graded it78 | 93.7 / 286 (B64) | 95.1 / 185 | 96.7 / 193 |
| textbook72 | thinking s1 it78 | 30.1 / 295 (B64) | 31.6 / 195 | 32.2 / 194 |
| textbook72 | graded it78 | 34.0 / 601 (B256) | 35.9 / 233 | 36.6 / 274 |
| textbook72 | EI thinking r2 | 31.6 / 905 (B256) | 33.8 / 429 | 34.6 / 439 |
| 13–20 | thinking s1 it78 | 10.3 / 600 (B64) | 10.8 / 359 | 14.2 / 382 |
| 13–20 | graded it78 | 12.8 / 729 (B64) | 13.8 / 423 | 18.0 / 463 |
| 13–20 | thinking s0 it39 | 11.9 / 835 (B256) | 13.0 / 418 | 17.2 / 455 |
| 13–20 | EI thinking r2 | 11.4 / 1,448 (B256) | 10.0 / 525 | 14.0 / 589 |

**The cascade.** At a fixed per-problem token budget, attempt at e0, then e64, then e256 (repeating e256), and stop
at the first Lean-verified success. Compare that with i.i.d. attempts at the best fixed level. Monte Carlo over stored
attempts, 40 resamples per problem; success within budget, %:

| set | checkpoint | 1,000 tokens: cascade / best fixed | 2,000 | 4,000 |
|---|---|---|---|---|
| hard12 test | thinking s1 it78 | 97.5 / 96.9 | 98.5 / 98.3 | 98.8 / 98.9 |
| textbook72 | graded it78 | 37.2 / 36.7 | 42.3 / 40.8 | 46.4 / 44.9 |
| textbook72 | EI thinking r2 | 36.1 / 34.5 | 43.0 / 41.5 | 49.1 / 47.6 |
| 13–20 | thinking s1 it78 | 10.6 / 11.4 | 17.4 / 17.0 | 23.3 / 21.9 |
| 13–20 | graded it78 | 12.9 / 13.1 | 19.6 / 19.0 | 25.5 / 25.2 |
| 13–20 | thinking s0 it39 | 11.2 / 12.3 | 19.7 / 17.6 | 26.6 / 24.0 |

**Oracle allocation under resampling** (`oracle_budget.py`): each problem's split-half best level, resampled i.i.d.
from its held-out odd attempts until the budget is spent, against the cascade. Success within budget, %; columns are
oracle / cascade:

| set | checkpoint | 1,000 tokens | 2,000 | 4,000 |
|---|---|---|---|---|
| 13–20 | thinking s1 it78 | 12.8 / 10.7 | 16.5 / 17.6 | 19.5 / 23.1 |
| 13–20 | graded it78 | 16.0 / 13.0 | 19.3 / 19.8 | 22.6 / 25.6 |
| 13–20 | thinking s0 it39 | 15.1 / 11.2 | 18.9 / 19.7 | 22.0 / 26.5 |
| textbook72 | graded it78 | 40.0 / 37.5 | 43.2 / 42.4 | 47.5 / 46.0 |
| textbook72 | thinking s1 it78 | 37.4 / 35.4 | 39.9 / 40.4 | 42.2 / 43.9 |

At 4,000 tokens the oracle is handicapped: it draws from only half of each problem's attempts, 4 of 8 on 13–20.

*Interpretation:*
- The accuracy ceiling for per-problem allocation is about 1–2 pp pass@1 over the best fixed level: +0.3 to +2.2 pp in
  8 of 9 cells, and −1.4 pp for EI on 13–20. Most of the gain is tokens, 30–65% fewer.
- Allocation pays only near one attempt per problem: at 1,000 tokens the oracle beats the cascade by +2.0 to +3.9 pp.
  From 2,000 tokens on, the cascade, which uses Lean and needs no training, matches it or wins.
- These are oracle bounds. A controller that reads only the state will capture part of them.
- Per-step allocation could do better than per-problem allocation. Nothing here measures that.

### 2.4 The model's own thinks in Lean-verified EI attempts are 10–15% false

- **Sample.** Every non-empty think block was replayed through `trace_dsl.check_step`, the strict validator. Source:
  300 random Lean-verified attempts per (round, budget) from `restart2_hard12/ei_T1v1e/found_4.jsonl` (EI's kept
  data), at B64 and B256.
- **Lower bounds.** The check stops at the first false claim in a block, so these counts are lower bounds.

| | round 1, B256 | round 4, B256 | round 1, B64 | round 4, B64 |
|---|---|---|---|---|
| blocks | 1,680 | 1,509 | 1,796 | 1,364 |
| blocks with no false claim | 84.8% | 89.8% | 79.0% | 85.7% |
| blocks whose first error is a false `<no>` (on a provable goal, a sound reduction, or a candidate that works) | 5.0% | 4.1% | ≥ 3.2% | ≥ 3.5% |
| `<no>` per block | 0.80 | 0.70 | 0.47 | 0.38 |

- At B64 the most common error is "an unrejected candidate fails" (10% of blocks in round 1). Part of that is forced
  closes cutting a block before the candidate's `<no>`.
- *Interpretation:*
  - Roughly 6% or more of the model's `<no>` verdicts are false, even in attempts that succeeded. EI keeps them
    because `--validate_think` is off (LIT_CRITIQUE C9).
  - Honest search will add more `<no>` lines. They are worth only as much as the model's ability to compute them.

### 2.5 The pool is exhausted, and the learnability sampler cannot compensate

**Skipped groups and seen pairs** (`metrics.jsonl`, restart3):
- At it78, 62–74% of groups were skipped: no-think 74%, effort s0 62%, effort s1 64%, one budget 67%.
- Arms with 4 budgets had drawn 14.3k of their 20k (target, budget) pairs at least once by it78. Between it60 and it78
  about half of each iteration's 256 draws were pairs never drawn before.
- Unseen pairs have p = 0.5, which gives the maximum weight, 0.28. So the sampler is still largely on its first pass.

**The weights are too flat even when every pair has been seen.**
- No-think had seen 4,998 of 5,000 targets by it78.
- A target solved 16/16 once has weight 0.083, against 0.28 for an in-band target: only 3.4×.
- *Interpretation (my arithmetic):* with about 93% of targets solved, roughly 400 in-band targets at 0.28 against
  4,600 near-solved targets at about 0.086 means about 22% of draws are in band. That is consistent with 74% of groups
  skipped.

**Per-target spread on hard12 test** (k 16):

| checkpoint | 16/16 | (0.1, 0.7] | 0/16 |
|---|---|---|---|
| GRPO no-think s0 it78 | 82.7% | 5.6% | 2.1% |
| GRPO thinking s0 it78, B256 | 81.8% | 4.9% | 0.9% |
| EI no-think r2 | 31.5% | 19.8% | 0.6% |
| EI thinking r2, B256 | 38.2% | 14.8% | 0.3% |

GRPO leaves 3.5× as many no-think targets at 0/16 as EI. That is the sharpening signature behind the pass@5 tie, the
same pattern as in Yue et al. [@yueDoesReinforcementLearning2025].

### 2.6 textbook72's 7–12-line band is 20 problems

- **Size.** dev58 by reference length: 15 problems at ≤ 6 lines, 20 at 7–12, 23 at 13+. train14 has no reference
  length.
- **Never solved.** In any one cell (k 64), 5–11 of the 20 are never solved. Across all 44 stored textbook72 cells,
  only 2 of the 20 are never solved.
- **Seeds.** GRPO no-think seeds 0 and 1 at it78 read 39.7% and 22.8% on these 20. The "33–40%" in the brief leaves
  out the 22.8%.
- *Interpretation:* any "failed vs solved" contrast on this band rests on about 10 problems, and which 10 depends on
  the seed.

### 2.7 The models never ask for more than 256 think tokens

**Forced closes at B256**, % of blocks:

| checkpoint | forced closes |
|---|---|
| GRPO thinking arms, every set | ≤ 0.5 |
| Graded, on 13–20 | 2.3 |
| LCPO (trained toward length) | 4.6 |
| Stage-1 graded | 1.2–2.3 |

**Think tokens per block at B256:**
- GRPO: 12–17 on hard12 and 28–33 on 13–20, about 2× more on OOD states;
- EI r2: 40 on hard12 and 56 on 13–20.

### 2.8 Longer composed problems are already an in-band supply

**Where the 13–20 items come from.** The sealed 13–20 set (3,003 items, 375 per length) is made of `cut_composite`
problems from test_21_30's pipeline ([test_13_20/README.md](test_13_20/README.md)):
- the pipeline composes pieces from the fork's generator by cuts on lemmas;
- it certifies vocabulary-minimal lengths by exact search;
- it applies the format limits.

So it already works as a proof-first generator, with lengths set by `build_composites.py --lo/--hi`.

Share of 13–20 test items solved 1–6 times of 8 at it78/r2. This is a proxy for fresh composites at these lengths.

| checkpoint | all lengths | 13–15 lines (n 375) |
|---|---|---|
| GRPO no-think s0 | 13.2% | 19.2% |
| GRPO no-think s1 | 12.3% | 17.9% |
| EI no-think r2 | 17.3% | 28.3% |
| GRPO thinking s1, B256 | 25.0% | 37.3% |

The fresh 7–12 stream in the hard12r pilot (no-think GRPO it35, k 8) had 2.7% at 1–2/8 plus 5.9% at 3–6/8, so about
8.6% in the same band. That stream came from hard12's generator configs, not from composites.

**The 21–30 set is at the floor.** Its rows from the eval running today, present at 08:31 UTC and not yet written up,
give 2,767 items at k 8:

| checkpoint | pass@1 | pass@8 |
|---|---|---|
| GRPO no-think s0 / s1 it78 | 0.23% / 0.26% | 0.87% / 0.83% |
| GRPO thinking s0 / s1 it78, B256 | 0.15% / 0.47% | 0.72% / 2.39% |
| graded it78, B256 | 0.79% | 3.94% |
| EI thinking r2, B256 | 1.22% | 5.24% |
| EI no-think r2 | 0.22% | 1.01% |

EI with thinking is the widest here too.

## 3. The planned changes

### 3.1 Honest-search traces (Stream-of-Search style)

**What the new generator does.** From `code/thinking/searchtrace.py`'s docstring, read at the end of this review, not
its code or output:
- each candidate reduction is checked by truth tables, which is complete for this fragment;
- the search follows the first candidate whose holes are all valid;
- it backtracks only when every candidate of a subgoal fails;
- each proof gets one of four strategies (`dfs`, `hyp`, `best`, `rand`), drawn from a hash of its id and absent from
  the text;
- a level fixes the token cap and the search's breadth and lookahead, and levels may produce different proofs;
- `min_level` is the smallest level whose commit-on-record action is still completable;
- countermodels are minimal partial assignments, checked by an extension of `trace_dsl`.

That design confirms risks 1, 3 and 4 below, so the gates matter.

One more point. The CLI's `--analysis-only` mode runs on sealed sets and writes no traces. Statistics are fine, but
design choices such as the e1024 trigger or the level ladder should come from training or dev statements, not from
the sealed sets.

**Risks specific to this setup.**

1. **A truth-table check makes most of the search trivial.**
   - Classical propositional logic over four atoms is decided by truth tables. A search that checks each candidate
     reduction's holes before descending finds every dead goal at depth one and never needs to backtrack deeper.
   - Dead goals are rare in these states:
     - non-ex-falso goal-level dead ends occur in about 12% of decision blocks, in both v1 and v2 (TRACE_AUDIT
       addendum);
     - 37.6% of decision steps are in inconsistent contexts, where no goal is dead.
   - So in most blocks a strategy-ordered honest search takes its first or second reduction. That is shallow search.
   - What a truth table cannot certify is the case where real search lives:
     - "provable, but a long or unclosable route";
     - "true, but a useless forward step".
   - An honest search learns about those only by expansion, which means long traces, or with a minimal-length
     oracle, which turns it back into a rationalisation of the shortest proof.
2. **The actions change, which confounds every comparison.**
   - "The proof is whatever it finds" means Stage 1 sees a new action distribution. A strategy that tries always-viable
     detours early will produce them: `Classical.byContradiction`, ex falso in inconsistent contexts, case splits.
   - Longer proofs cost accuracy on two measured axes (§2.1):
     - the per-step hazard compounds;
     - 46% of in-distribution failures already happen past the reference length.
   - Without a no-think model trained on the same found actions, a Stage-1 change cannot be attributed to thinking.
3. **The per-proof strategy is a hidden latent in a Markov format.**
   - The think block sees only the state, so the model must learn a mixture over strategy orders.
   - That is good for diversity, which is where thinking's gain lives (pass@5 on 13–20).
   - But it makes `min_level` labels (§3.3) and think content only partly predictable from the state.
4. **The effort token may come to condition proof style.**
   - The docstring says level 0 writes "an empty think and the action of the same search", yet "levels may produce
     different proofs", because breadth (`k_more`) and lookahead change with the level.
   - Where a level changes the proof, `<eB>` conditions proof style and length, and "e0 vs e256" then partly measures
     proof length.
   - Report the found-proof length per level. Keep the share of proofs that differ between levels small, or match
     levels on proofs as T1v1g4 did.
5. **Capacity and context.**
   - SoS's 250M model with a 4,096-token context imitated its 12 strategies at slightly below their average accuracy.
     It went beyond them only after STaR/APA.
   - APR finds that serialised search saturates around 3k tokens.
   - Here a pair has 1,024 tokens and the model has 9.5M parameters. RL cuts thinks to 12–30 tokens (§2.7), and LCPO
     showed that longer thinks bought nothing. A block that needs hundreds of tokens will be compressed away by RL.
6. **Content matters only partly.**
   - Physics of LMs 2.2 finds cheap fake mistakes ("retry_weak") nearly as good as realistic ones. Stechly et al.
     (cited in LIT_CRITIQUE) found swapped traces as good.
   - On the other side, the project's own filler arm T1f lost 3.3 pp against +1.7 pp for real traces (1 seed).
   - Content probably matters, but the size of the gain from *better* content is unknown.

**Literature.**
- **For:**
  - **SoS:** 43% of its training trajectories are failed searches, and it scores 51.27% vs 25.73% for optimal paths.
    After STaR/APA it solves about 36% of the problems its teacher strategies failed.
  - **Searchformer:** better than solution-only with ten times fewer sequences. Bootstrapping on its own shortest
    optimal traces cut trace length 26.8% with no loss of optimality.
  - **Procedure cloning:** imitating BFS computation states solves unseen 32×32 mazes where behaviour cloning fails.
  - **"Transformers Struggle to Learn to Search":** small from-scratch models learn search, including a
    propositional proof-search variant, only when the training distribution "preclude[s] the usefulness of shortcuts".
    That supports removing the reference-aware shortcuts TRACE_AUDIT found.
- **Against, or tempering:**
  - **"To Backtrack or Not":** at matched compute, models trained on backtracking traces lose to direct-solution
    models with best-of-n on CountDown (shallow search) and win on Sudoku (deep search).
    - A mix of 32 strategies approaches but does not beat direct solutions.
    - After GRPO, though, the backtracking model improves and keeps its diversity, while the direct model's best-of-n
      stops scaling.
  - **Wu et al. (2502.07266):** the best CoT length shrinks as the model gets more capable (an inverted U).
- **Missing:** no paper compares honest search with answer-aware rationalised traces directly.

**Better or cheaper alternatives.**
1. **Self-verification traces built from the policy's own errors.**
   - At each state of a solved proof, sample K candidate actions from the current policy and check each with the
     environment, `step_check` and truth tables.
   - The think block lists the policy's own failing candidates, with the looks that refute them, then the passing one.
   - This aims the think at the measured failure classes (§2.1). PROPOSAL §3.2 intended this, but base-model
     distractors reached only 7.8% of proofs.
   - Cost: GPU minutes of sampling.
2. **STaR-style self-traces from data on disk.**
   - `found_4.jsonl` holds 105k Lean-verified EI attempts with their thinks; 85–90% of the B256 blocks are strictly
     valid (§2.4).
   - Filter to fully valid attempts, keep the shortest per target (Searchformer's rule), and retrain Stage 1.
   - It needs no new generator and is the competitor honest search has to beat.
3. **Dualformer-style structured dropping for the levels.** Drop `<no>` certificates or rejected candidates rather
   than truncating the search. In Dualformer this made slow mode 49% shorter and better.

**Recommendations.**
1. Build the generator, but pass the CPU gates below before any GPU time.
2. Bound the found proofs: at most the minimal length + 2, using `gap_analysis`'s minimal-length search. Alternatively,
   order strategies so that introduction rules come before reductio and ex falso.
3. Measure how often levels change the proof (risk 4). If it is more than about 10% of proofs, train the levels on
   the same proof (the e256 search's actions, with thinks recorded at each budget), so that the effort token carries
   only thinking.
4. Draw at least half of the rejected candidates from the policy's own sampled errors, by class: `app-arg`, `and-elim`,
   binder mismatch, unbound (alternative 1).
5. Countermodel lines:
   - `trace_dsl` already has `parse_countermodel` and `fmt_countermodel`; keep each line to 8 tokens or fewer.
   - Measure the `<no>` precision of sampled thinks on held-out states. The baseline is §2.4's roughly 94%, measured
     on training targets.
6. Train **T0-S**, a no-think model on the same found actions, in the same run.
7. Run the STaR self-trace arm (alternative 2) as the cheap competitor.

**Pre-registered criteria and the cheapest test.**
- **CPU gates on the generated corpus.** All must hold, or the generator is redesigned before any GPU time.
  - **G1, shortcut.** TRACE_AUDIT's string-match verdict rule is ≤ 60% accurate (v1 94.7%, v2 51.7%). The action
    equals the term of the first reduction line in ≤ 50% of decision blocks.
  - **G2, real search.** ≥ 25% of decision blocks contain a certified non-ex-falso dead end or a rejected
    policy-sampled candidate (v2: about 12%).
  - **G3, proof length.** Found length ≤ minimal + 2 in ≥ 90% of proofs. Mean found/reference length ≤ 1.10.
  - **G4, length tracks search.** Spearman ρ between think tokens and expansions ≥ 0.6. ρ with goal-formula length
    ≤ 0.3 (v1: 0.33 and 0.48 respectively).
  - **G5, fit.** ≥ 99% of e256 pairs are ≤ 1,024 tokens.
- **Stage 1 only, about 5 GPU-h.**
  - Runs: S (search traces) and T0-S, 2 seeds each.
  - Reads: hard12_test_1k and textbook72 at B 0/64/256, and 13–20 at pass@5. 13–20 pass@1 is at the floor (0.5–0.7%)
    for Stage-1 models.
  - **Success**, all three required:
    - S at B256 is ≥ 3 pp above T1v1 at B256 (42.6%) on hard12 pass@1, seeds pooled with the interval excluding 0,
      and S is not below T1v1g4 (48.7%);
    - S − T0-S is > 0, resolved;
    - `--intervene swap` costs S at least 5 pp more than it costs T1v1.
  - **Failure:** S ≤ T1v1, or S − T0-S ≤ 0. Then stop: trace content is not the lever at Stage 1.
- **Only on success:** GRPO to it39 for S and T1v1, 2 seeds each. Success is ≥ +4 pp pass@5 on 13–20, seeds pooled.

### 3.2 The budget ladder: drop e16, and add e1024 only behind a gate

**Drop e16: agreed, no experiment needed.**
- e16 was below e0 at the init and after EI round 1 (−10.9 and −25.7 pp; ADAPTIVITY).
- 23% of e16 blocks are empty in the graded traces.
- After GRPO, e16 and e0 sit level. LIT_CRITIQUE's prediction P3 holds.

**e1024: no demand observed.**
- Forced closes at B256 are ≤ 0.5% for GRPO models (§2.7).
- Every forced minimum lowered accuracy (RL_RESULTS §4).
- LCPO made thinks longer and gained nothing.
- Literature:
  - s1: suppressing the end-of-thinking delimiter too often leads to repetition loops, and gains flatten.
  - L1: its limitations section says control may not generalise to lengths beyond those trained.
  - Wu et al. 2502.07266: the inverted U.

**The planned trigger measures the generator, not the model.**
- The trigger is "≥ 10% of uncapped generator blocks exceed 256 tokens".
- A median goal line is 22 tokens, so a search that lists 10–12 candidates passes 256 whether or not the model can use
  the extra tokens.

**The cost is real.**
- The prefill per step is only 90–145 tokens on average, so e1024 pairs fit in about 1.3k tokens.
- But Stage-1 training, `max_len` and the GRPO packing all assume 1,024.
- The runaway actions in §2.2 show that long generations are where this model degenerates.

**Recommendation.** Keep the ladder at {e0, e64, e256}. If the search corpus needs more than 256 tokens for some
blocks, compress them (Dualformer-style drops) rather than extending the context.

**Pre-registered gate, no extra GPU.** Add e1024 only if both hold, read from evals the S arm needs anyway:
- S-Stage-1 B256 − B64 ≥ +2 pp pass@1 on hard12, resolved over 2 seeds;
- forced closes at B256 are ≥ 5% of blocks on 13–20.

### 3.3 Model-decided effort: drop it, or defer it indefinitely

**Risks.**
1. **Little headroom (§2.3).**
   - The split-half per-problem oracle gains −1.4 to +2.2 pp pass@1 over the best fixed level.
   - Under resampling it beats the free Lean cascade by 2–4 pp at 1,000 tokens per problem, and not at all from 2,000
     tokens on.
   - The planned test ("beats the best fixed level at matched tokens") can be passed trivially under per-attempt
     accounting, by spending nothing on hopeless problems, while still losing to resampling with the verifier.
   - Under the project's standard (matched tokens per problem, pass@1 and pass@5), the remaining 1–2 pp is below the
     13–20 seed spread of 3–4 pp between two seeds.
2. **Continuous model-decided effort already exists.**
   - At B256 the model picks its own stopping point, and it thinks about 2× more on 13–20 than on hard12 (§2.7;
     ADAPTIVITY ρ −0.43).
   - "One budget + token cost" cut tokens 40% at unchanged accuracy.
   - A self-emitted `<eB>` adds a second, coarser controller on top.
3. **`min_level` labels measure the generator's strategy order, not the model's need.**
   - If honest search finds a viable action at depth one in most blocks (§3.1, risk 1), `min_level` will be e0 or e64
     for nearly every step, so auto records will teach "rarely think".
   - In TALE, the only true minimal-budget labelling found, monotonicity failed in about 9% of cases, and budgets below
     the minimum *increased* tokens.
   - Thinkless and AdaptThink use no labels; routing is learned from outcomes.
4. **The control token collapses under RL, and this is well documented:**
   - Thinkless collapsed toward short: under 10 long samples within 120 steps, from 1/(T+1) normalisation of the
     control token.
   - ARM collapsed toward long within about 10 steps.
   - AdaCoT collapsed to always-CoT.
   - AdaptThink never finds the cheap mode without forced sampling.
   - Here the in-distribution pool is 93% solved, so λ pushes toward e0. The OOD value of thinking (pass@5 on 13–20)
     is invisible to that trade-off.
5. **Reward shape.**
   - correct − λ·tokens penalises long failures as well.
   - Arora & Zanette penalise length on correct answers only, normalised per prompt, and their accuracy falls smoothly
     with α. That is safer if λ is mis-set.
   - Fatemi et al. show that per-token loss averaging lengthens wrong answers.

**Better alternatives.** Report the Lean cascade as the effort baseline, and keep "one budget + token cost" as the
efficiency arm.

**If it is built anyway:**
- use Thinkless's decoupled control-token weight, or AdaptThink's forced 50/50 control-token sampling inside RL;
- log each level's share every iteration;
- penalise only correct answers' tokens;
- run a λ sweep with ≥ 2 seeds.

**Pre-registered gate, CPU only.** Re-run both headroom analyses on the next thinking model's B0/64/256 rows. Build auto
mode only if both hold on 13–20:
- the split-half oracle beats the best fixed level by ≥ 3 pp pass@1 at no more tokens;
- the oracle under resampling beats the cascade by ≥ 3 pp at 1,000 tokens per problem.

The three GRPO checkpoints measured fail the first condition (+0.5 to +1.1 pp on 13–20; EI r2 is −1.4) and meet the
second in 2 of 3 cells (+2.1, +3.0, +3.9 pp). An oracle that passes only one of them leaves too little for a learned
controller.

**Success if built:** auto beats both "one budget + token cost" and the cascade at matched tokens per problem, on
13–20 and textbook72, at pass@1 and pass@5, with 3 seeds.

### 3.4 Harder 7–12-line problems

**Where this stands.**
- The gap analysis is done ([gap_analysis/README.md](gap_analysis/README.md)), and a proof-first generator built from
  its targets is in progress (`hard712/`: `pfgen.py`, `pfgates.py`).
- The generator samples box skeletons (I(N()), N(N()), C(N()), case splits inside boxes, and so on) and fills them
  with a backward normal-proof builder.
- Its gates are: compact statements; a shortest proof of 7–12 lines that is within 1 line of the constructed one;
  necessity of the target structure; and exclusion.
- That already follows several recommendations I would have made: structural hardness, necessity checks, size limits,
  and no generator artefacts.

**Risks.**
1. **Part of the design is tuned on 20 textbook problems.**
   - The manifest says "never train or tune on these problems or on a renaming of them". The gap analysis says so
     itself: dev58 "is therefore no longer a clean held-out set" for this generator.
   - Its strongest evidence needs no failure labels:
     - shortest proofs in the pool collapse: 57% of "7–12-line" pool problems have a shortest proof of 6 lines or
       fewer, against 0% for textbook problems;
     - nested ¬-boxes are needed in textbook problems but avoidable in 41/41 sampled pool cases;
     - the C(N()) and C(O(|)) skeletons occur 0 times in the pool.
   - The failure-label contrasts (TB fail n = 8 vs solved n = 12, about 45 features) survive no multiple-comparison
     correction, and seed swings on these 20 problems reach 17 pp (§2.6). They were used to weight T3 and T4.
   - *Interpretation:* T1, T2 and T5–T8 are justified by the textbook-vs-pool prevalence and necessity contrasts alone.
     T3 and T4 are too (25–35% vs 1–2% prevalence), so the within-dev failure contrasts add tuning without adding
     information.
2. **The textbook failures are mostly invalid steps (§2.1).**
   - 13–54% of attempts are Lean-rejected after the environment accepted them; 9–14% cite unbound names.
   - Harder theorems in the generator's existing shapes do not obviously target those errors.
   - Coverage of *shapes* the generator lacks would, for example zero-premise theorems or nested boxes. The gap
     analysis code already computes these from statements alone (`thm_features`, `intuit_provable`).
3. **"Hard" may come to mean "large".**
   - hard12r found that steerable structural features barely move post-RL hardness (3–9% at ≤ 2/8). A proof-first
     generator changes the support, which is a different matter.
   - Hardness bought with long formulas, many premises and many names triggers the formula-copy errors that already
     dominate on 13–20 (binder mismatch 32.9%). Minimo's harder conjectures grew longer, not deeper.
   - `hard712`'s compact-statement gate (≤ 2 premises, ≤ 7 connectives, depth ≤ 3) guards against this by design.
     Keep it, and report the in-band items' formula sizes.
4. **The lower bound on proof length holds only inside S.**
   - `fast_minlen` searches the subformula space S: the theorem's subformulas, their negations and double negations.
   - It certifies "no shorter proof in S", not a global bound. State this wherever the bound is used.
5. **Distractor premises move the generator away from the target.**
   - The gap analysis measured 0 redundant premises in textbook 7–12 problems against 1.19 per pool theorem, and its
     target T7 removes them.
   - Planned distractor premises would put the artefact back.
   - Anil et al. find distractor tokens are "the main culprit" in scratchpad length-generalisation failures.
   - Drop this item.
6. **Near-miss mutation and screening select for one checkpoint's blind spots.**
   - Alchemy's mutation gains were about twice as large in distribution as out of it.
   - The hard12 seed control showed the cost: T0 s0 read 27.8% against 42.4% for other seeds on a set it selected.
   - Mutating anything near a test item leaks.

**Literature.**
- **HTPS:**
  - its synthetic generator saturated at 100% while a supervised model never exceeded 36% on hand-written Identities;
  - the gap closed with online training on the target-style statements themselves, which is transductive.
- **Firoiu et al.:** the generator was exhausted at about 100%, yet only 77.7% of the TPTP target problems were
  expressible by the forward proposer. *Interpretation:* measure what fraction of the textbook shapes the generator
  can express at all.
- **INT:** for transformers, the diversity of rule combinations drives OOD generalisation: 25 → 300 training
  combinations took success from 29% to 91%. Training on 7-step proofs loses 5.7% per extra step, against 18.8% for
  3-step training.
- **AlphaGeometry:** hardness is *structural*. 9% of synthetic proofs need an auxiliary construction, and fine-tuning
  focuses on that subset.
- **MetaGen:** synthetic data adds only 1.5–3% on top of real data.
- **DeepSeek-Prover:** distribution-matched synthetic statements beat Mathlib-only data by about 11 points, with gains
  flattening past 100k examples.
- **STP:**
  - keeps conjectures with a pass rate in (0, 1/4], re-weighted toward the target statements;
  - conjectures add about 2–3 points;
  - the main win is reward density.
- **Minimo** [@poesia2024learningformalmathematicsintrinsic] (8.45M parameters, propositional logic): transfer to
  Kleene's theorems is real but small.
- **Zhang, Neubig & Yue (2512.07783),** 100M parameters from scratch: RL on in-distribution data raises pass@1 but not
  in-distribution pass@128; RL on edge-of-competence data raises OOD pass@128.

**Better or cheaper alternatives.**
1. **Use the composition pipeline as a baseline for `hard712`.**
   - test_21_30's `build_composites.py` already composes generator pieces by cuts, certifies minimal lengths and
     removes ex falso and inconsistent or redundant premises. So it delivers T1 (necessary length) and T7 (no
     artefacts) for free.
   - Its pieces come from the fork's generator, which almost never makes necessary negation work inside boxes (gap
     analysis §5). So composites probably lack T3–T5.
   - Screening 7–12 composites next to `hard712` separates the two questions: is the gap necessary length, or is it
     structure?
2. **Grow length with the same pipeline.** 13–15-line composites are already in band 2–4× as often as fresh 7–12
   candidates (§2.8), with no new code. This costs part of the 13–20 OOD read (§4.1).
3. **Structural hardness labels,** the AlphaGeometry analogue. Mark problems whose minimal proof needs an assumption
   the policy must invent (→I, ¬I or ∨E boxes, reductio, or a lemma that is not a subformula), and train or weight on
   that subset. `hard712`'s necessity gate already computes this label. Record it per row so that RL can weight on it
   and evals can bin by it.
4. **Textbook-like statements as RL targets,** the HTPS route. textbook72 is excluded by its rule, but
   `transfer_long`'s 282 textbook-schema theorems could be split into a training half and a dev half, keeping
   textbook72 sealed.
5. **Step-level checking in the environment** (§4.5), since invalid steps dominate.

**Recommendations.**
1. State `hard712`'s target shares from the textbook-vs-pool prevalence and necessity contrasts only, and stop using
   within-dev58 failure contrasts for any further choice.
2. Relabel dev58 as "dev" in every future report, as the gap analysis asks.
3. Before any claim about a `hard712`-trained model, get a fresh textbook-like test.
   - Candidates: new problems from other logic textbooks, or `transfer_long`'s 282 textbook-schema theorems (0
     renaming-class overlaps with textbook72; 11–14 lines).
   - For `transfer_long`, first check that its 6 schemata do not coincide with dev58's statements at the schema level.
4. Gate `hard712` on yield against the composites (below).
5. Drop distractor premises (risk 5).
6. Mutation: only from training-pool items, and exclude mutants within one edit of any test item, not just their
   renaming class.

**Pre-registered criterion and the cheapest test, about 0.5 GPU-h.**
- **Candidates,** all generated on CPU and excluded against the 120-file index:
  - 3k `hard712` candidates that pass its gates;
  - 3k 7–12-line composites from the existing pipeline;
  - 3k 13–15-line composites.
- **Screen:** every set with GRPO no-think it78 at k 8, plus thinking s1 at B256 on 1k of each.
- **Success for `hard712`:**
  - ≥ 20% of candidates in 1–6/8 (the fresh 7–12 stream: about 8.6%);
  - a yield no lower than the 7–12 composites';
  - within `hard712`, the in-band rate higher for rows with a necessary T3–T5 structure than without.
- **If the 7–12 composites match `hard712`:** necessary length, not structure, is the active ingredient. Use
  composites, which are cheaper and already sealed-set compatible.
- **If both are below the 13–15 composites' yield:** length growth is the cheaper supply.
- **The transfer test comes later, after RL on the chosen pool:** textbook-like dev reads only, never dev58.

### 3.5 Curriculum

**(a) Online pool refresh: keep problems solved 10–70% of the time.**
- **The pool is the binding constraint, not the sampler (§2.5).**
  - At it78 only about 280 of hard12's 5,000 targets would be in band, roughly one iteration's worth. This is
    estimated from the test split's 5.6%.
  - A refresh sampler without new supply will recycle them.
  - So this change only works together with §3.4 or with length growth.
- **The sampler should still be fixed, cheaply:**
  - Screening in the style of SPEED-RL: draw 4–8 attempts and keep the target if 0 < p̂ < 1, then sample the rest of
    the group. SPEED reports 2–6× wall-clock savings.
  - Or DAPO's refill. With 64% of groups skipped, that costs about 2.8× the rollouts per kept group; my arithmetic.
  - Pool each target's posterior across its budgets. Its e64 and e256 pass rates are close (§2.3), so this cuts 20k
    pairs to 5k.
  - Never retire a target permanently. A target with true p = 0.9 goes 16/16 19% of the time, and GRESO finds about
    20% of zero-variance prompts become informative again.
- **Band.**
  - Bae et al. find 0.3–0.7 best (30.1 against 26.3 for plain GRPO, Qwen2.5-3B), and one-sided filters hurt.
  - Theorem-proving practice skews harder: STP keeps (0, 1/4] and QuestA ≤ 4/8.
  - GRPO already leaves more targets at 0/16 than EI (§2.5).
  - *Interpretation:* keep (0, 0.7] at k 8, weight toward about 0.3, and do not cut the low tail.

**(b) Reverse curriculum from reference-proof prefixes.**
- **In this environment the policy is Markov in the state.**
  - Starting an episode at step t of the reference proof is the same as training on reference-proof states, and Stage
    1 already imitated every K12 reference state.
  - So on 7–12-line problems at 93% it adds nothing. Its value is on long problems, where late states with more names
    and hypotheses are novel and outcome reward is otherwise absent.
- **Where failures fall.**
  - The 13–20 hazard is flat (§2.1), so failures are spread over the whole proof.
  - A start-state curriculum would help by giving signal on late, deep states, not by solving start-of-proof
    exploration.
- **Literature:**
  - **Florensa et al.:** "good starts" are states with a success rate in 0.1–0.9.
  - **Salimans & Chen:** fragile; failed on Gravitar and Pitfall; sensitive to the entropy coefficient.
  - **R³:** mixed starts beat staged ones (65.6 vs 62.8 average, against 61.5 for outcome-only RL). Staged schedules
    drop at each transition.
  - **QuestA and AdaBack:** hints help the unhinted policy when aimed at near-zero-p problems, and pass@k is kept
    (shown in figures only).
  - **FLoP** [@zomboriFindingLongerProofs2021], in theorem proving: starts from the end of a training proof and moves
    back once the success rate passes a threshold. Its ablation gain is small (0.24 vs 0.23 and 0.51 vs 0.45 proofs
    found, 2 runs). It warns that the reward for shorter proofs teaches shortcuts on easy problems.
- **Risks:**
  - start states inherit the reference's environment-assigned names and its detours; K12 references are sometimes
    roundabout (TRACE_AUDIT finding 3);
  - suffix successes inflate the batch pass rate, so keep them out of curriculum statistics;
  - the policy never practises the early decisions.

**(c) Grow proof length as the policy saturates.** This is the strongest supply option (§2.8).
- **Lee et al.** grow difficulty by +1 within a "safe range"; big jumps fail.
- **INT:** training on longer proofs degrades more slowly with length.
- **Zhang, Neubig & Yue:** edge-of-competence data is what raises OOD pass@128.
- **Cost:** training on 13–15-line composites makes that third of the 13–20 set in distribution. Keep the 1,878
  items of 16–20 lines as the OOD read (§4.1).

**(d) Training on extracted sub-proofs.** In the Markov environment a box-opening state is already a sub-theorem with
extra hypotheses, so this is reverse curriculum by another name. Low priority.

**Add a coverage-preserving objective.** None of the six changes addresses the reason GRPO ties EI at pass@5 (§2.5).
Options with measured pass@k effects:
- **He et al.'s unlikeliness reward,** in Lean: 8,065 vs 7,860 training problems solved, with small benchmark gains.
- **Zhu et al.'s W-REINFORCE or negative-only training:** pass@256 kept at base level.
- **pass@k training** (Chen et al.).

Each is a one-line advantage change. The first literature review ranked this third (`2026-09-29-lit-review`, #3), and
it has not been run.

**Pre-registered criterion and the cheapest test, about 10 GPU-h.** Use the no-think arm, which has no thinking
confounds.
- **Arms:**
  - A: continue GRPO from it78 on hard12 (control).
  - B: the same, with k 4 screening and refill, on hard12 ∪ fresh 13–15-line composites (excluded against the sealed
    sets), keeping p̂ ∈ (0, 0.7].
- **Protocol:** 39 iterations, 2 seeds, attempts matched with screening counted.
- **Success:**
  - skipped groups ≤ 30%;
  - ≥ +3 pp pass@5 on the 16–20-line part of the sealed 13–20 set (1,878 items);
  - no loss on the dev half of `transfer_long`'s textbook-schema theorems;
  - hard12 test no worse than −1 pp.
  - The 21–30 set is at the floor (§2.8), so read it at pass@8 as a secondary check only.
- **Optional third arm:** B plus the unlikeliness reward.

### 3.6 Traces on the harder problems (Stage-1 retrain)

- This depends on §3.1 and §3.4 both passing their gates.
- It changes problems, traces and actions at once, so it is interpretable only as the last cell of a 2×2 (trace type ×
  problem set), with T0 trained on each problem set as the control for the problem effect.
- If the generator was informed by textbook dev58, textbook72 stops being a test at this point.
- **Recommendation:** defer until both gates pass.
- **Success:** the S × new-problems cell beats S × K12 by ≥ 3 pp on the dev reads, and the T0 × new-problems control
  accounts for less than half of that gain.

## 4. Cross-cutting issues

### 4.1 Evaluation validity and contamination

**textbook72.**
- The gap analysis used dev58's per-problem pass rates and shortest-proof structure to choose generator targets, and
  says so. See §3.4 for the rule, the n = 20 problem and what to do now.
- `runs/gap_analysis/minproofs_textbook.jsonl` holds shortest proofs of textbook problems. Its statements are
  textbook72's, so the class-based exclusion in `data.py` and the pool builders already removes them. The remaining
  risk is reuse of the file itself, for example as trace-generator input. Its README warning is the only guard.
- I checked `transfer_long`'s 282 textbook-schema theorems against all 72 textbook72 problems by renaming class, with
  premise order ignored: 0 overlaps.
- They are schema instances of 6 textbook schemata (De Morgan, both distributivity laws, negated conditional, Peirce).
  I did not check whether those schemata also appear among textbook72's statements. Check that before treating
  `transfer_long` as independent.

**The in-distribution test is already selected.**
- hard12_test_1k carries the T0 s0 2–16/32 selection (RL_RESULTS caveat). The rule "test splits stay unfiltered random
  samples" is therefore already broken for the main in-distribution set.
- Use it only for comparisons within one arm, and build an unfiltered split for any new pool.

**Pool selection.**
- A pool selected by arm A's checkpoint handicaps arm A (regression to the mean).
- Screen with a model that is not in the comparison, or with the union of two seeds, and give every compared arm the
  same pool.

**Mutation.** Seed mutations only from training items, and exclude test items' one-edit neighbourhoods as well as
their renaming classes.

**Length growth.**
- The sealed 13–20 set has 3,003 items, 375 at each length. If 13–15-line composites enter training, its 13–15 third
  stops being OOD.
- Keep the 1,878 items of 16–20 lines as the headline OOD read.
- Do not move the headline to 21–30: it is at the floor, with 0.15–1.22% pass@1 and 0.7–5.2% pass@8 for the it78/r2
  checkpoints (§2.8).
- Decide the training-length cap before any training run uses composites, and write it in the queue.

**Training-time probes.**
- Seed 0's runaway actions were visible only OOD (§2.2).
- Log a fixed probe every 10 iterations: for example 200 fresh 13–20 problems that are not in the sealed set, at k 4,
  with failure classes broken out (runaway actions, binder mismatch, Lean rejection).
- Never use the sealed sets for checkpoint selection.

**Matched tokens.**
- Count matched tokens per problem with resampling, which is the project's convention, never per attempt (§3.3).
- Include rejected redraws in guided arms.

### 4.2 Confounds between changes made together

**Stage-1 data.** Search traces change four things at once:
- the actions;
- the proof lengths;
- the share of empty blocks;
- what each level means.

The training budget is matched by optimizer steps, not by loss tokens (T1v1g4 had 27% fewer loss tokens than T1v1).
Controls: T0-S, plus a loss-token count reported per arm.

**RL recipe.**
- The pool (§3.4), curriculum (§3.5), λ and auto mode (§3.3) all change the effective training distribution.
- Change one at a time from a fixed init (it78, or Stage-1 T1v1), on the no-think arm first.

**The collapse.** Until `collapse_audit/` explains seed 0, any single-seed thinking result past it39 is
uninterpretable. Two seeds is the minimum for every claimed arm.

**Selection versus content.** EI keeps think blocks with false claims (§2.4). An EI-vs-GRPO thinking comparison
therefore also compares "trained on 10–15% false thinks" with "trained on sampled thinks under the GRPO loss".

### 4.3 Statistical power

- **13–20 pass@1:** two seeds of the same recipe differ by 3.2 pp (no-think) and 3.9 pp (effort).
  - Taking the seed SD as about 2.3 pp, 3 seeds per arm resolve differences of about 5 pp, and 3 pp needs about 9
    seeds per arm. This is rough two-sample arithmetic; pairing on theorems helps only with item noise.
- **Thinking effects on 13–20 pass@5:** +9 to +15 pp. That is the read that can resolve differences with 2–3 seeds.
- **textbook72:** nothing below about ±8 pp resolves, and its 7–12 band (n = 20) resolves nothing.

### 4.4 Compute budget (estimates)

Unit costs are from RL_RESULTS:
- Stage 1: about 1 GPU-h (T1v1_s0 took 52 min);
- GRPO to it78: about 3 GPU-h;
- one full eval suite: about 1.5 GPU-h.

| item | as planned (GPU-h) | as recommended (GPU-h) |
|---|---|---|
| 1. Search traces: Stage 1 for S and T0-S, 2 seeds each, plus Stage-1 evals; then RL to it39 for S and T1v1, 2 seeds | ~16 | ~16, gated on CPU first; ~5 if Stage 1 fails |
| 2. Ladder | ~2 (a 2k-context Stage 1) | 0 |
| 3. Auto effort: mixed Stage 1 × 2, RL λ sweep of 3 × 2 seeds | ~30 | 0 (CPU gate only) |
| 4. Harder problems: screening, then RL on the new pool for 2 seeds | ~12 | ~0.5 screen, then ~9 if it passes |
| 5. Curriculum: refresh and reverse arms, 2 seeds each | ~18 | ~10 (one refresh arm and control; reverse only on long targets) |
| 6. Stage-1 retrain on harder problems with traces, 2 seeds | ~11 | deferred |
| New: logical-arm read (§4.5) | — | ~3 |
| **Total** | **~90** | **~40** |

### 4.5 What to run first

**Step 0: CPU, now, free.**
1. Gates G1–G5 (§3.1) on the first `searchtrace.py` output.
2. Restate `hard712`'s target shares from the gap analysis's textbook-vs-pool contrasts only (§3.4).
3. Re-run the headroom and think-validity analyses on any new checkpoint. They are ad hoc scratchpad scripts, fully
   described in §7.

**Step 1: the single first GPU experiment, a logical-arm read (eval only, about 3 GPU-h).**
- **Checkpoints:** GRPO no-think s0 and s1 it78, GRPO thinking s1 it78 at B256, and graded it78 at B256.
- **Sets:** 13–20 and textbook72.
- **Arms:** `--arms plain,logical`. The logical arm redraws the whole think + action when the environment or
  `step_check` rejects a step, up to `max_rej` 10.
- **Compare** at matched tokens per problem, rejected draws included, at 1,000 / 2,000 / 4,000 tokens.
- **What it decides.** Whether the dominant failure is a missing valid action at a state the policy can otherwise
  handle (redraws recover it) or the absence of any viable continuation (redraws do not). It also supplies the
  baseline that LIT_CRITIQUE C1 asked every thinking claim to beat and that no RL'd checkpoint has been read against.
- **Pre-registered outcomes:**
  - **Step validity is the binding constraint:** logical − plain ≥ +5 pp at 2,000 tokens per problem on 13–20 for
    every checkpoint. Then:
    1. add `step_check` to the RL environment, as early termination and optionally as a redraw arm;
    2. build the search traces around self-verification of the policy's own candidates (§3.1, alternative 1);
    3. make every later thinking claim against the logical arm.
  - **Validity errors are symptoms of having no plan:** logical − plain ≤ +1 pp. Then proceed with the honest-search
    generator as designed.
  - **Between +1 and +5 pp:** do both, with the search traces still gated.
- **Prior.** Dan's guided-tts found +9.1 pp (IQM, 6 of 6 models) for logical redraws on long theorems at matched
  tokens. That was not measured on RL'd or thinking checkpoints.

**Step 2:** the yield screen (§3.4), about 0.5 GPU-h.

**Step 3:** Stage 1 for S and T0-S (§3.1), about 5 GPU-h.

**Step 4:** the curriculum arm on no-think (§3.5), about 10 GPU-h.

## 5. Ranked plan

| rank | change | decision | why |
|---|---|---|---|
| 1 | Logical-arm diagnostic; then `step_check` in the RL env if positive | **add** (not in the plan) | Invalid steps are almost all OOD failures (§2.1). Cheap and decisive for how item 3 below is designed |
| 2 | Curriculum (a) and (c): screening and refill sampler, supply by length growth with the existing composition pipeline | **do, on no-think first** | The pool is exhausted (§2.5); 13–15-line composites are in band 2–4× more often (§2.8). Keep the 16–20-line part of the sealed set as OOD |
| 3 | Honest-search traces | **do, redesigned and gated** | CPU gates G1–G5, policy-error candidates, bounded proof length, T0-S control, Stage 1 first. Run the STaR self-trace arm beside it |
| 4 | Coverage-preserving objective (unlikeliness, W-REINFORCE or pass@k) | **add as an arm of item 2** | It is the only lever on the GRPO-vs-EI pass@5 tie |
| 5 | Harder 7–12 problems (`hard712`) | **keep, gated** | Its structural targets are well supported by the textbook-vs-pool contrasts. Stop using failure labels, drop distractors, gate on yield against 7–12 composites, and get a fresh textbook-like test (§3.4) |
| 6 | Budget ladder | **drop e16 now; gate e1024** | No observed demand above 256 (§2.7) |
| 7 | Reverse curriculum and sub-proofs | **low priority** | Only for long targets, with mixed starts and ≥ 25% true starts; the Markov env makes it close to SFT on reference states |
| 8 | Traces on the harder problems | **defer** | Only after items 3 and 5 pass, as the last cell of a 2×2 |
| — | Model-decided effort | **drop** | Oracle headroom is ≤ 2 pp pass@1, and beyond one attempt per problem the free cascade captures it (§2.3). Continuous self-chosen length plus token cost already exists. The control token is collapse-prone and the test is underpowered |

## 6. References and read depth

**How these were read.**
- Four research subagents and I read these papers on 2026-10-08 through WebFetch. WebFetch returns a model-written
  extraction of the page in answer to targeted questions, so **nothing was read verbatim**, and values shown only in
  figures were not seen.
- Read-depth labels:
  - **full**: the whole HTML or ar5iv text was queried;
  - **partial**: named sections, or about the first 100k characters;
  - **abstract**: the abstract or secondary summaries only.
- I re-checked the abstracts of 2504.07052, 2512.07783 and 2504.03380 myself.
- Papers cited only through earlier notes (Stechly et al. 2505.13775; Gandhi et al. 2503.01307; POPE; Go-Explore;
  DeepHOL-Zero) were not re-read.
- None was added to Zotero or `references.bib`. The four `papers:` citekeys already exist there.

**Search-trace training**

| paper | id | read |
|---|---|---|
| Gandhi et al. 2024, Stream of Search (SoS): Learning to Search in Language | arXiv 2404.03683 | full |
| Lehnert et al. 2024, Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer) | arXiv 2402.14083 | full |
| Su et al. 2024, Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces | arXiv 2410.09918 | full |
| Yang et al. 2022, Chain of Thought Imitation with Procedure Cloning | arXiv 2205.10816 | full (ar5iv); most results figure-only |
| Qin et al. 2025, To Backtrack or Not to Backtrack: When Sequential Search Limits Model Reasoning | arXiv 2504.07052 | partial (main text, App. G–H) |
| Saparov et al. 2024, Transformers Struggle to Learn to Search | arXiv 2412.04703 | partial (~88%) |
| Saha et al. 2024, System-1.x: Learning to Balance Fast and Slow Planning with Language Models | arXiv 2407.14414 | full |
| Ye et al. 2024, Physics of Language Models Part 2.2: How to Learn From Mistakes on Grade-School Math Problems | arXiv 2408.16293 | partial (§3–6) |
| Pan et al. 2025, Learning Adaptive Parallel Reasoning with Language Models (APR) | arXiv 2504.15466 | full |
| Wu et al. 2025, When More is Less: Understanding Chain-of-Thought Length in LLMs | arXiv 2502.07266 | partial |

**Adaptive and learned compute**

| paper | id | read |
|---|---|---|
| Fang et al. 2025, Thinkless: LLM Learns When to Think | arXiv 2505.13379 | full (appendix not seen) |
| Zhang et al. 2025, AdaptThink: Reasoning Models Can Learn When to Think | arXiv 2505.13417 | full |
| Aggarwal & Welleck 2025, L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning | arXiv 2503.04697 | full |
| Hou et al. 2025, ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning | arXiv 2504.01296 | full |
| Muennighoff et al. 2025, s1: Simple test-time scaling | arXiv 2501.19393 | partial (main text) |
| Chen et al. 2024, Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs | arXiv 2412.21187 | full |
| Cuadron et al. 2025, The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks | arXiv 2502.08235 | full |
| Gema et al. 2025, Inverse Scaling in Test-Time Compute | arXiv 2507.14417 | partial (abstract, §3, §4.3) |
| Hassid et al. 2025, Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning | arXiv 2505.17813 | full |
| Arora & Zanette 2025, Training Language Models to Reason Efficiently | arXiv 2502.04463 | full |
| Fatemi et al. 2025, Concise Reasoning via Reinforcement Learning | arXiv 2504.05185 | full (proofs partial) |
| Liu et al. 2025, Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | arXiv 2503.20783 | abstract |
| Yeo et al. 2025, Demystifying Long Chain-of-Thought Reasoning in LLMs | arXiv 2502.03373 | partial (main text) |
| Graves 2016, Adaptive Computation Time for Recurrent Neural Networks | arXiv 1603.08983 | full |
| Banino et al. 2021, PonderNet: Learning to Ponder | arXiv 2107.05407 | full |
| Snell et al. 2024, Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | arXiv 2408.03314 | partial (main text) |
| Damani et al. 2024, Learning How Hard to Think: Input-Adaptive Allocation of LM Computation | arXiv 2410.04707 | partial (main text) |
| Lou et al. 2025, AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning | arXiv 2505.11896 | partial (main text) |
| Wu et al. 2025, Arm: Adaptive Reasoning Model | arXiv 2505.20258 | partial (main text) |
| Han et al. 2024, Token-Budget-Aware LLM Reasoning (TALE) | arXiv 2412.18547 | partial (main text) |
| Li et al. 2025, SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning | arXiv 2505.11274 | partial (main text) |

**Curricula, prompt selection and self-play**

| paper | id | read |
|---|---|---|
| Yu et al. 2025, DAPO: An Open-Source LLM Reinforcement Learning System at Scale | arXiv 2503.14476 | full |
| Bae et al. 2025, Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning | arXiv 2504.03380 | partial (~100k characters; abstract re-checked) |
| Zheng et al. 2025, Act Only When It Pays: Efficient RL for LLM Reasoning via Selective Rollouts (GRESO) | arXiv 2506.02177 | full |
| Zhang et al. 2025, SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning | arXiv 2506.09016 | full |
| Gao et al. 2025, Prompt Curriculum Learning for Efficient LLM Post-Training | arXiv 2510.01135 | full |
| Foster et al. 2025, Learning to Reason at the Frontier of Learnability (LILO) | arXiv 2502.12272 | abstract; numbers not verified |
| Chen et al. 2025, Self-Evolving Curriculum for LLM Reasoning (SEC) | arXiv 2505.14970 | full |
| Florensa et al. 2017, Reverse Curriculum Generation for Reinforcement Learning | arXiv 1707.05300 | full (ar5iv) |
| Salimans & Chen 2018, Learning Montezuma's Revenge from a Single Demonstration | arXiv 1812.03381 | full (ar5iv) |
| Xi et al. 2024, Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning (R³) | arXiv 2402.05808 | full |
| Li et al. 2025, QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation | arXiv 2507.13266 | partial (~94%) |
| Amani et al. 2025, RL for Reasoning by Adaptively Revealing Rationales (AdaBack) | arXiv 2506.18110 | full |
| Zombori et al. 2021, Towards Finding Longer Proofs (FLoP) [@zomboriFindingLongerProofs2021] | arXiv 1905.13100 | full (ar5iv; read by me) |
| Matiisen et al. 2017, Teacher-Student Curriculum Learning | arXiv 1707.00183 | full (ar5iv) |
| Portelas et al. 2019, Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments (ALP-GMM) | arXiv 1910.07224 | full (ar5iv) |
| Jiang et al. 2020, Prioritized Level Replay | arXiv 2010.03934 | full (ar5iv) |
| Poesia et al. 2024, Learning Formal Mathematics From Intrinsic Motivation (Minimo) [@poesia2024learningformalmathematicsintrinsic] | arXiv 2407.00695 | full |
| Dong & Ma 2025, STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving | arXiv 2502.00212 | partial (main text, App. A) |
| Zhao et al. 2025, Absolute Zero: Reinforced Self-play Reasoning with Zero Data | arXiv 2505.03335 | partial (reward, ablations) |
| Lee et al. 2025, Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges | arXiv 2502.01612 | full |
| Parashar et al. 2025, Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning (E2H) | arXiv 2506.06632 | full |

**Synthetic problem generation and length generalisation**

| paper | id | read |
|---|---|---|
| Lample et al. 2022, HyperTree Proof Search for Neural Theorem Proving | arXiv 2205.11491 | partial (§3.3, §6.1, §7, App. B.6) |
| Trinh et al. 2024, Solving olympiad geometry without human demonstrations (AlphaGeometry), Nature 625 | DOI 10.1038/s41586-023-06747-5 | partial (main text, most of Methods) |
| Wu et al. 2020, INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving | arXiv 2007.02924 | full (ar5iv) |
| Wang & Deng 2020, Learning to Prove Theorems by Learning to Generate Theorems (MetaGen) | arXiv 2002.07019 | full (ar5iv) |
| Xin et al. 2024, DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data | arXiv 2405.14333 | full |
| Wu et al. 2024, Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation | arXiv 2410.15748 | partial (~100k characters) |
| Huang et al. 2024, MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data | arXiv 2402.08957 | partial (~100k characters) |
| Firoiu et al. 2021, Training a First-Order Theorem Prover from Synthetic Data | arXiv 2103.03798 | full (ar5iv) |
| Anil et al. 2022, Exploring Length Generalization in Large Language Models | arXiv 2207.04901 | full (ar5iv) |
| Zhou et al. 2023, What Algorithms can Transformers Learn? A Study in Length Generalization | arXiv 2310.16028 | full (ar5iv) |
| Zhou et al. 2024, Transformers Can Achieve Length Generalization But Not Robustly | arXiv 2402.09371 | full |

**Small-model RL failure modes and coverage**

| paper | id | read |
|---|---|---|
| Yue et al. 2025, Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? [@yueDoesReinforcementLearning2025] | arXiv 2504.13837 | full |
| Cui et al. 2025, The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models | arXiv 2505.22617 | partial (~100k characters) |
| Cheng et al. 2025, Reasoning with Exploration: An Entropy Perspective | arXiv 2506.14758 | full |
| Chen et al. 2025, Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models | arXiv 2508.10751 | partial (~100k characters) |
| He et al. 2025, Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening | arXiv 2506.02355 | full |
| Zhu et al. 2025, The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning | arXiv 2506.01347 | full |
| Song et al. 2025, Outcome-based Exploration for LLM Reasoning | arXiv 2509.06941 | full |
| Liu et al. 2025, ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models [@liuProRLProlongedReinforcement2025] | arXiv 2505.24864 | full |
| Hu et al. 2025, BroRL: Scaling Reinforcement Learning via Broadened Exploration | arXiv 2510.01180 | full |
| Shenfeld et al. 2025, RL's Razor: Why Online Reinforcement Learning Forgets Less | arXiv 2509.04259 | full |
| Wu et al. 2025, The Invisible Leash: Why RLVR May or May Not Escape Its Origin | arXiv 2507.14843 | partial (~100k characters) |
| Zhang, Neubig & Yue 2025, On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models | arXiv 2512.07783 | partial (~100k characters; abstract re-checked) |

**Reported claims I rely on most, and their limits.**
- **To Backtrack:** the CountDown and Sudoku split is stated in its abstract. The numbers are figure-only.
- **Bae et al.:**
  - the band table is from the body, Qwen2.5-3B, one run per band;
  - the abstract claims up to +12% in under half of standard GRPO's steps. The subagent read the body as measuring the
    +12% against the SFT checkpoint; I did not verify that.
- **Thinkless's collapse:** reported as fewer than 10 long samples by step 120; the denominator is not stated.
- **Zhang, Neubig & Yue:** a 100M model on synthetic dependency graphs, not proofs.
- **None** of these was run on a model as small as 9.5M, except Minimo (8.45M) and Saparov et al. (tiny models).

## 7. How the numbers were computed, and checks run

All of this was CPU-only and read-only, single process, run on 2026-10-08. The scripts are ad hoc, in my session
scratchpad, and are not committed; this note describes them fully.

**§2.1 failure classes.**
- Source: `runs/eval/matched/<set>_<ckpt>/rows_plain_B<b>.jsonl`, using the per-attempt `ok` and `status` fields.
- Status mapping:
  - `ok` = 1 means solved;
  - `status == "done"` with `ok` = 0 means Lean rejected a proof the environment accepted;
  - `fail:S:<why>` means the environment rejected a step;
  - `fail:truncated` means the action never reached `<eos>` within `max_action` 512 (`sample.py`).
- Prefilter classes come from `lean_gate.leanrej.jsonl` (`filter`).
- The textbook 7–12 rows are dev58 records with `reference_lines` in 7–12.
- Hazard per step index s = attempts with `fail_step == s` ÷ attempts that reached s (with `fail_step ≥ s`, or ending
  at or after s).
- Positions use `fail_step / min_lines`, where `min_lines` is the reference length stored in the rows.

**§2.3 headroom (`oracle_effort.py`).**
- Per target, from rows at B 0, 64 and 256, using per-attempt `ok` and `tokens`: think + close + action, all draws.
  Prefill is excluded, as in RL_RESULTS.
- Split-half oracle: choose argmax of pass rate − λ·tokens/1000 on even attempts, with λ = 0 shown and ties going to
  the cheaper level; score on odd attempts.
- Cascade: attempts in order e0, e64, then e256 repeated, each drawn with replacement from that target's stored
  attempts at that level. Stop at the first success, or when the next attempt would exceed the budget (the first
  attempt is always allowed). 40 Monte-Carlo repeats per target, seed 0.
- The i.i.d. arms use the same procedure at one fixed level.
- `oracle_budget.py`: each problem's level is chosen on its even attempts. Draws then come only from that level's odd
  attempts, with the same budget rule and 40 repeats.
- Monte-Carlo noise is small on the 1,000-item sets and about ±0.5 pp on textbook72.

**§2.4 think validity (`think_validity.py`).**
- Source: `runs/restart2_hard12/ei_T1v1e/found_4.jsonl`, 300 records per (round, budget) at B64 and B256, seed 0.
- Each record is replayed in `trace_dsl.Env(prompt, canon=True, base=0, assign=False)`. Every non-empty think block
  goes through `trace_dsl.check_step(env, think, action)` (via `code.thinking.dan.trace_dsl`).
- A block counts as valid if it raises no error; the first error message is tallied.
- The records' `base` is 0, so names match the replay environment.

**§2.5 curriculum.**
- `runs/restart3_grpofix/<run>/metrics.jsonl`: `groups_skipped` and `curriculum.{seen,pairs,frontier}`.
- Weights follow `grpo.Curriculum.weight`: (s+1)/(n+2) and p(1−p) + 0.03.
- Per-target spreads are from the `h12_*` rows (k 16).

**§2.6, §2.8.**
- The same rows: per-problem `n_ok / k`, with lengths from the textbook72 files and the rows' `min_lines`.
- The 21–30 numbers are from `runs/eval/t2130/<ckpt>/rows_plain_B<b>.jsonl` as present at 08:31 UTC: mean `n_ok / k`,
  and the share with `n_ok > 0`, over 2,767 rows each.
- The sealed 13–20 composition is from `runs/pools/test_13_20/test_13_20.jsonl` (`reference_lines`, `family`).

**§2.7.** Row totals: `forced / draws`, `think / draws` and `prefill / draws`.

**§4.1 overlap check.** Renaming-class key: the minimum over the 24 permutations of P Q R S of (sorted premises,
goal). It was compared between `nd-takehome/data/ladder/transfer_long.jsonl` (`source == "textbook"`, 282 items) and
`eval_only/textbook72/*.jsonl` (72 items): 0 matches.

**Checks.**
- Totals per cell were cross-checked against `final_tables.md` on overlapping quantities:
  - hard12 GRPO no-think s0 it78 solved 92.8%, as in the table;
  - 13–20 thinking s1 it78 at B256 gives 10.1% at 678 tokens per attempt on the 1,000-item rows, against 10.1% / 679
    on the table's 986 items.
- The oracle's fixed-level rows reproduce the table's pass@1 and tokens per attempt.
- The validity script reproduced the same pass rates on a second run with different error grouping.
- Not run: `make -C code check`, because this is a documentation-only change.
- Links in this note are relative to this folder.
