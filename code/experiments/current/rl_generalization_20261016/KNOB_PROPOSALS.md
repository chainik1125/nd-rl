---
written_on: 2026-10-16
written_by: agent:claude
---

# Knob proposals: what makes RL generalise in the 9.5M thinking prover

This was a read-only proposal: nothing was run on a GPU and no repo file was edited when it was written. The
supporting analyses are rebuilt as scripts under [`analysis/`](analysis/), with their outputs in `analysis/out/`.

Scope: knobs for the mini test bed (S1g3 base, 2,000 problems, 10 + 10 iterations, seed 0, plus a noise-control
rerun). No lr, optimizer or other low-level training-dynamics knobs. I do not re-propose anything already tested or
planned (see the brief).

**Labels used below.** OBS: observed in existing files (which file, which seed). HYP: a hypothesis, not yet tested.
PRED: what a knob's result would show if the hypothesis is right.

Code read: `nd-rl-v2/code/thinking/grpo.py` (docstring, `rewards`, `advantages`, `attempt_items`, `group_items`,
`Curriculum`, `sample_groups`, argparse), `sample.py` (env set-up, name base), `vendor/danstate/state_env.py`
(rejection causes), `canary.py`, `effort.py`, `feedback.py`, the launcher header `grpo_v2/launch/line.sh`, the
curriculum README, and the test_13_20 / test_21_30 READMEs. Results read: `EXPERIMENT_QUEUE.md`, `RL_RESULTS.md`,
`matched_eval/v2_results_ynez.md` §§ 1, 4, 6, 7, 8, `length_split_ynez.md`, `source_split_ynez.md`.

---

## 0. What existing data already says (zero GPU)

O1 to O3 are documented results. O4 to O10 are my own one-off analyses of existing read rows (`runs/eval/v2/*/rows_plain_B64.jsonl`)
and pool files. They are mostly seed 0 and one checkpoint per arm, with no intervals, so treat them as leads. The
scripts that reproduce O4 to O10 are in [`analysis/`](analysis/).

| # | observation | source |
|---|---|---|
| O1 | RL from S1g3 lifts 13–20 from about 7% pass@1 (Stage 1, serrano) to 30–36%, and textbook from about 42% to about 74%. The pools are 10–15 lines. Most of the L2 → L3 learning happens in L2: L3 starts at a training pass rate of 0.75–0.87 by budget and ends at 0.85–0.95 (merged recipe, seed 0). | OVERNIGHT_20261008, grpo_v2 metrics |
| O2 | GRPO beats EI on the human-written sets (Charles149 +6.5 / +7.0 pass@1). EI beats GRPO on the generated long composites, mostly at pass@5 (−3 to −10 pass@5 at 16–30 lines without the verifier). | length_split_ynez §§ 1a, 2a |
| O3 | KL anchor 0 > 0.02 > 0.1 on the long sets, only under plain reads (level under guided reads). The anchor's data are long-state steps with ≥ 10 hypothesis lines (`canary_long`, built from `long_cands` proofs). | queue; canary.py |
| O4 | **Statement size is far outside the RL pools.** Whitespace tokens of the prompt: L2 / L3 / L3_3p pools max 35 (mean 30–31); t1320 mean 62 (p10 46, max 90); t2130 mean 74 (p10 59); Charles149 mean 32 (p90 48). `max_state` tokens: pools mean 80–92 (max 124); t1320 mean 178 (max 312); t2130 mean 215. Stage 1's K12 statements (`train_k12.jsonl`, source of the S1g3 search traces) are larger: mean 41, 30% over 45 tokens. So RL narrows the size range Stage 1 saw. | pools, test sets, bench data |
| O5 | **At a fixed proof length, statement size predicts failure about as strongly as length does.** On t1320 (merged + KL 0, seed 0), pass@1 for the smallest vs the largest statement tercile (≤ 56 vs > 69 tokens) is 57% vs 21% at 14–15 lines and 49% vs 11% at 18–19 lines. Across lengths 12–21, the smallest tercile only falls from 67% to 36%. EI + verifier shows the same pattern. Binder-mismatch failures rise from about 11% to 27% of attempts with size. This is observational: size is confounded with premises, connectives and padding. | my analysis |
| O6 | **On the long sets, attempts fail on a step-level invalid action, rarely by running out of room.** On t1320 / t2130: env rejections 48–63% / 77–90% of attempts (binder mismatch about 15% / 30%, unbound 12% / 18–20%, Or.elim on a non-or 5–14%); env-accepted but Lean-rejected 12–21% / 7–16% (mostly app-arg / app-result). Solved proofs are 1.5–1.7× the minimal line count. Failures happen at about 0.8–1.3× min_lines steps. | my analysis, all v2 arms |
| O7 | **High name indices are not a factor.** I rebuilt each attempt's random name base exactly (1,000 / 1,000 rows match the accepted texts). Within a problem, pass@1 is flat across bases 0–10 / 11–21 / 22–32, with no trend in base: within ±0.5 points for the GRPO reads on t1320, t2130 and Charles, and −1.2 to +0.9 for EI + verifier on t1320. So a name-range knob is ruled out. | my analysis |
| O8 | **Training with the logical stop arm did not lower the Lean-reject share at test.** That arm penalises Lean-invalid steps at the step. Seed-paired with plain GRPO on t1320: 21.0 vs 21.1, 12.5 vs 11.8, 12.4 vs 16.0. The init seed matters more (seed 0 about 20%, seed 1 about 12%). | my analysis |
| O9 | EI knob pilots (seed 0, pass@1, from summary.json, no intervals). 13–20: rr 30.8, ret0 (no Stage-1 replay) 28.4, r2 32.2, t10 31.1, 3p 30.3. 21–30: rr 6.5, ret0 6.2, r2 7.6. Charles173 (all 173, not the clean 149): rr 71.5, ret0 69.6, r2 72.8. Removing EI's anchor to Stage 1 *hurt*, whereas removing GRPO's KL anchor helped. | my read of runs/eval/v2 |
| O10 | The g64 pilot visited 1,635 distinct L3 pairs vs about 5,900 for the merged recipe. But kl0 (best) visited 5,416 and pb (worst) 7,575, so the count of distinct pairs alone does not explain the pilots. | grpo_v2 metrics |

**Working hypotheses the knobs test**

- **H1, state coverage.** What fails out of distribution is per-step validity on states bigger than any RL state:
  long formulas to copy (binder mismatch) and many hypotheses to look up (unbound). RL generalises only as far as
  its training states reach, and the pools stop at about 35-token statements and about 124-token states (O4–O6).
- **H2, structure / strategy.** The composites need lemma reuse (cut lemmas, ∨E results reused in 65–85%). The
  model re-derives instead, so its proofs and states grow. What transfers is a proof strategy, learned only on
  problems that need it.
- **H3, sharpening vs diversity.** GRPO concentrates mass on proofs it already finds. That helps the human-written
  sets but costs pass@5 on the generated composites (O2). Whatever keeps diversity transfers to the composites.
- **H4, distance from S1.** RL has to move S1's behaviour on large states (O3), but staying close to S1's broad
  data helps EI (O9). Which kind of anchoring helps or hurts depends on *where* it anchors.
- **H5, locus.** RL's transferable change sits in the action policy or in the think (search) policy.

## 1. How to read every mini run (cheap, applies to all knobs)

The headline numbers cannot separate these mechanisms. These reads mostly can, and most are analysis only:

1. **Reads:** the S1g3 base (so each arm's gain over its own start is known); L2 / L3 `test_1k` (in distribution);
   t1320 split into 13–15 and 16–20; t2130; Charles149. Thinking knobs also read at B0.
2. **2 × 2 decomposition:**
   - L3 test_1k (same generator, same length) vs t1320 13–15 (composite, same length) gives the *structure* gap;
   - t1320 13–15 vs 16–20 gives the *length* gap at fixed structure.
3. **Statement-size terciles within each length bin** (the O5 table, with paired bootstrap intervals). It shows
   whether a knob moves size generalisation or length generalisation.
4. **Failure-cause shares:** binder mismatch, unbound, Or.elim on a non-or, Lean reject. Also fail_step /
   min_lines, steps / min_lines of solved proofs, and repeated formulas across branches. Each hypothesis predicts
   which class should move.
5. **KL(π_RL ‖ π_S1)** on held-out L3 states and on t1320 states: a distance covariate, as in RL's Razor.

Each prediction below is a pattern across these reads. A pattern holds up better against the mini's run-to-run noise
than a single Δ does.

## 2. Proposals

Cost is in mini runs (about 1 GPU-h of training plus reads each). "No code" means flags or a new targets file only.

### A. Training-state distribution (data)

**A1. Statement-size augmentation by atom substitution, proof skeletons unchanged** ★
- *Change.* For half the mini's 2,000 rows (a fixed hash), apply a substitution σ: each atom maps to a small
  non-degenerate formula over P, Q, R, S, padding forms included (A ↦ A ∨ A, A ↦ A ∧ (A ∨ B), A ↦ (S → A)). Target
  45–90 statement tokens (t1320's range). Apply σ to every line of the reference proof. ND rules are schematic, so the
  proof stays valid with the same line count.
  - Lean-check every row.
  - Reject rows with inconsistent premises or a tautological conclusion (the test recipe's rules).
  - Re-run `excl.py`, `heldout.py` and the textbook-instance shape gate.
  - Keep `min_lines` as an upper bound, optionally re-certified with minlen at ≤ 15 lines.
  - Feed the rows through `--targets`, or keep them on their own curriculum with `--targets_extra` and
    `--targets_extra_frac 0.5`.
- *Mechanism (H1).* RL's step skills (copying a binder formula, citing the right hypothesis) are practised only on
  statements of ≤ 35 tokens. Same proofs with bigger formulas isolate size from length and from structure.
- *PRED, supports H1:*
  - t1320 and t2130 rise, mostly in the large-statement terciles of each length bin;
  - binder mismatch falls;
  - the 13–15 composite bin gains about as much as 16–20;
  - L2 / L3 test_1k stay flat, and Charles stays flat (its statements are small).
- *PRED, refutes H1:* no gain, or a gain spread evenly across terciles. An even gain would mean more varied data
  helps, not size as such. Gains on t1320 but not t2130 would mean size coverage only interpolates up to the trained
  size.
- *Cost.* CPU build, about half a day of agent work; 1 mini run.
- *Confounds.*
  - σ also adds a fourth atom and padding shapes, both test features. Split later into padding-only vs fresh-atom σ.
  - Substituted rows may be harder, which means more learning signal; report training pass rates.
- *Literature.* I found no direct precedent for substitution augmentation in RL for provers. Related, loosely:
  - Zhou et al. 2023: length generalisation depends on whether the per-step computation is length-invariant;
  - Zhang, Neubig & Yue 2025: contextual generalisation needs some exposure first.

**A2. Hypothesis-count augmentation: one unused premise** (variant of A1)
- *Change.* Add one consistent, unused premise over the problem's atoms to half the rows, after checking that it
  does not shorten the minimal proof. Rows stay at ≤ 3 premises.
- *Mechanism (H1, lookup half).* More hypothesis lines in every state at fixed formula size. This targets unbound
  and Or.elim-on-non-or (lookup) errors, where A1 targets binder mismatch (copy) errors.
- *PRED.* Lookup-error shares fall on the long sets and binder mismatch does not. A1 should show the opposite.
- *Cost.* CPU; 1 run.
- *Confound.* The sealed sets never have unused premises, so the model may learn to ignore premises. Watch Charles.
  The 3-premise rewrites (x66) do not test this: uncurrying turns a box assumption into a premise and leaves the
  hypothesis count about the same.

**A3. Late-state exposure: start mid-proof on long proofs (reverse curriculum), with a matched SFT arm** ★
- *Change.* Source rows: `long_cands` / `long2_cands` rows not used by `canary_long` (about 1,870; 13–20 lines,
  statements mean 47 tokens, states up to 328), or `cands_clean` rows ≥ 16 lines after exclusion.
  - Each attempt replays the first j reference actions in the env, using the attempt's name base, and samples from
    there.
  - Reward: Lean accepts the prefix plus the completion. Prefix tokens are prompt, not loss.
  - j starts close to the end and slides backward over training (R³ style); the remaining length is 4–12 lines.
  - Mix in as a third of the groups.
  - Control arm A3-SFT: the same start states, teacher-forced cross-entropy on the reference remainder at matched
    tokens, with no RL.
- *Mechanism (H1 vs H2).* Puts the policy on-policy in the large, late states that pool attempts never reach,
  without the horizon of a full long proof. RL vs SFT on identical states tests whether on-policy credit, or mere
  exposure, is what transfers.
- *PRED, supports H1:*
  - t2130 gains more than t1320;
  - failures move later (fail_step / min_lines rises) and unbound / binder shares fall;
  - in distribution stays flat.
- *PRED, other outcomes:*
  - RL > SFT on the long sets: on-policy credit transfers (Chu et al. 2025);
  - SFT ≥ RL: exposure is enough;
  - no change: late-state coverage is not binding, so look at strategy (H2).
- *Cost.* Medium code: `think_generate` takes per-prompt prefix actions, the prefix is replayed with the base, the
  judge sees prefix plus completion, plus tests. Data filter. 2 runs.
- *Confound.* These are 13–20-line problems, so the arm partly is "train on longer problems". Compare it with the
  planned 16–18 pool.
- *Literature.*
  - Florensa et al. 2017 (start near the goal, move back);
  - Salimans & Chen 2018 (reset to states of one demonstration);
  - Xi et al. 2024, R³ (slide the start back along a reasoning demonstration).

**A4. Short composites at in-distribution length** (the earlier "Option A"; needs Leon's yes)
- *Change.* Run `test_21_30/build_composites.py --lo 10 --hi 15` on pieces no sealed composite uses
  (`sealed_parts.jsonl` lists 7,814 parts to avoid). Replace half the mini pool with these.
- *Mechanism (H2).* Cut and lemma-reuse structure at training lengths. This separates structure from length (Yuan et
  al. 2025 report RL learning compositions of known skills).
- *PRED, supports H2:*
  - t1320 13–15 (composite, same length) gains at least as much as 16–20;
  - repeated formulas across branches fall and steps / min_lines falls;
  - L3 test_1k and Charles stay flat.
- *Confound.* Composite statements are big (t1320 p10 is 46 tokens), so A4 also moves size. Read it against A1 at
  matched statement-size terciles.
- *Cost.* CPU (the builder exists); 1 run.
- *Literature.* Yuan et al. 2025; Dziri et al. 2023 (limits on compositionality).

**A5. Pool size and repetition at a fixed attempt count** ★
- *Change.* Targets are fixed random subsets of N = 250 and N = 500 of the mini's 2,000 (the same L2 / L3 split),
  with the same iterations, prompts and group size. Add a read of the training subset itself, so the train vs
  held-out gap is measured.
- *Mechanism (skill vs coverage).* If RL teaches step skills that do not depend on the problem, a small pool keeps
  most of the long-set gain while the train vs held-out gap widens. If the variety of problems drives transfer, the
  long sets fall faster than in distribution. In LLMs one example can suffice (Wang et al. 2025), but there
  pretraining carries the skill; a 9.5M model may not behave that way.
- *PRED.*
  - N = 250 ≈ 2,000 on long sets and Charles: transfer is skill-based;
  - long sets drop more than test_1k: transfer depends on coverage;
  - the training subset's pass rate measures memorisation directly.
- *Cost.* No code; 2 runs.
- *Confound.* Small pools saturate, so more groups are all-solved and skipped. Report `groups_skipped` and
  `with_grad`; `--refill` can hold the gradient groups constant.
- *Observation link.* This separates the "fewer distinct problems" half of g64's −5 from its group size (O10).

**A6. Difficulty band by the Stage-1 pass rate**
- *Change.* Label candidates from the 10k L2 / L3 pools with one S1g3 read (k 8, B64). Build N = 2,000 *hard*
  (p ≤ 0.25, 0 included) and *easy* (p ≥ 0.5) pools with the same L2 / L3 split. Online, `--max_pass` drops only
  the easy end; a minimum pass rate needs an offline file or a small flag.
- *Mechanism (H3).* Learning from rare discoveries vs sharpening proofs the model already finds. Zhang, Neubig &
  Yue 2025 and Bae et al. 2025 find intermediate or edge-of-competence data most useful.
- *PRED.*
  - hard: larger long-set and pass@5 gains, test_1k pass@1 lower;
  - easy: test_1k pass@1 up, long sets flat, pass@5 down (sharpening).
- *Cost.* One S1 read plus a data filter; 2 runs.
- *Confound.* Difficulty tracks features (classical-only, nested ¬, statement size). Report the feature shares of
  both pools.
- *Literature.* Hase et al. 2024 (easy data can carry to hard tasks) is the counterpoint.

**A7. Length mix and curriculum order**
- *Change* (no code): L3 only for 20 iterations; L2 only for 20; reversed (L3 then L2); one shuffled L2 + L3 file.
- *Mechanism.* Whether length transfer depends on the longest training length or on the easy-to-hard order.
- *PRED.* If the maximum length matters: L3-only ≈ L2→L3 > L2-only on 16–20, with order irrelevant.
- *Expected information: low.* L3 is 82% 13-line (+2.2 lines over L2), so the range is small. This is better done
  with the planned 16–18 pool.
- **Check before that pool is used.** The curriculum builder gates compact statements (≤ 7 connectives, 2–3 atoms,
  depth ≤ 3). Kept, that gate makes the 16–18 pool a pure length knob, orthogonal to A1. Relaxed, it confounds length
  with size. A1 and the 16–18 pool together give a clean 2 × 2 of size × length.

**A8. Surface-form mix for Charles (goal types, 0 premises)**
- *Change.* Add curried rewrites of pool rows (all premises moved into the goal, giving 0-premise problems: Charles
  20% vs pools 4%). Add A ∧ B-goal rows from pfgen goal weights or `cands_clean` (Charles 36% vs pools 1%), after
  exclusion.
- *Mechanism.* Source transfer as coverage of surface forms.
- *PRED.* Charles gains concentrated in those categories; long sets flat.
- *Power is weak.* The categories have n ≈ 30–54 in Charles149. GRPO's lead over EI on Charles is broad across
  categories (source_split), which hints that a general mechanism, not surface coverage, is at work. Rank low.

### B. Reward and credit assignment

**B1. Positive-only vs negative-only reinforcement** ★
- *Change* (about 15 lines in `group_items`):
  - PSR keeps only items with A > 0 and drops the step penalty;
  - NSR keeps only items with A < 0 and keeps the step penalty;
  - optionally W-REINFORCE, which scales positive advantages by 0.1.
- *Mechanism (H3).* Zhu et al. 2025 find that positive reinforcement raises pass@1 and lowers pass@k, while
  negative-only training keeps pass@k by moving mass to the other plausible candidates.
- Here GRPO's deficit on the generated long sets is at pass@5 (O2), but EI, which is positive-only, keeps pass@5
  better, against the paper's direction. That disagreement is what makes the knob informative.
- *PRED.*
  - If sharpening costs transfer: NSR ≥ baseline at pass@5 on 16–30 with lower test_1k pass@1, and PSR the reverse.
  - If reinforcing whole successful proofs is what transfers: PSR ≈ baseline > NSR on the long sets.
  - Charles, where GRPO beats EI, may favour PSR.
- *Cost.* Small code; 2 runs.
- *Confounds.* The unlikeliness reward acts only on positive rewards, so it is inert under NSR. Ten plus ten
  iterations may be short for diversity effects.
- *Literature.* Zhu et al. 2025; Yue et al. 2025 (RLVR narrows coverage at large k).

**B2. Concision reward: lines beyond the certified minimum**
- *Change.* A success pays 1 − λ · max(0, steps − m) / m, with m the row's `min_lines` and λ = 0.25. Failures pay 0.
  Tiny code: `rewards()` needs the target's m.
  - First check, analysis only: training attempts' steps / m on L2 / L3. If they are already near 1, the knob has
    nothing to act on.
  - `--len_penalty` is a no-code approximation, but it also charges think tokens (about half the tokens at B64), so
    it is confounded with the planned think-length arms.
- *Mechanism (H2).* O6: solved long proofs are 1.5–1.7× minimal. About 43–51% of `have` lines on the generated sets
  repeat a formula stated elsewhere in the proof (25% on Charles), while adjacent duplicates are only about 1%. HYP:
  this is re-derivation in each branch instead of hoisting the lemma once, which enlarges states and adds failure
  chances. A concision reward should teach hoisting.
- *PRED.* steps / min_lines and cross-branch repeats fall on t1320 / t2130, and long-set pass rises with test_1k flat.
  Refuted if the lengths fall and the pass rate does not.
- *Cost.* Tiny code; 1 run.

**B3. Progress credit from rollouts at intermediate states (VinePPO style)** (after A3's code)
- *Change.* For a subset of attempts, sample m = 4 continuations from 2–3 intermediate states. Advantage per segment
  = the value difference (Monte Carlo progress).
- *Mechanism.* Credit goes to the decisive steps rather than spreading over the whole attempt.
- *PRED.* Gains concentrated on long problems, where outcome credit is most diluted.
- *Prior.* O8 says step-localised validity penalties did not cut Lean errors at test, so I expect a modest effect.
- *Cost.* Medium-large code; about 1.5–2× the sampling per iteration (fewer groups at matched compute). Rank low for
  the mini.
- *Literature.* Kazemnejad et al. 2024 (VinePPO); Setlur et al. 2024 (process reward as progress under a prover
  policy).

**B4. Explicit diversity objective** (low priority; overlaps the kept unlikeliness reward)
- *Change.* Either the analytic pass@k advantage of Chen et al. 2025, or a historical novelty bonus per problem on
  proof skeletons, after Song et al. 2025.
- *PRED.* pass@5 up on long sets and Charles, pass@1 in distribution about level.
- *Cost.* Small to medium code; 1 run.

### C. Thinking

**C1. Where the gradient enters: action tokens only vs think tokens only** ★
- *Change* (about 20 lines in `_step_mask`; `--rej_penalty_scope action` already has an action-only path for
  penalties): arm *act*, loss on action tokens only (think tokens and `</think>` masked); arm *think*, loss on think
  tokens only.
- *Mechanism (H5).* Separates RL improving the step policy from RL improving the search policy. The failure classes
  in O6 are action-level copy and lookup errors, and training pass rates rise as much at B0 as at B64
  (0.16 → 0.79 at B0).
- *PRED.*
  - *act* ≈ full on every set and *think* far below: the transferable change is in the action policy.
  - *think* keeping a large share of the long-set and Charles gain relative to its in-distribution gain: RL transfers
    through the think block.
  - Read at B0 too: only *act* should lift B0.
- *Confound.* A loss mask is not a parameter freeze; shared weights still move. Read the result as where the
  gradient enters. Watch think-length drift under *think*.
- *Cost.* Small code; 2 runs.
- *Literature, loosely.* Wang et al. 2025: gradients on a small subset of tokens drive most of RLVR's gain.

**C2. Training budget mix** (no code: `--budgets`)
- *Change.* Arms `--budgets 0` and `--budgets 256` vs the default `0,64,256`. Read at B0 and B64 on t1320 and
  Charles.
- *Mechanism (H5).* A transfer matrix between practising with thinking and without.
- *PRED.* If thinking carries the transfer: B256-only training gives the best B64 long-set reads, and B0-only
  transfers poorly to B64. If RL works through actions: B0-only ≈ default at B64.
- *Cost.* No code; 2 runs plus extra B0 reads.

**C3. Per-step think dropout** (low priority)
- *Change.* Within a B64 attempt, close the think block at 0 tokens on a random 50% of steps. The effort token is
  kept, since S1g3's effort token is constant within an attempt.
- *Mechanism.* Forces the action policy to work without thinking at some steps; robustness at large states.
- *Cost.* Small sampler code; 1 run.
- *Literature.* Su et al. 2024 (Dualformer: randomly dropped trace parts).

### D. Distance from the Stage-1 model and off-policy data

**D1. Where the KL anchor sits: short in-distribution states vs long states**
- *Change.* Build a canary/1 file from L2 / L3 reference steps with ≤ 6 hypothesis lines. `build_canary.py`'s sources
  are hard-coded to `long_cands` and it has only `--min_hyp`, so it needs a source option and `--max_hyp`: small code.
  - Arm *short*: `--kl_anchor_coef 0.02 --kl_anchor_data SHORT --kl_anchor_ref S1g3`.
  - Arm *long*: the same coefficient on the default long-state data (the tested condition, at mini scale).
  - Baseline: the recipe's KL 0.
- *Mechanism (H4).* Whether the anchor hurts because it holds S1's large-state behaviour in place, or because any
  closeness to S1 limits RL.
- *PRED.*
  - short ≈ KL 0 on the long sets (possibly better on test_1k or Charles), and long < KL 0: RL's gain needs freedom
    on large states;
  - both anchors hurt equally: a generic distance effect.
- *Cost.* Small data and code; 2 runs (*long* can be skipped if the full-scale result is trusted).
- *Literature.* Shenfeld et al. 2025 (KL to the base predicts forgetting); Liu et al. 2025, ProRL (KL plus
  reference resets).

**D2. Stage-1 replay inside GRPO (EI's `--retain`)**
- *Change.* At each AdamW step add λ × cross-entropy on a minibatch of S1g3 pairs (think + action), through the same
  per-step auxiliary-loss hook the KL anchor uses. λ is set so the replay is about 20% of tokens, like EI's 20,000
  retained records. Small to medium code.
- *Mechanism (H3 / H4).* O9: EI without its replay lost 2.4 points on 13–20 at seed 0. HYP: replay keeps S1's broad
  search behaviour, and that may explain part of EI's pass@5 lead on the composites.
- *PRED.* GRPO + replay moves toward EI: pass@5 up on t1320 / t2130 and test_1k pass@1 slightly down. If Charles
  drops at the same time, the two kinds of transfer trade off: GRPO's source edge comes from moving away from S1.
- *Cost.* 1 run.
- *Literature.* Chu et al. 2025 (SFT vs RL); Yan et al. 2025, LUFFY (off-policy guidance in GRPO).

**D3. Off-policy long-proof traces in groups** (low priority)
- *Change.* For long problems (16–20-line `long_cands` rows not in the canary), add one honest-search trace
  (`searchtrace`) per group as an off-policy success with a regularised importance weight (LUFFY). A cheaper relative
  is self-imitation replay of the model's own rare long-proof successes (Oh et al. 2018).
- *Mechanism.* Tests whether long-proof failure is limited by exploration: on-policy sampling rarely finds a success
  to learn from.
- *PRED.* Long-set gains with in distribution flat, if exploration is the limit.
- *Cost.* Trace generation plus medium code.

### E. Verifier in training vs at test time

**E1. Mixed-arm groups: half guided redraws, half the stop arm** (cheap proxy for A3)
- *Change.* Per group, 8 attempts on the logical arm with redraws (`--max_rej 10`, penalty on the redrawn draws) and
  8 on the stop arm (`--max_rej 0`), with one baseline over the group. Small code: two `think_generate` calls per
  budget.
  - Not tested before: on 2026-10-10 "half of each group sampled plainly" was left as an alternative for Leon.
- *Mechanism.* Redraw attempts survive invalid steps and reach late, large states of in-distribution problems: late
  states on-policy without the new env code. Also tests what the verifier should be used for in training.
- *PRED.*
  - Plain long-set reads rise if late-state coverage matters.
  - Guided reads rise while plain reads fall: the model relies on the verifier. The all-redraw arm collapsed under
    plain reads (4.4–14.9% on 13–20).
- *Cost.* Small code; 1 run. Watch action entropy and the plain/guided gap.

## 3. Top 5 by information per GPU-hour

| rank | knob | runs | code | why it ranks here |
|---|---|---|---|---|
| 1 | **A1** statement-size substitution | 1 | CPU data build only | Tests the strongest lead in the data (O4, O5): the training statements are all smaller than the smallest t1320 tercile. Proof skeletons stay fixed, and the prediction names the error class (binder mismatch) and the cell (large tercile) that should move. |
| 2 | **C1** loss on action only vs think only | 2 | ~20 lines | Locates RL's transferable change (H5). Each outcome is interpretable, and the result tells which thinking knobs are worth pursuing. |
| 3 | **A5** pool size 250 / 500 vs 2,000 | 2 | none | Separates skill from coverage of problems at fixed compute. The train vs held-out gap measures memorisation directly. It also separates g64's confound (O10). |
| 4 | **B1** positive-only vs negative-only | 2 | ~15 lines | Sharpening vs diversity (H3), tied to the clearest GRPO-vs-EI pattern (pass@5 on composites, O2). The literature predicts a direction, and EI's behaviour contradicts it. |
| 5 | **A3** mid-proof starts + SFT control (or **E1** as a 1-run proxy) | 2 (1) | medium (small) | Separates late-state coverage from horizon, and RL from SFT on identical states. More code, but no other knob tells whether reaching large states on-policy is what transfers. |

Runners-up:
- **D1**: cheap, and it refines the one knob that showed a dose response;
- **A4**: high information, but needs Leon's yes on composites in training;
- **D2**: tests whether EI's replay explains its pass@5 lead;
- **A6**: needs an S1 read.

Suggested first wave: A1, C1 (both arms), A5 (N = 250) and B1 (both arms), plus the rr noise rerun. That is about
7 mini runs, about 1 day on 4 GPUs. Before any of it, the zero-GPU readouts in § 1 should be extended to every
existing v2 arm with intervals.

## 4. Existing grpo.py flags vs proposals

| proposal | existing flags | what is missing |
|---|---|---|
| A1, A2, A4, A5, A6 (offline), A7, A8 | `--targets`; `--targets_extra` + `--targets_extra_frac` (own curriculum) | new data files only |
| A6 (online, easy end) | `--max_pass` (drops pairs above x; no floor) | a minimum-pass filter |
| A3 mid-proof starts | none | prefix replay in `think_generate`; judging prefix + completion |
| A5 saturation control | `--refill`, `--refill_max` | — |
| B1 PSR / NSR | none (`--rej_penalty` is a negative-only signal for invalid draws) | advantage-sign masking |
| B2 concision | `--len_penalty` (tokens, think included: an approximation) | a line-based term using `min_lines` |
| B3 progress credit | none | rollouts from intermediate states (after A3) |
| B4 diversity | `--unlikely_beta` (already in the recipe) | pass@k advantage / novelty bonus |
| C1 action / think loss | `--rej_penalty_scope action` (penalties only) | a loss mask for accepted items |
| C2 budget mix | `--budgets`, `--effort`, `--force_mode` | — |
| C3 think dropout | `--force_mode exact` (per attempt only) | per-step budget |
| D1 anchor location | `--kl_anchor_coef`, `--kl_anchor_ref`, `--kl_anchor_data`, `--kl_anchor_n` | a short-state canary/1 file (`build_canary.py` sources, `--max_hyp`) |
| D2 Stage-1 replay | none (the KL anchor's per-step hook is the template) | a cross-entropy term on S1 pairs |
| E1 mixed arms | `--arm`, `--max_rej`, `--rej_penalty`, `--rej_penalty_scope` (per run) | a mix within a group |
| monitoring for § 1 | `--monitor`, `--canary`, `--dump_batches` | — |

## 5. Considered and not proposed

- **Name-index range in training** (bias the name base high): ruled out by O7.
- **Process penalties on invalid steps beyond `--rej_penalty`:** p3 was within noise, and O8 shows step-localised
  validity penalties do not reduce Lean errors on the long sets. B3 is the only credit-assignment variant kept, at low
  priority.
- **State format** (goalstack vs flat), model size, Stage-1 data mix: they need a new Stage 1, outside the mini.
- **Generator in the loop** (frontier problems made during RL, as in STP; Dong & Ma 2025): too large for the mini.
  Worth revisiting if A1 or A5 show that problem coverage is binding.

## 6. References (each checked by web search on 2026-10-16; claims kept to the abstracts)

- Chu, Zhai et al. 2025. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training.
  arXiv:2501.17161.
- Zhu, Xia, Wei, Chen, Chen, Meng 2025. The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning.
  arXiv:2506.01347.
- Yue, Chen et al. 2025. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base
  Model? arXiv:2504.13837.
- Xi et al. 2024. Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning
  (R³). arXiv:2402.05808.
- Florensa, Held, Wulfmeier, Zhang, Abbeel 2017. Reverse Curriculum Generation for Reinforcement Learning.
  arXiv:1707.05300.
- Salimans & Chen 2018. Learning Montezuma's Revenge from a Single Demonstration. arXiv:1812.03381.
- Yuan, Chen et al. 2025. From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones.
  arXiv:2509.25123.
- Dziri et al. 2023. Faith and Fate: Limits of Transformers on Compositionality. arXiv:2305.18654.
- Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran 2023. What Algorithms can Transformers Learn? A
  Study in Length Generalization. arXiv:2310.16028.
- Zhang, Neubig, Yue 2025. On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models.
  arXiv:2512.07783.
- Wang, Yang, Zeng et al. 2025. Reinforcement Learning for Reasoning in Large Language Models with One Training
  Example. arXiv:2504.20571.
- Bae, Hong et al. 2025. Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning.
  arXiv:2504.03380.
- Hase, Bansal, Clark, Wiegreffe 2024. The Unreasonable Effectiveness of Easy Training Data for Hard Tasks.
  arXiv:2401.06751.
- Shenfeld, Pari, Agrawal 2025. RL's Razor: Why Online Reinforcement Learning Forgets Less. arXiv:2509.04259.
- Liu, Diao et al. 2025. ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language
  Models. arXiv:2505.24864.
- Yan, Li et al. 2025. Learning to Reason under Off-Policy Guidance (LUFFY). arXiv:2504.14945.
- Oh, Guo, Singh, Lee 2018. Self-Imitation Learning. arXiv:1806.05635.
- Kazemnejad, Aghajohari et al. 2024. VinePPO: Refining Credit Assignment in RL Training of LLMs.
  arXiv:2410.01679.
- Setlur et al. 2024. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. arXiv:2410.08146.
- Chen, Qin, Wu, Ling, Ye, Zhao, Shi 2025. Pass@k Training for Adaptively Balancing Exploration and Exploitation of
  Large Reasoning Models. arXiv:2508.10751.
- Song, Kempe, Munos 2025. Outcome-based Exploration for LLM Reasoning. arXiv:2509.06941.
- Wang et al. 2025. Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for
  LLM Reasoning. arXiv:2506.01939.
- Su, Sukhbaatar, Rabbat, Tian, Zheng 2024. Dualformer: Controllable Fast and Slow Thinking by Learning with
  Randomized Reasoning Traces. arXiv:2410.09918.
- Dong & Ma 2025. STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving. arXiv:2502.00212.
