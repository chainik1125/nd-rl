---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - yueDoesReinforcementLearning2025
  - liuProRLProlongedReinforcement2025
  - donoway2026excessdescriptionlengthlearning
---

# Literature review 2: create vs elicit since 2025, exploration outside support, model organisms (run `lit-review-2`, 2026-10-02)

This review extends `lit-review` (2026-09-29, 58 papers), which is not re-screened here. **51 new papers were screened**
(48 at abstract level or deeper, plus 3 abstract-only id checks) and **11 were read in depth** (`notes/`). The count is
above the brief's 25–40: three parallel screening agents each took 12–17, and I added four to cover two threads the
brief names. Every quote was fetched into `~/lr_sources/` and grepped. The three agents' ledgers (108 claims; 5 marked wholly or partly
UNVERIFIED *[reviewed: V3]*) and my independent re-check (33 / 33 verbatim) are in `claims.md`. No model was run, so no number here is
new. Where our runs are cited: `trajectory` and `rl-from-ckpt` used best-cap12, a 9.56 M `lean_staten` model trained from
scratch on K12 (cap 12), 3 seeds, Lean alone. `trajectory-cap6` is the cap-6 sibling.

> **Librarian's note (2026-10-02).** This is the reviewed copy of the fork's `lit_review_2/REVIEW.md` (branch
> `dan_lit-review-2`, `7df1dd58`). The review
> ([`review_lit-review-2.md`](https://github.com/dan-pandori/nd-takehome/blob/35b6eef4/review_lit-review-2.md))
> re-fetched every cited arXiv version with its own code: 45 / 45 versions exist with the cited titles, and 106 / 106
> checkable ledger claims are verbatim at the cited version. Verdict: **stands**, with seven rewordings. V1–V6 are
> applied here and in `notes/`, each marked *[reviewed: V…]*: V1 (our A/B split is equal-k at k 256, not an unequal
> budget), V2 ("always" / "only" narrowed to pretrained LLMs; on-policy RL alone reaches base-pass@1024 = 0 problems
> in from-scratch organisms), V3 (108 claims, 5 UNVERIFIED), V4 (expectation #7 held), V5 (n for 2607.07646v1: 16
> problems per bucket, 3 seeds), V6 (an exact quote in the EDL note). V7 (`log.md` times) concerns the fork's log,
> which is not copied. `claims.md` and `screened.md` are copied unchanged; copyrighted source texts are not.

## Bottom line

1. **The standard is still Yue et al.'s equal-k pass@k** (2504.13837v5). It is a finite-budget support test. Its own
   likelihood evidence covers two problems (Fig. 6 caption). *[reviewed: V1]* Our A/B/C split is the same equal-k design at k 256: groups
   come from r0 and r8, both read at k 256 on evaluation-only pools (textbook72 + holdout250). What it lacks is the
   large k (Yue: 1,024 on AIME24) and a large-k base curve on B ∪ C, and its groups come from one sample seed.
2. **The convincing creation designs control pretraining and intervene on it.** The closest analogue to us is
   Abdulsalam, Patel & Saxe (2607.07646v1). It pretrains from scratch on a rewrite grammar that excludes the shortcuts.
   "even at pass@1024, the pretrained policy gets 0% on Buckets 4–5, while RL later solves them at pass@16", and "RFT
   then plateaus". *[reviewed: V5]* Each bucket has 16 problems and the curves are 3 seeds, so "0 % at pass@1024" is a
   weak support statement, the same caveat as Yue's two-problem Fig. 6. It reports no likelihoods, which are exactly what we can add.
3. *[reviewed: V2]* **The dedicated exploration methods with out-of-support evidence draw on a source other than the
   policy's own samples:** random actions after returning to an archived state (Go-Explore), enumerated actions with root noise (AlphaZero), heuristic
   injection (DeepHOL-Zero), oracle prefixes (POPE), or a frontier LLM (AlphaEvolve). In pretrained LLMs, on-policy
   bonuses do not move zero-reward problems and external sources do (2601.18779v1). In small from-scratch organisms, though,
   on-policy RL alone does reach problems the base gets 0 % on at pass@1024, through iterated shifts: plain GRPO in 2607.07646v1 (16 problems per bucket, 3 seeds),
   and policy gradient that "can explore and discover new correct paths that were absent from the initial training set"
   (2509.22613v2). Neither measures base support, but neither do the methods listed above.
4. **For group C and proposal 20,** the best-supported additions are:
   - start episodes from reference-proof prefixes (POPE, and Go-Explore's backward algorithm);
   - keep ε on for the whole run (DeepHOL-Zero: 7.0 % without exploration, 56.3 % with, 1905.10501v3 Fig. 4).

## Q1. Create vs elicit, 2025–2026

**Evidence.**
- **Yue** finds that the base surpasses RLVR "as k increases". Base-unsolved, RL-solved problems are 0.0 % (AIME24,
  k 1,024) and 1.0 % (MATH500, k 128) (Table 2).
- **ProRL** (2505.24864v1) claims gains "where base models fail entirely". Those gains are largest where the base is
  weakest, on Reasoning Gym tasks where "the base model struggles with formatting" (§4).
- **Successors** make the boundary depend on training length and width:
  - early shrinkage, later expansion (2510.04028v1);
  - BroRL "revives models saturated after 3K ProRL training steps" by widening rollouts (2510.01180v1).
- **"Mirage or Method?"** (2508.21188v2) finds that one-shot RL, noisy rewards and negative-only training work "only
  when the model and task already exhibit strong model-task alignment", measured by pass@k. That bears on the
  negative-gradient thread (Zhu 2506.01347; sample polarity 2512.21625v1, where "negative samples encourage exploration
  of new reasoning paths").
- **Power sampling's successors** make the sharpening null cheaper. A token-level approximation "matches or surpasses
  one-shot GRPO" at more than 10× less latency than MCMC (2601.21590v1). The power distribution is "the closed-form
  optimizer of KL-regularized RL when the model's sequence-level log-probabilities are used as the reward" (2605.04542v1).
- **Theory.** Sharpening cannot "create information that is not already in the model" (2412.01951v2). Coverage is
  "necessary and sufficient for Best-of-N" (2510.15020v2), and coverage "lower bounds the runtime" of RL
  (2503.07453v2). Coverage is **sequence-level**: mass on correct outputs ≥ ≈ 1/k.
- **Excess description length** (2601.04728v1) predicts a shape test: EDL per token falls with data for elicitation
  and rises first for teaching. Its empirics are in a companion paper I could not find (UNVERIFIED), and it is "SFT
  only": "Extensions to reinforcement learning … require additional development".

**Designs, ranked by how well they separate creation from elicitation:**
1. Controlled pretraining plus an intervention that adds or removes the skill. Examples: 2607.07646v1 (ρ sweep),
   Tsilivis 2510.11495v2 (p_cot dial), Interplay 2512.07783, behaviour priming 2503.01307v2.
2. Clean generators with reward controls (true, random and wrong reward; 2507.10532v3, 2506.10947v2).
3. A pre-registered sequence-level coverage threshold, plus a matched power-sampling null.
4. Equal-k pass@k plus base perplexity (Yue).

**Where our designs fall short:**

| shortfall | literature | cheapest fix |
|---|---|---|
| *[reviewed: V1]* Base read only at k 256 (B is ✗ r0 / ✓ r8, both at k 256: equal-k, as in Yue's Table 2); no large-k base curve on B ∪ C; groups from one sample seed | Yue Table 2; 2510.08325v2 | re-read end-of-pretraining on B ∪ C at k ≥ 4,096; plot equal-k curves |
| `rl-from-ckpt`'s creation threshold was calibrated post hoc | 2510.15020v2 | pre-register "covered at k" as sequence log p ≥ −ln k (−5.5 nats at k 256) |
| Worst-step w1 is a token-level unit, but the ladder samples whole proofs | 2503.07453v2 (token-level coverage suffices only with multi-turn exploration) | report sequence log p beside w1; w1 is right for the state env's step sampler |
| One reference proof's log p is only a lower bound on the mass on correct proofs | 2510.15020v2 | large-k solve rate on C, not the reference's −12 nats, decides "outside support" |
| No causal intervention on coverage | 2607.07646v1, 2510.11495v2 | remove one rule pattern from pretraining and test whether RL creates it |
| Short, narrow RL (8 × 32) | BroRL; 2510.04028v1 | one wide arm on C (k 256 per round) before calling C "unreachable" |
| No sharpening null | 2601.21590v1 | scalable power sampling of the end checkpoint at matched tokens |

Lean makes our pass@k equal to CoT-pass@k (2506.14245v2), so the lucky-guess critiques do not apply to us.

## Q2. Exploration outside support

| method | out-of-support evidence | strength | pretrained prior? |
|---|---|---|---|
| AlphaZero / MuZero (root Dirichlet noise, self-play) | superhuman from "random play"; concepts "not present in human games" taught to 4 grandmasters (2310.16410v1 Table 4) | strong for the system; noise never ablated (1712.01815v1) | no |
| AlphaProof PUCT + progressive sampling | policy samples only; variant curriculum (proposal 20) | IMO results, no support measure | yes |
| RMaxTS | +1.2 points at equal samples (earlier note) | weak | yes |
| Go-Explore | Montezuma "completely" solved; return-then-explore against derailment (2004.12919v6) | strong (state coverage) | no |
| Count / RND / Agent57 / ICM | 15 Montezuma rooms (1606.01868v2); scores | moderate | no |
| DeepHOL-Zero heuristic injection | 7.0 % → 56.3 % from random init; one-off seeding not enough | strong for theorem proving | no |
| Entropy, pass@k and clip-higher objectives | "do not resolve" zero-reward problems (2601.18779v1) | negative | yes |
| Representation bonus (RepExp) | large-k pass@k kept; samples "less likely under the base"; weak models "no benefit or even degradation" (2510.11686v2) | moderate-weak | yes |
| CDE curiosity | ≈ +3 AIME points (2509.09675v1) | weak | yes |
| Self-play variants (SvS) | pass@k to 1,024 above the initial model; augments only problems at 12.5–50 % accuracy (2508.14029v4) | moderate | yes |
| Oracle prefixes (POPE) | unguided solving of hard problems; transfer back | moderate (abstract-level read) | yes |
| LLM evolution (AlphaEvolve) | beats SOTA on ~20 % of > 50 problems | strong, but search, not a policy | frontier LLM |

**Qualifications to proposal 20:**
- **E2.** No source applies ε-random actions to a sequence policy (UNVERIFIED, from absence after search). Go-Explore
  argues that random actions mixed in "throughout an episode" derail, so apply ε **after returning** to a stored partial
  proof; the state env makes return free. DeepHOL's goal-similar injection supports E2's subformula bias.
- **E3.** DeepHOL-Zero is the precedent and predicts our step-0 null: "With no new training data, the learning process
  would stall." Keep ε on throughout, and add a seeded-only control.
- **E1.** AlphaZero's noise works over an enumerated move set. With sampled tactics it can only reweight what was
  sampled (our reading), so E1 + E2 root noise over valid actions is the true analogue.
- **Correction to the Q2 agent's synthesis.** "Group C's single −12-nat step" is not supported. The `trajectory` review
  rejected "C stays at one bad step", and proposal 20 describes "several far-off steps".

## Q3. Model organisms

| domain | exact coverage control | RL phase published | what it could show |
|---|---|---|---|
| rewrite grammar (2607.07646v1; 16 problems per bucket, 3 seeds *[reviewed: V5]*) | yes (ρ) | yes, GRPO vs RFT | new action types; RFT ceiling |
| parity, short/long CoT (2510.11495v2) | yes (p_cot) | yes | amplification closed form p_n = 2p_{n−1}/(1+p_{n−1}); **p_cot = 0 untested** |
| graph path-finding (2509.22613v2, 2601.15158v4) | yes | yes, theory | PG without KL = SFT on own successes (our ladder); collapse to "one path per pair" |
| Physics of LMs (2305.13673v4, 2407.20311v1) | yes | not in Parts 1 / 2.1 | RL to unseen operation depth |
| Othello-GPT (2210.13382v5) | yes | no | does RL change the probed world model? |
| grokking (2301.05217v3; 2602.14872v3) | yes | theory predicts RL plateaus | circuit formation under a verifier |
| chess, Transcendence (2406.11741v4) | rating cap | no; gains from low temperature | RL past the denoising ceiling ("ChessFormer 1500 is unable to transcend") |
| program synthesis (DreamCoder) | partial | wake-sleep | library growth |
| FOL / ND (Minimo; DeepHOL; ours) | yes | yes | per-step log p of any proof at every checkpoint (ours only) |

Ours is the only organism with likelihoods of arbitrary proofs along the whole trajectory. Parity and graphs are
minutes-cheap calibration controls. The rewrite grammar is the one to contrast with directly.

## Implications for our project (ranked by value)

1. **Equal-budget support re-read plus a pre-registered coverage threshold.** Re-read B ∪ C at the end of pretraining
   (best-cap12) at k ≥ 4,096, with "covered" = sequence log p ≥ −ln k, alongside a scalable power-sampling null.
   This settles "elicitation" for `trajectory` and `rl-from-ckpt` to the field's standard. It is cheap: reads only.
2. **Reference-prefix starts for group C** (POPE / backward algorithm): start at the state before the far-off steps,
   move the start back over rounds, and evaluate unguided. *[reviewed: V2]* It is the route the literature backs for
   zero-reward targets in pretrained LLMs (2601.18779v1), and proposal 20 lacks it. In from-scratch organisms, on-policy
   RL alone has also reached base-pass@1024 = 0 problems (2607.07646v1, 2509.22613v2).
3. **Proposal 20 E2 / E3 revisions:** ε after return, ε on throughout, a seeded-only control (DeepHOL-Zero, Go-Explore).
4. **A coverage-removal intervention:** delete one rule pattern from pretraining, at p = 0 and at small p (the
   Tsilivis dial), then RL. Creation at p = 0 would be new to the literature.
5. **GRPO beside the ladder.** RFT plateaus in 2607.07646v1, and the ladder is RFT. "Mirage or Method?" predicts
   negative-only gains will not reach C, where base pass@k is low.
6. **Amplification null and diversity tracking:** the per-round share against Tsilivis's recursion, and per-theorem
   proof diversity (Wang).
7. **EDL shape test** from existing per-checkpoint log p. Low priority: its theory is SFT-only.

## Pre-registered expectations vs outcomes (`d024bd0e`)

| # | expected | outcome |
|---|---|---|
| 1 | no LLM study shows RL beyond base at k ≥ 1,024 with a likelihood check | **held**: none found in this screen (51 papers, 34 read at abstract level) *[reviewed]*. 2607.07646v1 has the k-1,024 half from scratch (16 problems per bucket, 3 seeds), with no likelihoods |
| 2 | ProRL rests on format-weak tasks | **held, partly**: conceded for Reasoning Gym; other tasks not checked |
| 3 | ≥ 2 new controlled-pretraining + RL papers | **held**: 2607.07646, 2510.11495, 2509.22613, 2601.15158 |
| 4 | out-of-support evidence only from tabula-rasa search and Go-Explore / count methods | **partly falsified**: heuristic injection, oracle prefixes and LLM evolution also qualify; *[reviewed: V2]* more directly, plain on-policy GRPO / PG on from-scratch organisms (2607.07646v1, 2509.22613v2) reaches behaviour absent from pretraining. The "no bonus measures support at k ≥ 1,024" half held |
| 5 | ≤ 4 brief families have coverage control + RL; ours at least as controlled | **held** (grammar, graph, parity, FOL) |
| 6 | ≥ 90 % of re-checked claims verify | **held**: 33 / 33 (reviewer: 106 / 106 checkable ledger claims) |
| 7 | ≥ 1 of our own claims mis-stated | *[reviewed: V4]* **held**: the pre-registered wording is "mis-stated or needs a qualification", and there are three qualifications (E1 noise, E2 placement, the per-step bound in `lit-review` §c item 1, which is sequence-level in coverage terms; the reviewer calls the last arguably a mis-statement). Tally: 6 held, 1 (#4) partly falsified |

## Screened papers

Full rows with each paper's technique or design: [`screened.md`](screened.md).

| # | paper | id | rel. | depth | Q |
|---|---|---|---|---|---|
| 1 | Does RL Really Incentivize Reasoning Capacity Beyond | 2504.13837v5 | 3 | full | Q1 |
| 2 | ProRL | 2505.24864v1 | 3 | full | Q1 |
| 3 | Excess Description Length of Learning Generalizable Predictors | 2601.04728v1 | 3 | full | Q1 |
| 4 | The Coverage Principle | 2510.15020v2 | 3 | full | Q1 |
| 5 | Is a Good Foundation Necessary for Efficient | 2503.07453v2 | 3 | partial | Q1 |
| 6 | RLVR Implicitly Incentivizes Correct Reasoning in Base | 2506.14245v2 | 2 | abstract | Q1 |
| 7 | Self-Improvement in LMs | 2412.01951v2 | 2 | abstract | Q1 |
| 8 | Cognitive Behaviors that Enable Self-Improving Reasoners | 2503.01307v2 | 2 | abstract | Q1 |
| 9 | Beyond Pass@k | 2510.08325v2 | 2 | abstract | Q1 |
| 10 | Spurious Rewards | 2506.10947v2 | 2 | abstract | Q1 |
| 11 | Reasoning or Memorization? Data Contamination | 2507.10532v3 | 2 | abstract | Q1 |
| 12 | BroRL | 2510.01180v1 | 2 | abstract | Q1 |
| 13 | The Debate on RLVR Reasoning Capability Boundary | 2510.04028v1 | 2 | abstract | Q1 |
| 14 | Curriculum RL Can Incentivize Reasoning Beyond the | 2606.22317v1 | 1 | abstract | Q1 |
| 15 | First return, then explore | 2004.12919v6 | 3 | full | Q2 |
| 16 | Bridging the Human-AI Knowledge Gap | 2310.16410v1 | 3 | full | Q2 |
| 17 | Representation-Based Exploration for LMs | 2510.11686v2 | 3 | full | Q2 |
| 18 | Learning to Reason in Large Theories without | 1905.10501v3 | 3 | full | Q2 |
| 19 | POPE | 2601.18779v1 | 3 | partial | Q2 |
| 20 | Beyond Pass@1 | 2508.14029v4 | 2 | partial | Q2 |
| 21 | Mastering Chess and Shogi by Self-Play | 1712.01815v1 | 2 | partial | Q2 |
| 22 | MuZero | 1911.08265v2 | 1 | partial | Q2 |
| 23 | Acquisition of Chess Knowledge in AlphaZero | 2111.09259v3 | 2 | partial | Q2 |
| 24 | Intelligent Go-Explore | 2405.15143v4 | 1 | partial | Q2 |
| 25 | Unifying Count-Based Exploration and Intrinsic Motivation | 1606.01868v2 | 2 | partial | Q2 |
| 26 | Exploration by Random Network Distillation | 1810.12894v1 | 1 | abstract | Q2 |
| 27 | Agent57 | 2003.13350v1 | 1 | abstract | Q2 |
| 28 | Curiosity-driven Exploration by Self-supervised Prediction | 1705.05363v1 | 1 | abstract | Q2 |
| 29 | CDE | 2509.09675v1 | 1 | partial | Q2 |
| 30 | AlphaEvolve | 2506.13131v1 | 2 | partial | Q2 |
| 31 | TacticZero | 2102.09756v2 | 2 | partial | Q2 |
| 32 | RL Post-Training Builds Compositional Reasoning Strategies | 2607.07646v1 | 3 | full | Q3 |
| 33 | How RL After Next-Token Prediction Facilitates Learning | 2510.11495v2 | 3 | full | Q3 |
| 34 | Benefits and Pitfalls of RL for LM | 2509.22613v2 | 3 | full | Q3 |
| 35 | Outcome-Based RL Provably Leads Transformers to Reason, | 2601.15158v4 | 2 | abstract | Q3 |
| 36 | Atomic Skills are the Prerequisite | 2512.01970v3 | 2 | partial | Q3 |
| 37 | On the Emergence of Implicit Curriculum in | 2602.14872v3 | 2 | abstract | Q3 |
| 38 | Transcendence | 2406.11741v4 | 2 | abstract | Q3 |
| 39 | SFT Memorizes, RL Generalizes | 2501.17161v2 | 1 | abstract | Q3 |
| 40 | Physics of LMs Part 2.1 | 2407.20311v1 | 2 | abstract | Q3 |
| 41 | Physics of LMs Part 1 | 2305.13673v4 | 1 | abstract | Q3 |
| 42 | Emergent World Representations | 2210.13382v5 | 1 | abstract | Q3 |
| 43 | Progress Measures for Grokking via Mech. Interp. | 2301.05217v3 | 1 | abstract | Q3 |
| 44 | The Pitfalls of Next-Token Prediction | 2403.06963v3 | 2 | abstract | Q3 |
| 45 | ALPINE | 2405.09220v3 | 1 | abstract | Q3 |
| 46 | DreamCoder | 2006.08381v1 | 1 | abstract | Q3 |
| 47 | Grokking | 2201.02177v1 | 1 | abstract | Q3 |
| 48 | Scalable Power Sampling | 2601.21590v1 | 2 | abstract | Q1 |
| 49 | Power Distribution Bridges Sampling, Self-Reward RL, and | 2605.04542v1 | 2 | abstract | Q1 |
| 50 | Rethinking Sample Polarity in RLVR | 2512.21625v1 | 2 | abstract | Q1 |
| 51 | Mirage or Method? Model-Task Alignment | 2508.21188v2 | 3 | abstract | Q1 |
