# Run brief: capability-defs — what should "capability" mean for this project, and how do we measure it?

Run id `capability-defs`. Repository `~/work/capability-defs` (branch `dan_capability-defs` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-04 (proposal 26).

- **Budget:** pods **$50 / 100 pod-hours**, optional. Use them only where existing artefacts cannot answer a question.
  Claude time is expected to be very heavy, and that is the point of the run.
- **Thinking:** every role of this run runs at `--effort max` (set by the driver).
- **Time:** **take your time.** Dan: "the agents should really take their time, use max thinking mode, and just
  consider every possible way to define capabilities for the project." The hard stop is 72 h, but there is no
  deadline before it. Prefer depth and care to speed.

## Why

The project's question is whether RL against a verifier **creates** a capability or **elicits** one the pretrained
model already had. We have never pinned down what "capability" means precisely enough to answer it. In Dan's words
(2026-10-04): "frankly, I'm a little confused here! So I think that I'm happy with the agents really thinking about many
different ways to define 'capabilities', as long as there is some way to make them quantifiable."

The working notions so far:
- **Teacher-forced probability of specific proofs** under the pretrained model and the RL model.
- **pass@k at large k**, pretrained vs post-trained.
- The group's stated goal (meeting of 2026-09-20): "Quantify threshold of RL that creates novel capabilities, vs
  elicitation or pass@k", and "How to clearly quantify the difference between base model capabilities and
  fine-tuned model capabilities."
- Dan's own objection (SPAR doc): "There is some k at which even randomly initialized weights will solve any proof in
  our dataset. Is there a more principled way to say that a capability has emerged than just picking a k and
  sticking with pass@k?" He also suggested measuring the log-likelihood of reference proofs directly.

## Read first (extend; don't redo)

- **Literature:** nd-rl `docs/literature/2026-09-29-lit-review/` (58 papers) and
  `docs/literature/2026-10-02-lit-review-2/` (51 papers, create vs elicit, exploration, model organisms).
  `docs/literature/README.md` and `TEMPLATE.md` for conventions.
- **Our measurements so far** (nd-rl `experiment-summaries/` on `dan`):
  - `support-curves` and its follow-ups `support-followups` and `support-state`: 29 theorems the base never solves in
    400k attempts that its EI descendant solves;
  - `lit-measures`: one bad step;
  - `trajectory` and `trajectory-cap6`: eventual proofs, worst-step log p across pretraining and RL;
  - `rl-from-ckpt`, `rl-continue`, `rl-continue-cap6`: group C, the excluded-middle schema on one seed;
  - `organism-analysis`, `claim-audit` (and `docs/project_strategy/2026-10-03-claim-audit.md`), `evidence-atlas`;
  - `noise-floor` and `NOISE_FLOOR.md`;
  - `guided-tts`, `mcts-a`, `grpo-best`, `compute-match`.
- **Context:** `docs/project_strategy/2026-10-01-week-in-review.md`, `docs/project_strategy/big_picture/`, the group
  meeting notes in nd-rl `main:docs/research_calls/group_meeting/`, and the model cards in `docs/models/`
  (`state_env_october`, `state_env_october_r16`).
- **Leon's excess description length (EDL) work** on `leon/benchmarking` and `leon/benchmarking-dan`. These are other
  people's branches: **read only.**

## Part 1: literature, extensive

How do ML and RL, and the fields they borrow from, define and measure a "capability", "ability", "skill" or
"competence"? Above all, how do they tell a new capability apart from better access to an old one? Older
foundational work is welcome here. Threads to follow, as a starting point; go wherever they lead:

- **Sampling-based:**
  - pass@k and coverage (Chen et al. 2021's estimator, Large Language Monkeys, pass@k power laws and tail
    extrapolation);
  - "reasoning boundary" and support arguments (Yue et al. 2025, ProRL, Invisible Leash, "RL squeezes, SFT expands");
  - sharpening (Huang et al.).
- **Likelihood-based:** teacher-forced log-likelihood and perplexity of reference solutions; marginal vs
  single-solution probability.
- **Elicitation and evaluation practice:**
  - password-locked models and the stress-testing of capability elicitation;
  - sandbagging;
  - fine-tuning-based upper bounds in dangerous-capability evaluations;
  - capability vs propensity.
- **Information and compression:** MDL probing, prequential coding, V-information, excess description length.
- **Learnability:** sample complexity to learn or elicit, few-shot and few-step fine-tuning, PAC framings.
- **Psychometrics:** item response theory, latent ability, test equating; observational scaling laws (capabilities as
  low-dimensional latents); emergent abilities and metric choice ("mirage").
- **RL theory:**
  - KL-regularised RL as reweighting the base distribution;
  - coverage and concentrability;
  - exploration and reachability;
  - when RL provably cannot leave the base support.
- **Composition and generalisation:** composing known skills into new ones, length generalisation, invariance to
  renaming.
- **Representational and mechanistic:**
  - probes for latent knowledge;
  - model diffing and crosscoders;
  - activation steering from the RL model into the base model;
  - how sparse or low-rank RL updates are (if a tiny update suffices, is that elicitation?).
- **Games:** AlphaZero concept discovery and teaching concepts back to humans; "novel" moves.
- **Philosophy of capabilities:** dispositional definitions, Hernández-Orallo's *The Measure of All Minds*.

**Targets:** screen ≥ 60 papers new to the two earlier reviews, and read ≥ 20 in depth. **Verify every cited claim
against the source text,** as in `lit-review`: fetch it into a file, grep it, and record the location. Mark anything
unverified.

## Part 2: a catalogue of definitions

Write **at least 15** candidate definitions, more if they earn their place. Each gets a card
(`capability_defs/cards/<slug>.md`) with these fields:

1. **Definition, formally:** the quantity, computed from which model(s), which theorem(s) and which protocol.
2. **Decision rule:** what result means "RL created a capability", what means "RL elicited one", and what means
   neither.
3. **Null or floor:** how it answers the random-weights objection. Options include a null model (random init, a
   uniform-over-the-grammar policy), a budget tied to RL's own sampling budget, or a reference model.
4. **How to compute it here:** which checkpoints and artefacts already exist, and the cost in CPU / GPU hours and $.
5. **Sensitivity:**
   - temperature and decoding (plain vs guided);
   - k and thresholds;
   - the proof representation (whole proof vs proof state);
   - renaming, premise order and curried forms;
   - seed and redraw noise (`NOISE_FLOOR.md`).
6. **Failure modes:** how it can mislead or be gamed.
7. **Relations:** which other cards it implies or is implied by.
8. **Literature anchor.**
9. **Critic's verdict:** spawn a critic subagent whose only job is to argue the definition fails. Record its strongest
   argument and your answer.

Dan's two notions must be among them, **made strict**:
- **(a) Teacher-forced probability of specific proofs.** Specify:
  - which proof: the reference proof, the shortest known proof, the RL model's own proofs, or the marginal over every
    valid proof (the probability of solving is a sum over proofs);
  - the normalisation: total, per step, worst step;
  - how base and RL compare;
  - the threshold;
  - how it relates to the probability that sampling finds a proof.
- **(b) pass@k at large k.** Specify:
  - the unbiased estimator;
  - extrapolation beyond the sampled k;
  - k-to-solve (≈ 1/p) and compute-to-solve;
  - a budget tied to something principled;
  - a null that answers the random-weights objection.

A seed list beyond those two. Add to it, merge entries, discard any that don't hold up:
1. bits of skill over a null model, log p_θ(solve) − log p_null(solve);
2. ε-support, and splitting RL's gain into **sharpening** (mass moved within the base support) and **expansion**
   (mass on proofs the base essentially never produces);
3. "new proof" vs "new theorem": the base's log p of the RL model's proofs;
4. step-level novelty and **schema acquisition**, measured over a schema family (e.g. the excluded-middle burst on
   `rl-continue` s1);
5. elicitable by search or guided sampling within compute B;
6. elicitable by small fine-tuning (sample complexity; EDL, prequential MDL);
7. transfer, length generalisation and invariance (capability vs memorised instance);
8. reliability (pass@1 thresholds, consistency across seeds);
9. **IRT ability:** fit a Rasch or 2PL model across every checkpoint × theorem result we already have, then put all
   models and items on one scale;
10. probes and activation steering (latent capability);
11. KL(π_RL ‖ π_base) and the size of the update;
12. "RL worth X× pretraining compute", or the base-sampling compute equivalent to the RL gain;
13. composition of skills that each exist in the base;
14. capability as the maximum over elicitation methods within a budget, vs propensity as default behaviour.

## Part 3: quantify on our models

Choose **at least 6** definitions spanning different families. Compute them on the same comparisons:
- **models:** cap-12 pretraining-end vs r8 vs r16 (`trajectory` seeds, `rl-continue`); the cap-6 equivalents; and the
  `rl-from-ckpt` ladders where useful;
- **sets:** `rl_targets`, holdout250, group C, the long pools, and textbook72. textbook72 is **scoring only,** under
  its manifest rule.

For each definition, report:
- the set of theorems on which "RL created a capability";
- how big it is, with its noise floor (seed and redraw spread).

Then report:
- an **agreement matrix** across definitions (overlap / Jaccard of those sets);
- threshold sensitivity;
- the cases where definitions disagree most, with examples in Lean.

**Pods:** only when existing artefacts cannot answer. Examples:
- teacher-forced log-probs of the RL proofs under the base checkpoints;
- large-k samples from a null model;
- a few elicitation fine-tunes.

Pre-register each pod job in `log.md` before launching it, and record compute rows per AGENT_POLICY.

## Part 4: recommendation

- **Recommend 2–4 definitions as the project standard,** each with an exact protocol (k, T, decoding, seeds,
  thresholds, null). For each, give one paragraph on what result would count as "RL created a capability", and say
  what our existing results already show under it.
- **A plain-language glossary** for Dan: each term in one or two sentences.
- **The open questions Dan has to decide,** each with your recommendation.

## Process

- **Subagents** (the Agent tool) are encouraged:
  - for parallel literature reading: **at most 3 at a time,** downloads one at a time and never in bulk (the VPS has
    3 GB RAM and a 4-process limit; `lit-measures` crashed itself with 160 parallel downloads);
  - for the critic pass on every card;
  - for an independent second derivation of any quantity that matters.
- **Checkpoint in `STATUS.md` after each part** (`PART n DONE <UTC>`), so a resumed session knows where it is.
- **Write for Dan.** He says he is confused. The report must open with a plain answer of ≤ 150 words, and define
  every symbol where it is first used.

## Deliverables

- `capability_defs/REPORT.md` (≤ 5,000 words plus tables), in this order:
  1. the plain answer;
  2. a summary table of the catalogue;
  3. the quantification (with figures);
  4. the recommendation;
  5. the glossary;
  6. the open questions.
- `capability_defs/cards/*.md`.
- `capability_defs/lit/REVIEW.md`, plus `lit/notes/*.md` (TEMPLATE format), `lit/screened.md` and
  `lit/to_add_to_zotero.md`.
- `capability_defs/analysis/`: scripts and figures.
- `numbers.md` § capability-defs, and compute rows for any pod work.
- `run_capability_defs.md` (≤ 250 words), `log.md`, `STATUS.md` line `CAPABILITY-DEFS DONE <UTC>`, and
  `touch ~/runs/capability-defs/executor.done`.
- **The librarian** copies:
  - the literature to nd-rl `docs/literature/2026-10-04-capabilities/` and links it from
    `docs/literature/README.md`;
  - the report and cards to nd-rl `docs/project_strategy/capabilities/`;

  and writes the experiment summary as usual.

**Rules:**
- Everything is a draft for Dan. Do not contact anyone.
- Other people's branches are read-only.
- Do not edit `references.bib` or run `code/tools/literature.py`.
- Never train or tune on textbook72.
- Every number comes from a file.
- At most 4 concurrent processes on the VPS.
- Never run `test_run_once.sh`.
