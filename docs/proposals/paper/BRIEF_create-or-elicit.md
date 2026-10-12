# Run brief: create-or-elicit — the experiments for a paper on whether RL created it

Run id `create-or-elicit`. Repository `~/work/create-or-elicit` (branch `dan_create-or-elicit` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-11 (proposal 27).

- **Budget:** **$140 / 300 pod-hours.** The writing run that follows gets the remaining $10 of Dan's $150. More may
  follow, depending on what this run finds.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 120 h; no hurry. Dan: Claude quota is plentiful, and agents "should be given wide agency to
  propose extensions or literature reviews (etc etc) that they think might help us learn more."

## The paper this run serves

**Working title:** "Did RL create it? Budgets, controls and a pretraining knockout in a from-scratch prover with exact
likelihoods."

What we already have, all reviewed (`capability-defs` and its reading copy `docs/project_strategy/capabilities/REPORT.md`):
- **The budget shrinks the "creations."** At equal k (256), RL "creates" 54 / 51 / 60 theorems per seed. At a
  compute-matched budget that drops to 22 / 21 / 14, and 4 / 5 / 5 are never solved by any seed's base. 4 (theorem,
  seed) pairs on 3 theorems survive every RL-free control. A 1.2–1.5 M-attempt certification confirmed 7 of the 9 pairs
  it tested.
- **The definitions disagree:** mean Jaccard 0.29 / 0.34 across 12 of them.
- **The known-proof estimator is accurate where it can be checked:** it recovers 0.93–0.98 of measured p, on
  calibration theorems only.
- **Excluded middle is a discovery that is cheap to teach:** one seed in three found it, and 4 demonstrations teach
  every base.

What is missing is a **causal** result. Without it, the paper is a careful measurement note rather than a finding.

**Precision points to carry into every write-up:**
- **Interface:** "a proof-state base reaches 28 of the 29 earlier creations" is from the older 3.2 M organism, with 2
  seeds per arm. The claim audit found it confounds the proof state with step-by-step generation.
- **Estimator:** validated only where p can be measured directly, ≈ 3–100 successes in 512.
- **Budget:** as computed, `K_eval-set` divides RTX A6000 ladder seconds by an A40 attempt cost. In ladder attempts it
  is ≈ 5.6–6.2 × 10³ per theorem, and at 0.3 × the budget the compute-matched set grows to 33 / 28 / 28.

## Read first

- `capability-defs`: the report, the cards (especially `causal-ablation`, `schema-acquisition`, `elicit-finetune`,
  `passk-budget`, `marginal-bracket`) and the review.
- `rl-continue` and `rl-continue-cap6`; `trajectory`; `support-curves`, `support-state`, `claim-audit`.
- Proposals 20 (exploration) and 24 (guided sampling).
- The literature:
  - nd-rl `docs/literature/2026-10-02-lit-review-2/`, especially its table of designs and gaps;
  - `docs/literature/2026-10-04-capabilities/`;
  - the notes on Abdulsalam, Patel & Saxe (2607.07646), Tsilivis et al. (2510.11495) and Interplay (2512.07783).

## Phase 0: cheap fixes before any new training (≤ $5)

1. **The estimator at small p.** `capability-defs`' J1 / J2 artefacts hold hard theorems with a handful of base
   successes in 16k–65k attempts. On those, compare the known-proof sum with the measured p.
   - Add a **Good-Turing (or similar) estimate of the unseen proof mass**, from how often a new sample's proof is one
     never seen before.
   - Say where the estimator can and cannot be trusted.
2. **The budget in base attempts, with the GPU class measured.** Measure the base's per-attempt cost on the ladders' GPU
   class (or restate everything in ladder attempts), and restate the compute-matched sets.
3. **Pre-register the thresholds** for the whole run (e.g. log p ≥ −ln K for "within reach").
4. **A claims ledger** (`claims_ledger.md`). List every number the paper might use, with its source file and its review
   status. Re-audit the claims inherited from `capability-defs` and `support-*` with a critic subagent.

## Phase 1: the causal experiment (≈ $90–100, the core of the run)

**Skill removed:** σ = the composition "double negation applied to a negation-introduction line." In
`capability-defs`' count, 16,703 of K12's 155,000 records contain it. DN and ¬I each stay in the data; only their
composition goes. Every classical-only key-step family we track needs it (excluded middle, both Peirce families, the
classical members of the negated-conditional and De Morgan families). Confirm that, and say which it does not cover.

**The knockout set K12⁻:**
- Remove every record containing σ in written **or** pruned form.
- Refill to 155,000 with fresh records from K12's own generator settings. They must pass every K12 filter: flat over
  lengths 2–12, box depth ≤ 2, disjoint by renaming class from every evaluation pool and from K12's own classes, and
  free of σ.
- Report what else the removal shifts (rule mix, lengths, premises). Also report the knockout base's general ability
  (held-out greedy, holdout250) against the full base's.

**Models:** `state_env_october`'s recipe exactly: 9.56 M ALiBiGPT, `lean_staten`, 1,200 s of pretraining, then T1 expert
iteration with k 32, T 0.8, 600 fine-tune steps and 20,000 replay records, for **16 rounds** as in `rl-continue`.
- **Replay comes from each arm's own pretraining set.** The knockout replays K12⁻, so σ can only enter through RL's own
  accepted proofs.
- Run every new ladder on one GPU class and record it.

**Arms and seeds:**
- **knockout (0 % σ):** ≥ 6 seeds;
- **full exposure:** the existing `trajectory` + `rl-continue` seeds s0–s2, plus ≥ 3 new seeds on the same protocol.

The new full-exposure seeds also give the headline counts more than three seeds. Discovery was a lottery (excluded
middle fully on 1 seed in 3, the Peirce families on another), so state the power of the arm comparison **before**
running, and choose seed counts with it in mind.

**Readouts, per arm and seed:**
- per round, the training share of each classical key-step family (`cd_schema.py --keystep`), and the round of first
  acquisition;
- held-out key-step members at k 256, plain and guided (`guided_eval.py --arm logical`), and at the compute-matched
  budget;
- the compute-matched reach on the 322 (textbook72 scoring only + holdout250), with k-to-solve intervals (standard A);
- certification for every theorem a headline rests on;
- for the knockout base: large-k sampling and certification on the classical families. Show it cannot reach them
  within the budget; do not assume it.

**Decision rule** (adapted from the `causal-ablation` card; pre-register your exact version):
- **Created:** RL from the σ-free base reaches a classical family at the bar (held-out rate ≥ 0.5) on ≥ 2 seeds, while
  the σ-free base is certified out of reach within the budget.
- **Needs exposure:** acquisitions occur only in the full arm. Also compare acquisition rates between arms with an exact
  test, and report the uncertainty honestly.
- **Neither:** no arm acquires a family. EI may never sample a σ step at p ≈ 0, so a negative result says "this RL
  cannot", not "RL cannot". Plan the exploration extension below for that case.

## Phase 2: a sharpening null (≈ $10–20)

Power-sample the pretrained base, sampling from p^α at the sequence level (Karan & Du 2025's MCMC, or the scalable
approximation in 2601.21590), at matched sampled tokens. Does it reach the compute-matched "created" set, and the
classical families? Run it on both arms' bases. Reaching them would be sharpening by definition.

## Phase 3: your choice with what remains

Pre-register each item before running it. Suggestions, roughly in order of information per dollar:
1. a **small-dose arm** (σ at ≈ 0.1 % or 1 % of pretraining, 3 seeds), the first points of a dose-response curve
   (Interplay; the Tsilivis dial);
2. **exploration on knockout bases**, if EI never finds σ: guided rollouts (proposal 24 Part B), ε after return, or
   reference-prefix starts (POPE; proposal 20);
3. a pilot of **interface** on this organism (whole proof vs proof state, same data and RL);
4. a pilot **harder target pool** built from our own generator (13–20 lines, classical-heavy, disjoint), to test the
   ~98 %-saturated-targets objection.

**Proposing more:**
- **More literature:** extend the reviews whenever it would sharpen the paper's positioning or design (verify claims
  against sources, as in earlier reviews).
- **Beyond this budget:** write costed proposals in `next_steps.md`, ranked by what they would teach us, for Dan to
  decide.

## Deliverables

- `run_create_or_elicit.md` (≤ 400 words + key figures), `numbers.md` § create-or-elicit, `log.md`, compute rows,
  `preregistration/create-or-elicit.md` (before the first pod), and `STATUS.md` line `CREATE-OR-ELICIT DONE <UTC>`.
- `claims_ledger.md`, the input to the writing run.
- Figures and the scripts that make them, paper-ready: vector or high-dpi, labelled, colour-blind-safe.
- `next_steps.md`.
- Bulk files to `hf://buckets/dan-pandori/nd-rl/create-or-elicit/`; pods deleted; `touch ~/runs/create-or-elicit/executor.done`.

**For the reviewer:** recount the knockout set's construction (σ absent; filters held), every arm's acquisition table,
the certification, and the claims ledger's sources.

**Rules:**
- Do not contact anyone. Everything is a draft for Dan.
- textbook72 is scoring only, under its manifest rule.
- At most 4 concurrent processes on the VPS; downloads one at a time.
- Never run `test_run_once.sh`.
- Keep the RunPod balance above $35.
