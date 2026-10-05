---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/latent-probe-steer.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: latent capability — probes, steering and model diffing — not computed; DROPPED as a decision rule, kept as diagnostics (critic, §9)

Family M (mechanistic). Slug `latent-probe-steer`. Notation: `_FRAME.md`.

## 1. Definition, formally

Three operational tests of "the base already represents / implements it".

- **(P) Probe with control** (Hewitt & Liang).
  - Train a linear probe on the base's activations at the decision state to predict the key decision (e.g. "this goal
    needs reductio / DN", or which of two candidate steps is valid).
  - Selectivity = probe accuracy on the real label minus accuracy on a control task (random labels per type).
  - Compare the base's selectivity with init's (random-network null) and with R's.
- **(S) Steering, 2 × 2** (Ward et al.; Venhoff et al.).
  - Compute a direction from activation differences, then add it to the other model's residual stream at the decision
    state.
  - The four cells (direction from base / from R; applied to base / to R) give three outcomes:
    - **elicited:** a base-derived direction steers the base into R's behaviour;
    - **repurposed:** a base direction works only in R;
    - **created:** only an R-derived direction works, and only in R.
  - Venhoff's share of the base-to-RL gap recovered by steering the base is a continuous score.
- **(D) Model diffing** (crosscoders; Lindsey et al., Minder et al.). Features whose base decoder norm is ≈ 0 are
  candidate RL-new features, after the Latent Scaling check (refit the feature on base activations).

## 2. Decision rule

| verdict | rule |
|---|---|
| **elicited** | (P) base selectivity ≥ R's minus a margin and far above init's; or (S) base-derived steering recovers ≥ 50 % of the gap |
| **created** | (P) the base's selectivity ≈ init's while R's is high; or (S) only R-derived directions work in R; or (D) RL-only features survive the Latent Scaling check |
| **neither** | no behavioural gap |

## 3. Null or floor

- Init activations are the probe null (the random network): Hewitt & Liang's random ELMo matched layer 2's accuracy
  (96.3 vs 96.6) but not its selectivity (20.6 vs 31.4).
- For steering: a random direction of equal norm. Ward: noise alone raised the behaviour.
- For diffing: a pend-vs-second-pend crosscoder.

## 4. How to compute it here

- **Cheap.** A 6 × 384 network; activations for a few thousand states take seconds on a GPU.
  - Probe the LEM decision (does the state need DN?) at pend, r16 (s1) and init: ≈ 0.2 GPU-hours.
  - The 2 × 2 steering on holdout250 A ∨ ¬A states: ≈ 0.5 GPU-hours.
  - A crosscoder: ≈ 2–4 GPU-hours.
- **Not run in this run** (scope and time); proposed as a follow-up with an exact protocol (REPORT § open questions).

## 5. Sensitivity

- **Layer and position:** choose by validation, not by test results.
- **The control task** matters as much as the probe.
- **Steering magnitude:** a large direction can force any output. Use the matched-norm random null, and count the bits
  the steering leaks (Venhoff: the tuned model chooses when and which vector, up to log₂ K bits per step).
- **Representation:** activations depend on the interface.
- **Seed:** probes per seed.

## 6. Failure modes

- **Decodable ≠ used:** a probe can read information the model never uses (Hewitt & Liang; Belinkov).
- **Steering can teach:** a vector carries information. Without counting its bits, "steering recovers it" can be
  disguised teaching.
- **Crosscoder artefacts:** most "chat-only" features from the original loss are artefacts (Minder Sec. 3.1).
- **Small models may not have linear, separable representations of "needs reductio".**

## 7. Relations

- The internal counterpart of `capability-vs-propensity` (steering is an elicitation method in M) and of
  `elicit-finetune` (Jain et al.'s revival test).
- A positive probe without behaviour is Firestone's competence without performance.

## 8. Literature anchor

- **Hewitt & Liang 2019 (1909.03368):** selectivity with control tasks (Sec. 4.2, Table 2).
- **Jain et al. 2023 (2311.12786):** wrappers vs capabilities; revival in 0.1–3K iterations vs 4.5K for a never-had-it
  control (Fig. 9, Table 1).
- **Prakash et al. 2024 (2402.14811):** fine-tuning enhances existing mechanisms (Sec. 6.2).
- **Ward et al. 2025 (2507.12638):** the 2 × 2 steering design (Sec. 3.2, Fig. 3).
- **Venhoff et al. 2025 (2510.07364v4):** ≈ 76 % gap recovery for RL vs 11 % for SFT (version caveat: v1 claimed "up to
  91%" for all models).
- **Lindsey et al. 2024** (transformer-circuits.pub), **Minder et al. 2025 (2504.02922).**
- All L6 notes; verified in `_claims_L6.md`.

## 9. Critic's verdict

**Strongest argument (critic): the steering test alone can say "elicited", and it measures *where* pend's surprisal
sits, not *how much* there is.**
- `la_transfer_795` (an A ∨ ¬A): 24.7 of its reference proof's 29.3 nats sit in the first two actions, both opening a
  ¬-box. Forcing them with a pend-derived direction leaves ≈ 4.6 nats, so the verdict is "elicited".
- Yet pend solves it 0 / 1,024, r8 0 / 3,584, the replay-only control 0 / 512, and only 1 of 3 seeds ever learned it.
- `la_transfer_2208` costs about the same but is spread over five steps, so it gets the opposite verdict, although RL
  learned it on all 3 seeds.
- Secondary arguments:
  - (D) and (S) can give opposite verdicts on the same pair, with no rule for which wins.
  - The placebo update is as large as RL's.
  - Probing the seed that learned the skill is selection by outcome.
  - The thresholds were never set.

**My answer: accepted; dropped as a decision rule.** Probes and steering stay as diagnostics of where RL's change sits.
The only repair that survives (charge each steered step −log π_pend against ln K) reduces to `tf-proof-prob`'s
teacher-forced test, which needs no activations.
