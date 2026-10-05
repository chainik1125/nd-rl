---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/capability-vs-propensity.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: capability = the best elicitation within a budget; propensity = default behaviour

Family E (elicitation cost). Slug `capability-vs-propensity`. Notation: `_FRAME.md`.

## 1. Definition, formally

- **Propensity.** prop_θ(t) = p_θ(t) under the default protocol: one plain attempt at T 0.8. It is what the model
  *would* do.
- **Capability at budget C.** cap_θ(t; C) = max over **per-theorem** elicitation methods m with cost(m) ≤ C of
  P(m elicits a valid proof of t from θ). It is what the model *can* do. *Revised after the critic (§9):* a method is
  admitted only if it uses nothing learned from Lean's verdicts or proofs of other theorems:
  - plain sampling at any T;
  - guided sampling (step-checked redraws, `guided_sample.py`);
  - prior-only best-first / PUCT search with the model as the policy (no learned value head);
  - other renamings of the same theorem.

  Fine-tuning and value heads trained on other theorems are excluded: they are RL's own mechanism (`elicit-finetune`
  treats them).
- **Cost** is measured in GPU-seconds, or in attempt-equivalents of the base's plain sampling.
- **Budget C** is tied to RL's compute as in `passk-budget`: the headline is K_eval-set (C_per and C_total beside it).

## 2. Decision rule

For base B and RL model R, per theorem:

| verdict | rule |
|---|---|
| **created** | cap_R(t; C_eval) high, and cap_B(t; C) < 0.05 for every method in M at the budget C: no allowed method gets the base there |
| **elicited** | cap_B(t; C) ≥ 1 − 1/e for some method within C, while prop_B(t) is low. RL converted a capability into a propensity: "can" was already true, RL changed "would" |
| **neither** | R fails at C_eval |

This is the exact sense of the dangerous-capability literature: "capability evaluations are about whether an AI system
*can* do some task … alignment evaluations are about whether it *would*" (van der Weij et al.).

## 3. Null or floor

- The budget is the floor. Random weights have cap(t; C) ≈ 0 for every allowed method at any C we can afford, because
  every method samples from the model's own distribution.
- The method set must not supply the proof. Guided sampling only checks steps; it never proposes them. The
  "reasonable prompt" condition of Greenblatt et al.: "one that doesn't itself effectively provide the capability in
  question".

## 4. How to compute it here

- **Plain sampling at large budgets:** J2.
- **Guided sampling:** J3 (pend / r8 / r16, cap 12; cap 6 on pod C), ≈ 2 A40-hours per seed.
- **PUCT search** at pend and r8 exists from `mcts-a`: it closed ≈ 44 % of the 8-round EI gain on rrQ100 at matched wall
  clock, but nothing on group C.
- **Analysis:** the method-maximum per theorem at matched cost (guided costs ≈ 1.8× the tokens and 2.2× the GPU time
  of plain per attempt; use `guided-tts`'s matched-token accounting). As implemented (`cd_part3.py`, `guided`): the
  compute-matched set (sampling at K_eval-set, uncertified) minus theorems pend's single guided read solves at k 256.
- **Results** (`cd_j3.py`, `out/j3.txt`; k 256, T 0.8; plain / guided solved):
  - cap-12 pend, holdout250: 196 / 218, 205 / 223, 200 / 222 (s0 / s1 / s2); r8: 240 / 239, 236 / 237, 237 / 241. The
    r8 − pend gap shrinks from 44 / 31 / 37 (plain) to 21 / 14 / 19 (guided). Cap-6 pend gains +30 / +39 / +47.
  - pend's guided read solves 50 / 55 / 60 % of the equal-k created set (cap 6: 40 / 38 / 44 %): much of RL's plain gain
    is propensity.
  - Within the compute-matched set (22 / 21 / 14) it rescues only 3 / 4 / 1, leaving **19 / 17 / 13** (net of the
    replay-only control 17 / 12 / 8): theorems that 2 × 10⁴ plain attempts cannot reach are not reached by step
    checking either.

## 5. Sensitivity

- **The method set M.** Adding a stronger method can only move theorems from created to elicited. The verdict is
  relative to M and must be stated with it.
- **Budget C:** as in `passk-budget`.
- **Temperature:** part of M; take the best T.
- **Representation:** the proof-state interface is itself a powerful elicitation method for whole-proof models
  (`support-state`, *[review]* 3.2 M models, not best-cap12: 28 of 29 survivors). Whether "interface" is in M is a design decision.
- **Renaming:** a prompt-level method in M. Allow all renamings of t, but count each as an attempt.
- **Noise:** per method, binomial at the read's n. Maximising over methods inflates (a winner's curse), so confirm the
  winning method on a fresh draw.

## 6. Failure modes

- **Open-ended M.** "No method we tried elicits it" is not "no method can". Absence of capability is never certified,
  only bounded (Greenblatt et al. Sec. 2: "it is difficult to show that a model does not possess a particular
  capability").
- **Methods that teach.** A search with a strong learned value function, or a prompt with worked examples, carries
  information in; its cost must count, or it must be excluded.
- **Maximum over methods vs noise:** see §5.

## 7. Relations

- `passk-budget` is the special case M = {plain sampling}.
- `elicit-finetune` extends M to weight changes.
- `reliability` is the propensity side.
- The pair (cap, prop) is what the report recommends showing for every theorem set.

## 8. Literature anchor

- **van der Weij et al. 2024 (2406.07358v4).**
  - "actual capability to be the best performance on a certain task it can achieve, given the best currently available
    capability elicitation techniques such as prompting and fine-tuning" (Sec. 2).
  - The 1 % of training cost convention, quoted from Anthropic (2023) (Sec. 2).
  - "Capability evaluations are about whether an AI system can do some task … whereas alignment evaluations are about
    whether an AI system would do some task" (Sec. 2).
- **Shevlane et al. 2023 (2305.15324v2):** capability vs propensity (Abstract; Sec. 1). "Researchers will need to bring
  latent capabilities to the surface" (Table 2).
- **Hofstätter et al. 2025 (2502.02180v3):** "A latent capability … is one that a model exhibits with low probability
  by default, but can be revealed through small changes to the context, activations, or weights" (Sec. 3.3).
- **Greenblatt et al. 2024 (2405.19550v1):** the reasonable-prompt definition (Sec. 2).
- **Firestone 2020 (PNAS 117(43)):** performance constraints mask competence; a fair comparison accommodates them.
- All verified in `_claims_L2.md`.

## 9. Critic's verdict

**Strongest argument (critic): nothing in the card separated its elicitation methods from RL.**
- The PUCT + value search of `mcts-a` trains a value head on Lean-labelled rollouts of pend on all 4,495 `rl_targets`
  plus 1,500 K12 theorems. That is two-thirds of an EI round of data (≈ 1,490 A40-s per head, not counted in the
  "matched" budget).
- Of the +14.3 rrQ100 theorems per seed that search recovers, the value head supplies +5.3 over prior-only search, and
  the card counted these as pend's capability. The same data trained into the policy is EI round 1, which the card
  counts as RL.
- Secondary arguments:
  - "Created" needs every method bounded below 0.05 within C, which needs about 60 C of attempts; beyond plain
    sampling the card reduces to `passk-budget`.
  - Its own anchors (van der Weij, Hofstätter, Greenblatt) count small fine-tunes as elicitation, yet the card
    excluded them.
  - It had no "undetermined" row (J2 already has theorems with cap_B in [0.05, 1 − 1/e)).

**My answer: accepted. The method set M is now defined by a stated criterion.**
- **Admitted: per-theorem methods only**, i.e. nothing trained on Lean's verdicts or proofs of *other* theorems.
  Sampling at any T, renamings of t, step-checked redraws (guided sampling), and prior-only search, with all compute
  charged to t.
- **Excluded:** value heads or fine-tunes trained on other theorems' verdicts. Carrying Lean's verdicts across theorems
  into weights is what defines RL here, and it is studied separately by `elicit-finetune`.
- **Verdicts:**
  - "created" = RL's learning on other theorems reached t, and no per-theorem method did within the per-theorem
    budget;
  - "elicited" = some per-theorem method reaches t within the budget;
  - an **undetermined** row is added.
- This criterion is the run's cleanest statement of the distinction: **elicitation is what can be done to one theorem
  with the base alone; RL is what is learned across theorems from the verifier.**

