---
written_on: 2026-09-28
written_by: agent:claude
status: proposed; run `state-env` authorised by Dan 2026-09-28 ($30)
---

# Proposal 13: give the policy the proof state (an AlphaProof-style environment)

## The gap

Every model in this project writes a whole proof in one pass, conditioned only on the theorem
statement and its own partial proof: π(y_t | prompt, y_<t) (`grpo.py`, `ladder_ei.py`,
`expert_iter.py`). Nothing tells it which hypotheses are in scope, which boxes are open, or what
the current goal is; it must reconstruct all of that from the text it has written. The verifier
runs once, on the finished proof. For a 4-layer, 3.2 M-parameter model, tracking scope and goal
across a 12-line proof with nested boxes is a plausible cause of the wall the project keeps hitting
(required length ≈ 9–10 on the reductio pool; `L_true` ≥ 13 solved 0 times in 11 ladder runs;
T1 `L*` = 12 at every cap from 6 to 14). The one relevant data point points the same way:
`ds-rendering`'s R1 removed the premise re-statement lines — explicit, citable copies of the
hypotheses — and was worse on every measure.

## What AlphaProof gives its policy

From Hubert et al., "Olympiad-level formal mathematical reasoning with reinforcement learning",
*Nature* (2025), open access (Europe PMC PMC12999475), Methods, "Lean environment" and "Prover agent":

- **State:** "the complete logical context within the Lean prover at a given step t of a proof
  attempt, encompassing all active hypotheses and remaining goals." **Observation:** "a
  pretty-printed string representation of the Lean tactic state" — each hypothesis as
  `name : type`, then `⊢ goal`.
- **Markov:** "The encoder takes the pretty-printed Lean tactic state as input." No tactic history
  and no separate copy of the theorem; the theorem is the initial state.
- **One goal at a time:** when a tactic splits a goal into N > 1 subgoals, "the environment splits
  the current Lean tactic state into N distinct child states. In each of these child states, all
  but one goal are assigned with 'internalSorry' to leave a single open goal."
- **Action:** "a Lean tactic, represented as a text string." Invalid tactics are discarded; tactics
  reaching the same state are merged.
- **Reward and value:** r_t = −1 per tactic; the value head predicts the number of steps left in the
  longest open branch. Search is AlphaZero-style over tactic states, with AND nodes for subgoals.
- **Learning:** the learner trains on (state, tactic) pairs, "10 % from the Mathlib SFT dataset and
  90 % from a replay buffer filled with proofs and disproofs collected from the actors"; "failed
  attempts are filtered out". The policy update is therefore close to our expert iteration with a
  retained supervised mix — the difference that matters here is **what the policy sees**, not the
  update rule.

## The change

A step-level environment for the `lean_seq` fragment, AlphaProof's observation, our update rule.

- **Observation:** the Lean-style tactic state of the focused goal — every hypothesis in scope as
  `name : formula` (premises `h_i`, earlier `have`s at this or an enclosing level, box binders), then
  `⊢ goal`. The top-level goal is the theorem's conclusion; opening a box sets the goal the box must
  prove (`A → B` binds `n : A` and sets `⊢ B`; `¬A` binds `n : A`, `⊢ False`; `Or.elim` gives two
  subgoals, one at a time; `Classical.byContradiction` binds `h : ¬A`, `⊢ False`); `exact n` closes the
  focused goal and returns to the parent, where the finished `have` is now a hypothesis.
- **Action:** one `lean_seq` step — a `have` line, a box opening, or `exact`. The actions concatenate
  back into an ordinary `lean_seq` proof, so the **final proof is checked by Lean exactly as today**
  (`lean_judge.py`); the environment is only a different way of producing it.
- **State without a type checker.** In this fragment every `have` declares its formula, so scope,
  hypotheses and goal follow from the syntax of the actions alone; a wrong term shows up as a Lean
  rejection of the finished proof, as it does now. Syntactic errors (an out-of-scope name, `exact n`
  whose declared type is not the goal) end the attempt early. A per-step symbolic type check is an
  optional filter, allowed only if measured to agree with Lean.
- **Everything else fixed:** the 3.2 M model, the control's 155,000 proofs (converted to
  (state, action) pairs), the 6,000-step schedule, plain EI on the ladder (rung T1) with the frozen
  control at equal attempts, Lean as the judge. The only change is the policy's input.

## Arms

- **S — state only (AlphaProof-faithful).** Input: the current tactic state. Output: the next action.
- **SH — state plus history.** Input: the proof so far and the current state. Separates "the state
  helps" from "dropping the history helps". Run if the budget allows; dropped first.
- **C0 — the current whole-proof model**, not re-run: the on-file control checkpoints
  (`stage1_a1_seq_s{0,1}`) and their T1 / frozen ladder results.

## What would count

The ladder is where the wall is, so the headline quantities are there: transfer `L*`, theorems solved
at `L_true` ≥ 13 (0 in all 11 whole-proof ladder runs of `ds-generator`, T1 and frozen), and frozen and
T1 transfer solves. Against `NOISE_FLOOR.md`: the frozen ladder's run-to-run range is 62–265, so only
a large move in solves counts; any robust count at `L_true` ≥ 13 is new. Held-out accuracy is reported
but at two seeds is not resolvable on the 6-line bin (±82 pp) or the depth-3 slice.

- **The state is the wall** if arm S reaches T1 `L*` ≥ 13 with ≥ 5 solved at `L_true` ≥ 13 on both
  seeds.
- **The state is not the wall (at this scale)** if arm S stays at T1 `L*` ≤ 12 with 0 solved at
  `L_true` ≥ 13 on both seeds. That would point at capacity or search (proposal 12, experiment 2).

## Not in scope

Tree search and a value head (AlphaProof's other half) — the natural next step if the state helps,
since the step-level environment is what search needs. GRPO in this environment. Larger models.

## Budget

$30, ceiling 60 pod-hours (Dan, 2026-09-28). Most of the engineering and all validation is CPU work.
