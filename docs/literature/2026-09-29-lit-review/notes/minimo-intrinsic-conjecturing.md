---
written_on: 2026-09-29
written_by: agent:claude
papers: [poesia2024learningformalmathematicsintrinsic]
---
# Minimo: difficulty-conditioned conjecturing plus hindsight, on propositional logic, with an 8.45M model
Paper: [@poesia2024learningformalmathematicsintrinsic]
Source: https://arxiv.org/abs/2407.00695v2

## Learnings
- One LM encodes a proof-search policy, a value function, and "a difficulty-conditioned conjecturer P_θ(c | d)" (§3). Conjectures are sampled by constrained decoding + type-directed synthesis so they are well-formed "by construction, even as we start with a randomly initialized model" (Abstract, §3.1); rejection rate "<10%" (§3.1).
- Difficulty = log-likelihood of the found proof under the current policy, shown to track MCTS iterations (App. B, Fig. 5). Discretisation (§3.4): "the 10% least likely proofs … 'hard' … the 50% most likely … 'trivial', and the remaining … 'easy'", relative to the last batch — so the target moves with the agent.
- Hindsight (§3.3): paths after forward actions in failed search trees are relabelled as proofs of what they derived; two data-quality steps: "(1) we clean up the solutions by eliminating steps irrelevant for the proof of the new goal, and (2) we only add proofs of goals never seen before". Beyond our version: hindsight goals also become training examples for the conjecturer (with their difficulty label).
- Ablation (§4.1, Fig. 3; 3 seeds): without hindsight "training tends to collapse to by proposing only easy conjectures"; only "around 10-20% of the conjectures are proven" per initial batch.
- Scale (§4; App. A): 5 iterations, 200 conjectures per batch, MCTS 1000 expansions; "GPT-2-style character-level Transformer models totalling approximately 8.45M parameters".
- Length growth (§5): Propositional Logic "average length from 2.75 to 4.21 steps, longest proofs from 5 to 11 steps".
- Extrinsic (§4.2, Fig. 4): success on 35 propositional theorems from Kleene's textbook rises over training; four (iff commutativity/transitivity, a double-negation law, currying) "are only proved after the last iteration". Numbers only in the figure.

## Evidence and limitations
- Tiny budgets and 5 iterations; no comparison against a fixed-curriculum or generator baseline, so Minimo does not show that conjecturing beats a good synthetic generator. Proofs found are short (≤11 steps) — below our L*=12.
- Search is MCTS with a value head; we sample without search.

## Connections and questions
- Closest published analogue of our setup (small from-scratch model, propositional logic, hindsight). Our generator already supplies valid statements, so constrained decoding adds little.
- What is new for us: (a) difficulty measured as proof log-likelihood under the current policy — free with our sampler, and a continuous frontier signal for rungs where pass rate is 0 at feasible k; (b) dedup of hindsight goals ("never seen before"), relevant if our hindsight data is dominated by trivial repeats; (c) a difficulty-conditioned prefix token ("hard"/"easy") on theorem statements — a cheap "conjecturer head" analogue if we let the model also emit statements.
- B4: Kleene-textbook improvement from self-generated data is weak evidence that self-posed curricula help textbook theorems, but at short lengths.
