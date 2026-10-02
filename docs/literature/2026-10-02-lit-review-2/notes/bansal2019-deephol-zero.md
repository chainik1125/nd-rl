---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - bansal2019deepholzero
---

# DeepHOL-Zero: theorem-proving RL without human proofs, with heuristic action injection

Paper: [@bansal2019deepholzero]
Source: https://arxiv.org/abs/1905.10501v3 (v3 reviewed; html text)

## Learnings
- Goal: automated theorem proving in HOL Light (the HOList benchmark) "without learning from human proofs", using "an exploration mechanism that mixes in additional premises selected by a tf-idf ... based lookup" (Abstract).
- Mechanism (text around Fig. 3b): the learnt model picks the top-k₁ premises, and a second list of k₂ premises is "picked according to a metric" (tf-idf cosine similarity to the goal). "The final set of premises is obtained by interleaving the two premise lists." The Reference setup has k₂ = 0. Proofs are **pruned** of unnecessary premises before training, so over-approximated actions do not pollute the training data.
- Failure mode without exploration: "With no new training data, the learning process would stall. Thus, it is crucial that we continuously expand the horizon of theorems that we are able to prove" (§4.1).
- Seed data come from "a randomly initialized model (i.e. the 0-th checkpoint of a model)" (Appendix).
- Main result (Fig. 4, % of validation theorems proved, final / cumulative): **Zero reference 7.0% / 7.3%**; **Zero explore 56.3% / 64.2%**; Human reference 59.5% / 68.2%; Human explore 59.9% / 69.1%; pure human imitation (Paliwal et al.) 49.95%.
- Ablations (§6, Fig. 5): "Zero Seeded" (a one-off 20% of proofs found with the hand metric plus random tactics) "does not stall like the reference loop" but "does not reach the same level of performance as the Zero Explore, which explores throughout". Hand metric alone (k₁ = 0): "43% of the statements cumulatively, compared with 64%" for the mixed loop.
- Cost: "over 25 years of CPU resources" per full RL loop (§4.1).

## Evidence and limitations
- **Strong evidence for from-scratch learning with exploration in theorem proving.** The same loop with no exploration stalls at 7%. With heuristic off-policy action injection it reaches the human-data level. Exploring throughout beats one-off seeding, and the learned plus heuristic mixture beats the heuristic alone, so the policy learned beyond the injector.
- The "out of support" here is relative to a random-init policy, not a pretrained one. It shows exploration supplying signal the policy could not, not new knowledge beyond humans (the zero loop roughly matches the human-data loop).
- The injection is targeted (goal-similar premises), not uniform random. Pruning is essential to make over-approximated actions harmless.
- The tactic set is fixed and small; only premise arguments are explored.

## Connections and questions
- **E3 (direct precedent).** Our step-0 `rl-from-ckpt` arm learning nothing reproduces "Zero reference" stalling. DeepHOL's fix is E2-like injection kept on *throughout* training. Our interpretation: E3 should keep ε on for the whole run (not anneal it early), and should report a "seeded only" control.
- **E2 (formula choice).** DeepHOL's tf-idf goal-similarity is the analogue of proposal 20's "subformulas of the goal and premises" heuristic. This is published support for heuristic-biased rather than uniform random actions. Add the pruning step: strip unused `have` lines from ε-found proofs before EI trains on them, so that random detours are not reinforced.
- **Mixing ratio.** k₁ : k₂ interleaving corresponds to a per-step ε. The injector proposes candidates at *every* step, so it acts as an argument-level proposal mixture rather than an occasional override. I did not verify the k₂ values used.
- **Group C.** The mechanism helps only if the missing step is in the injector's distribution. For group C we should check whether the −12-nat reference step is a goal/premise-subformula action.
- Earlier notes: `exit-anthony2017.md`, `minimo-intrinsic-conjecturing.md`, `kim2026-process-verified-rl-lean.md`; earlier screened rlCoP (1805.07563), another from-scratch prover.
