---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - schut2023alphazeroconcepts
---

# AlphaZero concept discovery and transfer to grandmasters

Paper: [@schut2023alphazeroconcepts]
Source: https://arxiv.org/abs/2310.16410v1 (v1 reviewed; html text)

## Learnings
- Aim: find knowledge in AlphaZero's representational space (M) that is absent from the human space (H), the "(M−H)" gap (Introduction).
- Method: concept vectors are found in AlphaZero's latent space and filtered for **teachability** ("whether it is transferable to another AI agent") and **novelty** ("whether it contains some information that is not present in human games") (§4).
- Novelty is operationalised by comparing latent spans. They stack latents of "17,184 chess positions sampled from AZ's games" against the same number from human games, then run a rank experiment and regress the concepts onto each span (§4, paragraph "Setup to measure novelty").
- Human study (§6, Table 4): four top grandmasters. Phase 1 solves concept puzzles. Phase 2 shows the same puzzles "alongside the associated AZ's suggested top line based on MCTS calculations". Phase 3 tests "unseen puzzles sampled from the same concepts". The authors write that "all study participants improve notably between phases 1 and 3" (§6.1). Scores go from Phase 1 to Phase 3 as 0→42, 33→58, 25→42 and 38→44 (improvements +42, +25, +16, +6). Each player saw 36–48 puzzles.

## Evidence and limitations
- This is the strongest available evidence that **tabula-rasa self-play RL with search** produced knowledge outside the human distribution. Novelty is measured in representation space against human games, which is a support-style measurement rather than a win-rate.
- It is still weak as causal evidence. There are n = 4 players and roughly 36–48 puzzles each. There is no control group (for example, grandmasters shown non-novel AZ puzzles, or no Phase 2), so practice effects are not excluded. "Novel" is defined relative to a sample of human games in latent space, not relative to all of chess knowledge.
- Concepts were selected to "emerge in the final stages of training". The paper does not isolate which component (Dirichlet noise, MCTS, self-play opponent curriculum) produced them.
- The companion paper McGrath et al. (2111.09259v3) finds the opposite direction too: "many human concepts can be predicted from AlphaZero's activations after training". Rediscovery and novelty coexist. AlphaZero (1712.01815v1) also reports that common human openings were each "independently discovered and played frequently by AlphaZero during self-play training".

## Connections and questions
- **E3 (from scratch).** AlphaZero is the existence proof that RL plus search plus exploration can, with no prior, reach and then exceed the human (here: generator) distribution. The ingredients it had and we lack are: a dense, always-defined outcome signal (every self-play game ends in win, draw or loss, whereas our unsolved theorems give zero signal); an opponent that grows with the agent (an automatic curriculum); and an enumerable legal-move set over which root Dirichlet noise places mass on *every* legal move. Our step-0 `rl-from-ckpt` arm had none of these.
- **Our interpretation.** Dirichlet noise can widen support only because chess and Go legal moves are enumerated and all are scored by the policy head. In AlphaProof, tactics are *sampled*, so root noise could only reweight already-sampled candidates. That may be why AlphaProof does not use it (proposal 20). E2's "random valid action" is the closer analogue to AlphaZero's noise for an enumerable ND action space.
- **Measurement idea for our project.** Schut's span/rank test transfers directly: compare the latent span of policy states on RL-found proofs with the span on generator proofs. That is a representation-level "out of support" measure to complement the −12-nat likelihood criterion for group C.
- Earlier notes: `minimo-intrinsic-conjecturing.md` (self-play curriculum from scratch in formal maths), `stp-self-play-conjecturer.md`.
