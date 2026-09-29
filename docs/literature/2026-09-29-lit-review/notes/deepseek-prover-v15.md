---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# DeepSeek-Prover-V1.5: RLPAF (GRPO), truncate-and-resume, RMaxTS
Paper: DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search, Xin et al., 2024
Source: https://arxiv.org/abs/2408.08152v1
## Learnings
- RLPAF (§2.3): GRPO on whole-proof generation, reward 1 if Lean verifies else 0; prompts are ~4.5k theorems where the SFT model has "a moderate success rate"; group 32, KL 0.02, lr 5e-6. Pass@128 miniF2F-test: SFT (CoT) 50.4% ± 0.4 → RL (CoT) 51.6% ± 0.5 (Fig. 3). At 16×6400 single-pass: SFT 57.4% vs RL 60.2% (Table 3).
- Truncate-and-resume (§3.1, Fig. 4): generate a whole proof, check in Lean, truncate at the first error, split the valid prefix into tactic nodes, resume generation from a selected node with the tactic state as a comment. Expansion = one whole-proof rollout, which may add a path of nodes.
- RMaxTS (§3.3): intrinsic reward = 1 if the expansion added at least one new node (new tactic state); extrinsic reward only for complete proofs. Selection uses discounted UCB (γ = 0.99) because the intrinsic reward is non-stationary. No value network (§5: critic named as future work).
- Results (Table 3, RL model, CoT): single-pass 58.4% ± 0.5 (4×6400) / 60.2% (16×6400) vs RMaxTS 59.6% ± 0.6 / 62.7%. Fig. 5 ablation: UCT without intrinsic reward 58.2 / 61.1; RMaxTS with UCB1 58.6 / 60.7; "in the absence of intrinsic rewards, the performance of UCT ... degenerates into a level comparable to that of non-search methods" (§4.3).
- RL and search gains are "orthogonal" (§4.2).
## Evidence and limitations
- 7B model pretrained on DeepSeekMath-Base. Search gains are ~1–2.5 points at equal generation budget, with σ ≈ 0.3–0.8. RMaxTS is test-time only: the training data for RL is whole-proof sampling, so the paper does not test search as the EI expert.
- Tactic-state comments matter: "RMaxTS (without tactic state)" reaches 58.4% ± 0.3 / 61.1% vs 59.6 / 62.7 with them; "the performance gain from applying tree search becomes moderate in the absence of tactic state information" (§4.3, Fig. 5). Our state arm already feeds the state.
- Our reading (not the paper's claim): the count bonus pays off when many distinct texts reach the same state, as in Lean with Mathlib automation.
## Connections and questions
- Truncate-and-resume maps directly onto our whole-proof arm: Lean's first error in a `lean_seq` proof gives a valid prefix; resume from prefixes rather than resampling from scratch. This is "search for free" on top of the existing sampler.
- RMaxTS needs no value head, so it is the cheapest MCTS for a 3.2M model. But our step env has a small, low-entropy action space (syntactic ND steps), so "new state" may saturate fast; the gain here could be smaller still.
- B2: novelty-driven expansion targets exactly the rare-state reachability that pass@400k measures. Smallest test: RMaxTS over the state env at equal step budget vs independent sampling, on the 29 hardest theorems.
