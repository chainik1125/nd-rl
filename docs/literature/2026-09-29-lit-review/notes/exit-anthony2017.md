---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Expert Iteration (ExIt): tree search as the expert, network as the apprentice
Paper: Thinking Fast and Slow with Deep Learning and Tree Search, Anthony, Tian, Barber, 2017
Source: https://arxiv.org/abs/1705.08439v4
## Learnings
- ExIt loop (§3): self-play states from the apprentice; the expert (MCTS guided by the apprentice) supplies an imitation target at each state; retrain; repeat. "removing the expert improvement step from online ExIt reduces it to DAgger" (§3).
- Target choice (§4.2–4.3): chosen-action targets (CAT, the move MCTS picked) vs tree-policy targets (TPT, visit distribution n(s,a)/n(s)). Similar top-1 errors (47.0% vs 47.7%) but "the TPT network is 50 ± 13 Elo stronger than the CAT network" (§4.3).
- Search as expert beats policy gradient for the *trained network* (§6.1, Fig. 2): "compared to REINFORCE, ExIt learns stronger policies faster"; x-axis is "number of neural network evaluations" (compute-matched), 5 runs, 90% CIs; no value network in this comparison "so that network architectures between the policy gradient and ExIt are identical". Online (aggregated data) "substantially outperforms the batch mode".
- Value (§5.2, §6.2): value trained on apprentice Monte Carlo rollouts, because expert-value targets need ">10^5 independent samples"; multitask policy+value heads with summed losses. "value-and-policy-ExIt significantly outperforms policy-only-ExIt" and "the improved plans from the better expert quickly manifest in a stronger apprentice" (Fig. 3).
- Policy prior in MCTS (§5.1): N-MCTS with policy network wins 97% vs vanilla MCTS; doubling vanilla iterations only 56%.
## Evidence and limitations
- Domain is 9×9 Hex (two-player, dense terminal outcomes, fixed small action set). Elo numbers read from text; Fig. 2/3 curves not digitised.
- Baseline is REINFORCE, not sampling-based expert iteration (rejection-sampling fine-tuning). The paper does not test "sample N rollouts, keep winners, imitate" vs "search, imitate" — the comparison we care about.
- Value network needed ~550k positions before it was trained (§6.2).
## Connections and questions
- B2 (support expansion): the claimed mechanism is that search finds plans the apprentice would not sample, then generalisation transfers them. Our EI with whole-proof sampling is the degenerate expert (no lookahead).
- TPT targets have no clean analogue for proof search (success is binary and we keep only proofs); the analogue is HTPS's "minimal proof" filter.
- Smallest test: in the state env, replace independent step-by-step sampling by a small best-first/MCTS search at equal step-generation budget; train on found proofs; evaluate the policy by plain sampling (no search) so we measure the apprentice, as ExIt does.
