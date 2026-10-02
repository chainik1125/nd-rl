---
written_on: 2026-10-02
written_by: agent:claude
status: drafted for Dan as an unqueued follow-up (2026-10-02); to run after `grpo-best`. Arm E1 moved to proposal 22 (MCTS)
---

# Proposal 20: exploration — can RL reach steps its policy almost never proposes?

## Why

Dan's point (2026-10-02): unless RL explicitly explores, it cannot reinforce an action the model never takes,
however good it would be. Within one context that is true by construction for expert iteration, which trains
only on sampled successes. Two mechanisms can still move probability onto unsampled actions:
- **Transfer:** a move sampled on one theorem becomes likely on another (Yuan et al. 2025, RL composing skills).
- **Negative gradients:** GRPO pushes down failures, and the mass flows to unsampled alternatives (Zhu et al.
  2025). `grpo-best` tests this.

What our runs show so far (`trajectory`, `trajectory-cap6`, `rl-from-ckpt`): RL completes theorems whose hardest
step is nearly reachable. The theorems with several far-off steps (reference worst step ≈ −12 nats, group C)
never move. This proposal asks whether **exploration** changes that.

## What AlphaProof does

Source: Hubert et al., Nature, published 12 Nov 2025, Methods "Standard tree search framework" and "Key adaptations
for AlphaProof". Checked on 2026-10-02 against the open-access text.
- **There is no root noise.** The text never mentions Dirichlet or exploration noise. Every action is sampled
  from the proof network's own policy.
- **PUCT selection** with a temperature-modified prior, π^{1/τ}(a | s), and an exploration factor that grows with
  visits: c(s) = c_init + log((N(s) + c_base + 1) / c_base).
- **Expansion:** "K tactics are sampled from the proof network's policy π(⋅∣s_L). Each tactic is validated in the
  Lean environment; invalid tactics are discarded." Tactics leading to the same Lean state are merged.
- **Progressive sampling:** if n(s) ≤ C·N(s)^α, "an additional K tactics are sampled from π(⋅∣s) and added as new
  edges … to explore a wider range of proof strategies along critical paths." AND nodes send search to the
  hardest unproven subgoal.
- **Training:** (state, tactic) pairs from successful proofs and disproofs (90 % replay, 10 % Mathlib SFT), with a
  cross-entropy policy loss. That is search-as-expert iteration on found proofs.
- **Curriculum:** test-time RL generates "simplifications or generalizations" of a hard target. Variants make an
  unreachable problem reachable step by step.

So AlphaProof also cannot reinforce what its policy never proposes. It explores **on-policy**: many more samples
where the search says they matter, plus problem variants that move the target closer. Its policy is pretrained on
about 300B tokens, so that support is broad.

## Design

**Starting point:** `trajectory`'s three cap-12 end-of-pretraining checkpoints (Robbie's recipe, proof-state
environment). The **EI arm is reused**: `trajectory`'s ladders, as in `grpo-best`. The headline target set is
**group C**, the theorems those ladders never solved.

| arm | exploration | what it tests |
|---|---|---|
| **E1. On-policy widening (AlphaProof-style)** | PUCT search expert with progressive sampling over proof states (temperature on the prior, K samples per expansion, widening at heavily visited nodes). Train on the shortest found proof per theorem. | `search-expert` done properly. Its best-first search used only about ¼ of its budget; this spends it on hard states. |
| **E2. Off-policy ε (two levels, ε = 0.05 and 0.2)** | With probability ε per step, replace the policy's action with a **random valid action**: a rule applied to existing lines. For the formula-introducing rules (assumptions, ∨-introduction), formulas are drawn from subformulas of the goal and premises, a standard ND heuristic (state that bias). EI trains on found proofs as usual. | Dan's structural point directly: does injecting steps the policy would not propose let RL reach group C? |
| **E3. Learning from scratch** | Start from step 0 (no pretraining), with E2's ε and an easy curriculum (generator theorems of 1–6 lines first, then the targets). | The AlphaZero question: can RL plus exploration learn natural deduction with no pretraining? `rl-from-ckpt`'s step-0 arm (no exploration) gets nothing. |

**Rules for every arm:**
- Compute is matched to the EI arm on generated actions, with GPU-seconds recorded.
- Rollouts *with* exploration produce training data. **Evaluation is always plain sampling from the trained
  policy,** with no search and no ε. The question is whether the policy itself has internalised what exploration
  found.

## Read-outs and analysis

- **Group C:** solved per arm and seed at k 256, by plain sampling at r8. This is the headline.
- **Internalisation:** for theorems first solved through exploration, the policy's own pass@k and the
  teacher-forced log-likelihood of the found proof, before and after training.
- **pass@k by `trajectory` group, and retention:** group A, held-out greedy.
- **E3:** the learning curve from no pretraining (proofs found per round by length), against `rl-from-ckpt`'s
  step-0 and step-1,600 arms.

## Cost

- **Code:** one executor session for the PUCT search with progressive sampling (on top of `search-expert`'s
  best-first code) and the valid-action sampler for ε.
- **Pods:** E1 3 ladders, E2 6, E3 3, about 12 × $4 ≈ $50, plus read-outs ≈ $10. **Total ≈ $60.**
- **Order:** after `grpo-best`, so the gradient-side answer (negative gradients, unlikeliness) is known first.
