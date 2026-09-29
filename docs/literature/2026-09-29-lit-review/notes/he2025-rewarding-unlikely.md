---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Unlikeliness reward: GRPO rank bias and pass@N in Lean theorem proving
Paper: Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening, Andre He, Daniel Fried, Sean Welleck (CMU), 2025
Source: https://arxiv.org/abs/2506.02355v2 (v2, 20 Jun 2025, reviewed; v1 intro text identical on the points below)

## Learnings
- Setting: base model DeepSeek-Prover-V1.5-SFT, GRPO in verl, binary Lean reward `R(x,y)=1{y proves x}` (§2). 32 samples per problem, LR 1e-6, KL 0.02, one epoch, 512-token responses (§3.2). Data: 9.6K train / 200 held-out val from a 10K "solvable" Lean Workbook subset + miniF2F-valid (§3.1). The paper text does not state the parameter count (the base model is 7B per DeepSeek-Prover-V1.5; not checked here).
- Diagnosis: "GRPO substantially boosts pass@1 to pass@16, but the improvement diminishes for larger N" (§3.3, Fig. 2, evaluated up to pass@512). Rank bias: "low-probability positive samples ... are almost never uplifted" (§3.5, Fig. 4).
- Toy argument (§3.4, Fig. 3): if RL multiplies correct-solution probability by (1+ε), the gain in pass@N is concentrated on solutions with p0 ≈ 1/N.
- Method (§4.1, Eq.): `r_i = R(x,y_i)(1 − β_rank (G − rank(y_i))/G)`, rank by sequence probability under π_old within the group, β_rank = 0.25; failures stay at 0; samples with zero advantage *before* the perturbation are still skipped. Second mitigation: more PPO epochs per batch (§4.2).
- Variants, Table 1: GRPO-Default (K=1, β_KL 0.02), Unlikeliness-1 (K=1, β_KL 0.10, β_rank 0.25), Unlikeliness-2 (K=2, 0.10, 0.25), Epochs-2/3.
- Results: Fig. 5 "substantial improvements in pass@N at large N, with a minor tradeoff in pass@1 and pass@2" (numbers only in the figure). Table 2 (training problems solved during one epoch, 32 samples each): static 7707/9600, GRPO-Default 7860 (+153), Epochs-2 8008 (+301), Epochs-3 8006, Unlikeliness-1 8023 (+316), Unlikeliness-2 8065 (+358). Table 3: miniF2F-test pass@32 / pass@128: SFT 47.1 / 49.2, V1.5-RL 49.2 / 51.2, ours 48.8 / 50.6; D_val: SFT 78.3 / 83.1, RL 84.8 / 87.5, ours 84.3 / 88.8.
- Fig. 7: unique proofs per step — Unlikeliness-2 diversity "initially drops but later recovers"; others decline monotonically (§5.3).
- App. D: raising KL alone "prevented the deterioration of pass@N" but "did not bring a substantial improvement over the base model".

## Evidence and limitations
- The largest N is 512 (App. C); no evidence at 10^4–10^5 attempts. The large-N gains exist only as plots (Fig. 5); no table.
- The intro claims the recipe is "substantially outperforming standard expert iteration" (§1), but I found no EI experiment in the v1 or v2 body. Treat this as unsupported.
- Single seed per variant as far as the text shows. The train set was pre-filtered to solvable problems, so it says nothing about problems with zero base success.

## Connections and questions
- B2. Drop-in for our GRPO: we already have per-sample sequence log-probs, so ranking within the group costs nothing.
- Like any group-relative reward, it gives no signal on groups with zero successes. On theorems the base never reaches, any effect can only come through transfer.
- EI analogue (our idea, untested): weight hindsight/EI SFT targets toward correct proofs with low base probability (or the min-probability distinct proof per theorem). Compare with the Polu shortest-proof rule we already proposed.
- Smallest test: GRPO vs GRPO+unlikeliness (β_rank 0.25) from the same Stage-1 checkpoint, ≥2 seeds. Read the support curve at 400k on the base-unreached set, plus distinct-proof counts per theorem.
