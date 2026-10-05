---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - shenfeld2025rlrazor
---

# RL's Razor: why online RL forgets less (KL to the base on the new task)

Paper: [@shenfeld2025rlrazor] (Shenfeld, Pari, Agrawal, 2025)
Source: arXiv 2509.04259v1 (HTML rendering read: abstract, Sec. 1, 3-7, App. A; App. B-C not read)

## Learnings

- **Claim.** "the degree of forgetting is determined by the distributional shift, measured as the
  KL-divergence between the fine-tuned and base policy evaluated on the new task"; "on-policy RL is
  implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can
  converge to distributions arbitrarily far from the base model"; RL's Razor: "among all ways to solve a new
  task, RL prefers those closest in KL to the original model" (Abstract).
- **Mechanism.** By sampling from itself, "RL constrains learning to outputs already given non-negligible
  probability by the base model" (Sec. 1). Binary-reward RL ("we used only a binary success indicator as the
  reward, without explicit KL regularization", Sec. 3.1).
- **Expert iteration is in the comparison.** "1–0 Reinforce" (A = 1 for correct, 0 otherwise): "This is
  equivalent to sampling from the model and performing SFT on correct answers only" (Sec. 5.1). "1–0
  Reinforce behaves similarly to GRPO, while SimPO resembles SFT", so "the critical factor is not the presence
  of negative gradients but the use of on-policy data" (Sec. 5.1, Fig. 4).
- **Theory.** Lemma 5.1 / A.1: rejection sampling from p with acceptance R(y) = 1 gives
  q_RS = argmin_q KL(q‖p) s.t. E_q[R] = 1 (the I-projection). Its proof writes, for any q supported on the
  success set S, KL(q‖p) = KL(q‖p(·|S)) − log p(S) (App. A, proof of Lemma A.1). Policy gradient = M-projection
  of the reward-reweighted policy (Lemma A.2); binary-reward RL = EM with information projection (Thm. A.3)
  and converges to argmin_{π∈P*∩Π} KL(π‖π0): "policy gradient selects, among all optimal representable
  policies, the one closest in KL-divergence to the starting policy" (Sec. 5.2, Thm. 5.2; Prop. A.4).
  Caveat: the network's policy set "induced by a neural network parametrization is not in general" e-flat,
  so exact convergence is not guaranteed (App. A, practical considerations).
- **Empirical forgetting law.** On ParityMNIST "A quadratic fit achieves R^2=0.96" for forgetting vs KL; in
  LLMs "with a quadratic fit achieving R^2=0.71" (Sec. 4). An oracle KL-minimal SFT distribution forgets
  even less: "SFT trained on the oracle distribution retained more prior knowledge than RL"; SFT on RL's own
  outputs matches RL: "The distilled SFT matched RL's accuracy–forgetting trade-off" (Sec. 4).
- **Alternatives fail.** Forward KL R² 0.96 ± 0.01 vs weight change L1 0.34 ± 0.02, Fisher-L2 and spectral
  0.58 (Table 1); among distribution distances "none approached the predictive power of forward KL" (Sec. 6).
- **Against update sparsity as a signal.** "the reason for the observed sparse updates was the use of
  bfloat16 for model training"; "Performing the same training with float32 resulted in models with
  identical performance but without any sparsity in their weight updates"; "we found that all algorithms
  lead to full rank weight updates" (Sec. 6).
- **Open.** "we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior
  knowledge" (Sec. 7).

## Evidence and limitations

- Evidence: Fig. 2-4, Table 1, Fig. 11 (LLM KL fit, not inspected); Qwen 2.5 3B-Instruct (math, science,
  tool use), OpenVLA 7B, ParityMNIST MLP. KL is estimated approximately (the authors attribute LLM residuals to
  "noise from approximate KL"). Appendices B-C (estimation details) not read.
- The theory is for a single prompt with finite outputs and binary reward; generalisation across prompts is
  what lets real RL leave the I-projection.

## Connections and questions

- **Definition offered:** none of capability; the *size of what fine-tuning changed* is the KL between the
  tuned and base policy on the new task's inputs, and RL's solutions are the KL-closest optimal policies.
- **New vs better access:** not discussed, but the I-projection algebra gives a sharp rule (our inference
  from Lemma A.1's proof). For theorem t with success set S (Lean-accepted proofs), any policy q that always
  succeeds satisfies KL(q‖π_base) = KL(q‖π_base(·|S)) + log 1/p_base(t). The second term, log 1/p_base(t)
  ("selection bits", ≈ log of the expected attempts to first success), is the unavoidable cost of pure
  elicitation by rejection sampling. The first term ("reshaping bits") is zero for pure elicitation and
  measures how far RL's choice *among correct proofs* departs from what the base produces when it does
  succeed. Rule: **elicited** if RL's correct-proof distribution ≈ π_base(·|S) (reshaping ≈ 0, all of RL's
  change is selection); **created/expanded** if reshaping is large, i.e. RL's success mass sits on correct
  proofs the base itself makes unlikely even conditional on success. This puts Dan's pass@k in bits:
  log2 k* ≈ selection bits, and gives a second axis that the k-objection does not touch.
- **Null / floor:** for the random-init checkpoint the selection bits log 1/p_init(t) are huge (≈ proof
  length × log |V|) but finite, which is the information-theoretic form of "any k eventually"; the
  comparison of interest is pend's selection bits against RL's actual KL budget.
- **Transfer to our setting:** per theorem, sample r16's proofs, keep Lean-accepted ones, score them
  teacher-forced under r16 and pend; reshaping = E_{y∼π_r16(·|S)}[log π_r16(y|S) − log π_pend(y|S)], with
  log π(y|S) = log π(y) − log p(t). Needs p_r16(t) (easy) and p_pend(t) (hard when tiny — use the bracket
  from the pre-registered definition 2). Cost: J1 scoring plus J2 sampling, already planned. Failure modes:
  p_pend(t) below sampling resolution makes both terms uncertain (only bounds); KL estimates are
  high-variance with few accepted samples; T 0.8 sampling vs T 1.0 scoring mismatch must be fixed in one
  convention; the forgetting law uses forward KL while the theory uses reverse KL.
- Related: `mukherjee2025subnetworks.md` (sparsity result this paper attributes to bf16),
  `lin2023urial.md` (token-level KL / rank shift), `blier2018description.md` (bits injected),
  `wu2024rareoutputs.md` and `hu2023passuntil.md` (estimating small p), `_screen_L3.md` rows for Zhao et al.
  2024 (twisted SMC: estimating p(S)) and Korbak et al. 2022 (KL-regularised RL as Bayesian inference).
