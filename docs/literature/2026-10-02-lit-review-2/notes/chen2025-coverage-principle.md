---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - chen2025coverage
  - foster2025goodfoundation
---

# The coverage principle (and Foster et al. on coverage-bounded exploration)

Paper: [@chen2025coverage]
Source: https://arxiv.org/abs/2510.15020v2 (v2 reviewed; html text). Also screened: https://arxiv.org/abs/2503.07453v2 [@foster2025goodfoundation] (pdf text, abstract and Sec. 1 only)

## Learnings
Paper claims (2510.15020):
- The coverage profile is $\texttt{Cov}_{N}(\pi\|\widehat{\pi}) = \mathbb{P}_{x,\,y\sim\pi}[\pi(y|x)/\widehat{\pi}(y|x) \ge N]$ (Eq. 1). It is the mass of data-distribution responses that the model under-weights by a factor of at least N. A good coverage profile "is necessary and sufficient for Best-of-N to succeed" (Sec. 1).
- "cross-entropy can be anti-correlated with BoN performance" (Sec. 1, Fig. 1). Coverage "generalizes faster than cross-entropy, avoiding spurious dependence on … sequence length" (abstract, Fig. 2; graph-reasoning task trained from scratch, App. C).
- The paper gives tournament procedures for checkpoint selection, which select better models for Pass@N than minimising KL does (Fig. 1 caption, Sec. 6.3).
- Some coverage "is thought to be necessary for the success of post-training methods like GRPO" (Sec. 2). This is stated as a belief, not proved for RL.

Paper claims (2503.07453, Foster/Mhammedi/Rohatgi):
- Coverage "lower bounds the runtime of any algorithm in our framework" (abstract, item 1). Training-time interventions "cannot achieve similar guarantees in polynomial time" (item 3). Multi-turn exploration can replace "sequence-level coverage with token-level coverage" (item 4).

Our interpretation: theory says that cheap RL can only reach what the base covers at about the 1/N level. "Elicitation" then means that the solution was covered at budget N, so the natural threshold is log N, not a calibrated one. Coverage is sequence-level and measured against the data distribution. It is a property of the whole class of correct responses, not of one proof's worst step.

## Evidence and limitations
- Read the abstract, Sec. 1–2.1 and the captions of Figs. 1–2. Did not check the proofs, Sec. 6 algorithms, or App. C.
- The theory concerns BoN and next-token prediction. For RL it is supported by Foster et al. under linear-softmax assumptions. A runtime lower bound is not a proof that RL cannot create.
- The experiments are small graph-reasoning models, which is close to our regime.

## Connections and questions
- Our worst-step w1 is a token-level coverage proxy. Foster's item 4 says token-level coverage is the relevant quantity only for step-wise (multi-turn) exploration. Our ladder samples whole proofs, so sequence-level log p is the governing quantity. Fix: report sequence log p of the eventual and reference proofs alongside w1, and pre-register "covered at budget k" as log p ≥ −log k (k = 256 gives −5.55 nats). This replaces the post-hoc threshold in `rl-from-ckpt`.
- One proof's log p is a lower bound on the model's mass on correct proofs, which is what pass@k reflects. For a C theorem, a −12-nat reference worst step does not exclude other proofs. Cheap complement: the empirical solve rate at large k on B and C.
- The trajectory finding that B's eventual-proof worst step rises in pretraining after step 1,600 fits coverage improving while CE plateaus. Check this with Cov_N computed on held-out generator proofs at each of the 14 checkpoints.
- Related: karan2025-power-sampling.md (sampling sharper from the base), he2025-rewarding-unlikely.md.
