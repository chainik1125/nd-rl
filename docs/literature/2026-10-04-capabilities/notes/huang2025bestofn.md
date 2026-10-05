---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - huang2025bestofn
---

# Is Best-of-N the best of them? Coverage C^{π*} = E_{π*}[π*/π_ref] sets what inference-time selection can reach

Paper: [@huang2025bestofn] (Huang, Block, Liu, Jiang, Krishnamurthy, Foster, 2025)
Source: arXiv 2503.21878v2 (HTML rendering read: abstract, Sec. 1-4, Sec. 5 opening and Table 1, Sec. 6;
proofs in App. C-G not read).

## Learnings

- **Claim.** The results "highlight the importance of the pre-trained policy's coverage over high-quality
  responses for performance and compute scaling" (Abstract); the best achievable reward "irrespective of
  computational cost, is determined by the base policy's coverage over high-quality responses, along with
  the mean-squared error of" the reward model (Sec. 1.1). The true reward may be whether y "passes a proof
  checker" (Sec. 2).
- **Framework.** Per prompt, sample-and-evaluate queries: draw y ∼ π_ref, observe π_ref(y|x) and r̂(x, y);
  "The efficiency (query complexity) of the algorithm is measured by the total number of queries N"
  (Sec. 2, Definition).
- **Coverage.** C^{π*}(x) = E_{y∼π*}[π*(y|x)/π_ref(y|x)] for a comparator π* (Sec. 2.1, Eq. 5); uniform
  coverage C∞^{π*}(x) = sup_y π*/π_ref (Eq. 11). Necessity: every algorithm suffers regret
  ≥ ¼ √(C* ε²_RM) on some instance (Eq. 6).
- **Best-of-N.** For N ≥ c C^{π*} log(R_max/ε_RM), regret ≲ R_max C^{π*} log(R_max/ε_RM)/N + √(N ε²_RM)
  (Sec. 3.1, Eq. 7). The first term "reflects the extent to which the set of responses" drawn from π_ref
  contains enough to compete with π*; the second is reward hacking. N "plays dual, but opposing, roles in
  both performing regularization (smaller N to stay on-support)" and drawing better responses. Under
  uniform coverage, regret ≲ R_max exp(−N/C∞) + √(N ε²_RM) once N ≥ C∞ (Eq. 12). "softmax policies, which
  are normalized exponentials of the logits, can have exponentially large" C∞ while C is Õ(1) (Sec. 3.2).
- **Optimal alternative.** InferenceTimePessimism samples by rejection from π^χ_β = π_ref · relu(β⁻¹(r̂ − λ)),
  which "is the exact solution to the" χ²-regularised objective, with D_χ²(π‖π_ref) = ½(C^π − 1) (Sec. 4.1,
  Eqs. 16-18); regret-optimal and monotone in N (Eq. 20-21).
- **KL target is expensive at inference time.** The KL-regularised optimum π_ref exp(r̂/β) has density ratio
  ≥ exp(R_max/β), so N ≳ exp(R_max/β) "sample-and-evaluate queries are required to simulate it with
  rejection sampling" (Sec. 4.2, Remark, Eqs. 25-26).

## Evidence and limitations

- Theory is per prompt, single-shot; the reward-model error ε_RM is measured on base samples. With a
  perfect verifier (ε_RM = 0, our Lean case) only the coverage terms matter; Eq. 7's log(R_max/ε_RM)
  factor diverges as ε_RM → 0, so the uniform-coverage form (Eq. 12, R_max exp(−N/C∞)) is the one to use
  (our reading; with π* = π_B(·|S_t), C∞ = 1/p_B(t) and Eq. 12 reduces to ≈ (1 − p_B)^N, the pass@N miss
  rate).
- Experiments: GSM8K, MMLU, MATH with Phi-3-Mini and other bases, four reward models, N up to 2^13
  (Sec. 5, Table 1, Fig. 2). Not checked in detail.
- The authors' own limit: the base is treated as a black box and the analysis "does not take advantage of
  any specific properties of the policy outside of coverage" (Sec. 6).

## Connections and questions

- **Definition offered:** the base policy's capability to produce what a comparator π* produces is its
  coverage C^{π*} (L1) or C∞^{π*} (L∞); the inference-time compute needed to match π* by sampling plus
  verification scales with it (N ≈ C∞ log(1/δ) under uniform coverage).
- **New vs better access:** not posed, but the framework *is* the "elicitation by inference-time compute"
  baseline. Our derivation (Cauchy-Schwarz, not in the paper): for any comparator supported on the
  accepted set S_t, C^{π*} = (1/p_B(t)) · (1 + χ²(π* ‖ π_B(·|S_t))) ≥ 1/p_B(t), with equality iff π* is the
  base conditioned on success (χ² here is E_q[(π*/q − 1)²] with q = π_B(·|S_t), i.e. twice the paper's
  D_χ²). So coverage factors into **selection** (1/p_B, the frame's k-to-solve) and
  **reshaping** (1 + χ² between RL's correct-proof distribution and the base's own correct-proof
  distribution). Rule: take π* = π_R(·|S_t). RL's behaviour on t is *elicitable at budget N* if
  C^{π*}(t) ≤ N (BoN with N base draws and Lean reproduces RL's output distribution, not only its success);
  *beyond inference-time reach* if C^{π*}(t) ≫ K_total. This is the χ² twin of the KL decomposition in
  `shenfeld2025razor.md`.
- **Null / floor:** the null is built in: a random-init policy has finite but astronomically large
  coverage (≈ |V|^length), so BoN "eventually" works for it too; the framework answers Dan's objection by
  making N an explicit price, comparable across models. The L1-vs-L∞ distinction warns that a sup-type
  quantity (one very unlikely proof) can be exponentially worse than the average-case one.
- **Transfer to our setting:** C^{π*}(t) = E_{y∼π*}[π*(y)/π_B(y)] is estimable from r16's accepted samples
  scored teacher-forced under pend and r16 (J1 + J2): importance weights π_R(y|S)/π_B(y) averaged over
  π_R(·|S) samples. Cost: scoring only. Failure modes: the estimator is dominated by rare proofs with tiny
  π_B (heavy-tailed weights; use medians / bootstrap and report log C); p_B(t) is again only bracketed;
  expert iteration's per-round selection is BoN with N = 32 on the current model, so iterating BoN-and-train
  can exceed one-shot BoN coverage, which is exactly the create-vs-elicit question this note cannot settle.
- Related: `xie2024xpo.md` (coverability of a class vs concentrability of the base), `korbak2022rlkl.md`
  (KL target), `brown2024monkeys.md` and `kazdan2025passk.md` (coverage curves), prior-list sharpening paper
  (2412.01951, not counted).
