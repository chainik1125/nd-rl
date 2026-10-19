---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - schaeffer2023mirage
---

# Are Emergent Abilities of LLMs a Mirage? — metric choice and resolution

Paper: [@schaeffer2023mirage]
Source: arXiv 2304.15004v2 (HTML rendering read: abstract, Sec. 1-7; appendices skimmed by heading only)

## Learnings

- **Thesis.** "emergent abilities appear due the researcher's choice of metric rather than due to
  fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics
  produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous,
  predictable changes in model performance" (Abstract).
- **The definition being attacked** (Wei et al. 2022): "abilities that are not present in smaller-scale
  models but are present in large-scale models; thus they cannot be predicted by simply extrapolating the
  performance improvements on smaller-scale models" (Sec. 1). Two defining properties: "Sharpness,
  transitioning seemingly instantaneously from not present to present" and unpredictability (Sec. 1).
- **Mechanism.** If per-token cross-entropy falls smoothly, the per-token probability of the correct token is
  exp(−L_CE(N)), and an all-tokens-correct metric gives "Accuracy(N) ≈ p_N(single token correct)^num. of
  tokens" = exp(−(N/c)^α)^L (Sec. 2), which looks sharp on a linear-log plot; Token Edit Distance
  ≈ L(1 − p_N) is smooth (Sec. 2, Fig. 2). The independence assumption is flagged: "While the independence
  assumption is not true, the approximation yields results qualitatively matching" (Sec. 2, footnote 1).
- **Second cause: resolution.** Apparent emergence is caused "secondarily by possessing too few test data
  to accurately estimate the performance of smaller models, thereby causing smaller models to appear wholly
  unable to perform the task" (Sec. 1), with resolution "set by 1/test dataset size" (Sec. 2). With more test
  data, "all models in the InstructGPT/GPT-3 family achieve above-chance accuracy" (Sec. 3, Fig. 4).
- **Meta-analysis.** "of the 39 preferred metrics in BIG-Bench, at most 5 display emergence" (Sec. 4,
  Fig. 5A); "2 metrics account for >92% of claimed emergent abilities": Multiple Choice Grade and Exact
  String Match (Sec. 4, Fig. 5C). LaMDA's emergence under Multiple Choice Grade disappears under Brier
  Score (Sec. 4, Fig. 6). Emergence can be induced in vision models by choosing a thresholded metric
  (Sec. 5, Fig. 7-8).
- **Limits of the claim.** "nothing in this paper should be interpreted as claiming that large language
  models cannot display emergent abilities" (Sec. 7); "the researcher can choose a metric to create an
  emergent ability or choose a metric to ablate an emergent ability" (Sec. 7). They contrast Caballero et
  al., who "explain emergence by assuming a piece-wise power law functional form; under this view, emergent
  abilities are real" (Sec. 6). They also warn that emergence claims "are possibly infected by a failure to
  control for multiple comparisons" (Sec. 7).

## Evidence and limitations

- Evidence: Fig. 3-4 (GPT-3 arithmetic, accuracy vs token edit distance, larger test sets), Fig. 5-6
  (BIG-Bench meta-analysis), Fig. 7-8 (induced emergence).
- The axis is model scale N (fixed family), not training stage or RL. The per-token independence model is
  an approximation (footnote 1). The paper does not show that a continuous metric is always the "right"
  one — only that a discontinuous one can manufacture sharpness.
- Not checked: App. A derivations (per-token error resolution limits), App. B.

## Connections and questions

- **Definition offered:** an ability is a property of a (task, metric, model family) triple; the paper
  argues the underlying quantity is the per-token error rate / log-likelihood, which changes smoothly,
  while "ability present/absent" is a thresholding of it. Wei et al.'s definition (emergence = not
  predictable by extrapolating smaller models) is the target.
- **New vs better access:** the paper's logic transfers directly: pass@k at fixed k is a thresholded,
  nonlinear transform of per-theorem log p (p^L-type behaviour for a proof of L steps), so a jump in
  pass@k from pend to r16 is not evidence of a new capability by itself. The test it suggests: if a
  continuous measure (teacher-forced per-step log p of proofs, or log p_i from sampling) moves smoothly and
  by an amount predictable from pend's trend, the pass@k jump is "metric emergence". Something that would
  count as "new" in this frame: a change that is discontinuous or unpredictable even under the continuous
  metric (or the piece-wise power law case of Caballero et al., Sec. 6).
- **Null / floor:** resolution, not a random-weights floor: a model that "appears wholly unable" may just be
  under-sampled (Sec. 1). For us the per-theorem resolution is 1/n (n = 256 or 512 samples), so n_ok = 0
  means "p below ≈ 3/n", not "absent".
- **Transfer to our setting:** compute, for pend, r8, r16 (and pretraining checkpoints if available),
  per-theorem teacher-forced log p of the reference proof and its per-step terms, and the sampling-based
  log p̂_i; plot both against training stage alongside pass@k. Cost: one forward pass per proof per
  checkpoint (cheap). Failure modes: (1) teacher-forced log p scores one reference proof, while RL may move
  mass to different proofs (pass@k can rise while reference log p falls), so it is a lower bound on log p_i
  only; (2) the multiple-comparisons warning applies to us too: ~320 theorems × several checkpoints ×
  metrics will produce some "emergent-looking" theorems by chance.
- Related: hu2023passuntil.md (resolution via sampling until success), wei2022emergent and du2024loss
  (screened), schaeffer2025powerlaws.md.
