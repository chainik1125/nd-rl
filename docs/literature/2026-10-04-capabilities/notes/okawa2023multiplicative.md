---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - okawa2023multiplicative
---

# Compositional abilities emerge multiplicatively: concept distance from training orders acquisition; products of constituent competences look like sudden emergence

Paper: [@okawa2023multiplicative] (Okawa, Lubana, Dick, Tanaka, NeurIPS 2023)
Source: arXiv 2310.09336v5 (HTML rendering read: abstract, Sec. 1, 3, 3.1, 4 incl. toy analysis, 4.1, 4.2,
5; appendix not read).

## Learnings

- **Claims** (Abstract): the order in which generating a concept and composing concepts emerge "is governed
  by the structure of the underlying data-generating process"; "performance on compositional tasks
  exhibits a sudden “emergence” due to multiplicative reliance on the performance of constituent tasks";
  composing low-frequency concepts out of distribution "requires considerably more optimization steps
  compared to generating in-distribution samples".
- **Definitions** (Sec. 3). Concept variables and values; a concept class is a tuple of values; concept
  distance is "the number of elements that differ between the two concept classes" (Def. 4); the concept
  graph links classes at distance 1 (Def. 5). Def. 6: "We define a capability as the ability to alter the
  value of a concept variable v_i to a desired value c_i. We say the model compositionally generalizes if
  it can generate samples from a class" at distance ≥ 1 from every training class. "Models that just
  memorize the training data lack the capability to generate samples from out-of-distribution classes".
- **Measurement.** Accuracy for a class is "the product of the probabilities outputted by the three
  probes" (linear probes for shape, colour, size; chance 0.5 each) (Sec. 3.1). An additive measure
  "independently relates each concept variable prediction accuracy to the compositional accuracy,
  deceptively suggesting smooth progress" (Fig. 6).
- **Dynamics.** OOD ability "emerges at a rate which is inversely related to a class's concept distance
  with respect to classes seen in training" (Sec. 4, Fig. 5): the model "first memorizes the training
  dataset and then sequentially generalizes" outward (Sec. 1). Toy model: n atomic abilities, each learned
  with probability p per step; P(n) = (1 − (1 − p)^t)^n and t* = ⌈log(1 − P*^{1/n}) / log(1 − p)⌉ (Sec. 4).
- **Frequency thresholds.** "memorization occurs first, and generalization is achieved superlinearly as a
  function of data frequency"; "a critical number of samples are required before we can see the onset of
  capabilities to alter a concept" (Sec. 4.1, Fig. 8). In an adversarial training set, fine-tuning "is
  generally insufficient to enable the learning of new capabilities" (Sec. 1, Fig. 9).

## Evidence and limitations

- Small conditional diffusion models on 3-4 binary concepts (5,000 synthetic images) plus a CelebA check
  where the pattern holds "partially" (Sec. 4.2). Single architecture; seed counts not checked. The authors
  warn that the observations "should not be considered definitive conclusions that can be directly
  transferred to modern large generative models" (Sec. 1).
- The toy model assumes independent, never-forgotten atomic abilities; it explains the shape of the curve,
  not why each atomic ability is learned when it is.

## Connections and questions

- **Definition offered:** a capability is an atomic transformation (change one concept variable);
  compositional generalisation is producing classes at concept distance ≥ 1 from all training classes;
  measured multiplicatively (product of per-concept probabilities), with distance from training as the
  explanatory variable.
- **New vs better access:** not addressed directly, but the multiplicative model is a **null for
  "sudden" acquisition**: a composite's success probability can jump while every constituent improves
  smoothly. Our rule (inference): a sudden rise in the solve rate of a composite theorem counts as a new
  capability only if it is *not* predicted by the product of its constituents' competences measured
  elsewhere. Teacher-forced proof log-probability is exactly a product of step probabilities, so the test
  is to decompose the rise in log π(proof | t) across rounds into per-step contributions, and ask whether
  the steps that moved also moved in other theorems (shared atomic ability, multiplicative emergence) or
  only in this one (a new, composite-specific behaviour).
- **Null / floor:** chance level per concept (0.5) and the product of chances; no sampling budget. The
  "distance from training" variable is the useful floor: a model that memorises solves only distance-0
  items.
- **Transfer to our setting:** concept variables = schema features of a theorem (premise-free or not, main
  connective, use of ¬¬, ∨-elimination depth, classical-only…); concept distance of a held-out theorem =
  Hamming distance to the nearest pretraining theorem in that feature space. Predictions to test on EI
  ladders: (1) the round at which a theorem is first solved increases with its distance from the
  pretraining set; (2) for seed s1, the steps of the A ∨ ¬A proof whose log p rose in rounds 12-14 are the
  ones that also rose on other (e.g. reductio, ¬¬-elimination) targets. Cost: per-step scoring of a few
  proofs on round checkpoints (J1-type), CPU feature extraction. Failure modes: needs intermediate round
  checkpoints; the feature space (what counts as a concept) is a choice that must be fixed in advance;
  per-step probabilities depend on the context the model itself wrote, so "the same step" in two proofs is
  not the same event.
- Related: `arora2023skills.md` (statistical theory of tuple competence), `schaeffer2023mirage.md`
  (metric-induced emergence), `keysers2020cfq.md` (distance from training as compound divergence),
  `hupkes2020compositionality.md`.
