---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - ethayarajh2022understanding
---

# Dataset and instance difficulty as (pointwise) V-usable information

Paper: [@ethayarajh2022understanding] (Ethayarajh, Choi, Swayamdipta, ICML 2022)
Source: arXiv 2110.08420v3 (HTML rendering read: abstract, Sec. 1-3, 4.4, 5-7; App. D table referenced
but not read)

## Learnings

- **Difficulty = lack of usable information.** Dataset difficulty w.r.t. a model family V is framed "as the
  lack of V-usable information (Xu et al., 2019), where a lower value indicates a more difficult dataset"
  (Abstract). Shannon MI "is not an option—it would not change after X is encrypted" (Sec. 2.1).
- **Estimation needs only two fine-tunes.** H_V(Y|X) is the held-out log-loss of V fine-tuned on (x, y);
  H_V(Y) that of V fine-tuned with the input replaced by a null input ∅ (empty string); "estimating
  V-information involves training or finetuning only two models" (Sec. 2.2).
- **Pointwise V-information (PVI)** (Def. 3.1, Eq. 4): pvi(x→y) = −log2 g[∅](y) + log2 g′[x](y), i.e. for a
  held-out instance it "is the difference in the log-probability these models place on the gold label"
  (Sec. 3). "pvi is to V-information what pmi is to Shannon information" (Eq. 5): V-information is the
  expectation of pvi. "Although the V-information cannot be negative, the pvi can be"; "A negative pvi simply
  means that the model is better off predicting the majority class than considering X" (Sec. 3).
- **PVI predicts correctness, with a common threshold.** Mean pvi gap between correctly and incorrectly
  predicted held-out instances (BERT-base) "is 3.03, 2.87, and 2.45 bits respectively" for SNLI, MultiNLI,
  CoLA (Sec. 3.2). "the point at which instances start being incorrectly predicted is similar across
  datasets" (pvi ≈ 0.5 bits; Sec. 3.2, Fig. 3).
- **Stability depends on how much usable information there is.** "The cross-model Pearson correlation
  between pvi estimates of SNLI instances is very high" (r > 0.80); lower for CoLA; across seeds
  "the correlation across seeds is r>0.85" (SNLI, BERT-base, 4 seeds); "if a dataset contained no usable
  information, then we would expect the correlation between pvi estimates across different models and
  seeds to be close to zero" (Sec. 3.2; Table 6 in App. D not read).
- **More sensitive than accuracy.** V-information falls with over-fitting before accuracy does, because
  "the models start becoming less certain about the correct label long before they start predicting the
  wrong label" (Sec. 2.5, Fig. 2).
- **Information beyond a baseline.** Using conditional V-information (Hewitt et al. 2021, Eq. 6), "offensive
  words contain 0.482 bits of BERT-usable information about the label beyond that which is contained in
  text sentiment" (Sec. 4.4).
- **Against order-dependent MDL.** Rissanen data analysis is rejected for difficulty: "Since the framework
  depends on the order of instances (i.e., what data has been transmitted thus far), it is unsuitable for
  estimating dataset difficulty" (Sec. 5).
- **Open problem relevant to us.** Future work: "Extending V-information to open-ended text generation,
  which does not induce explicit distributions over the output space" (Sec. 6). Also: "the average pvi of a
  slice of data is not its V-information" (Sec. 3.1).

## Evidence and limitations

- Evidence: Fig. 1-3 (SNLI / MultiNLI / CoLA), Table 1 (hardest CoLA items, several mislabelled), Sec. 4
  artefact analyses. Classification tasks only.
- PVI is relative to the training distribution of the two fine-tunes; per-instance values are noisy when the
  dataset carries little usable information (Sec. 3.2).

## Connections and questions

- **Definition offered:** difficulty of an instance for a model family = pvi, the log-probability gain on
  the gold output from conditioning on the input, with both conditional and null models fit within the same
  family; dataset-level ease = V-usable information (mean pvi).
- **New vs better access:** not addressed. Two transferable pieces: (1) **the null-input subtraction** —
  a proof can be likely because it is generic, not because the model understood the theorem; pvi(t → y) =
  log2 π(y | t) − log2 π(y | ∅ or a shuffled theorem) separates "can prove t" from "writes common proof
  text". (2) **Family-relative difficulty at instance level**: compute pvi of each theorem's reference
  proof under V(pend) and V(r16) (each = "the checkpoint, fine-tuned with the same small budget"); a
  theorem whose pvi under V(pend) is already above the correctness threshold is elicited, one with
  pvi(pend) ≪ 0 but pvi(r16) high is a candidate creation. The threshold has to be recalibrated for
  sequences (the 0.5-bit value is for 2-3-way classification).
- **Null / floor:** the null-input model is a built-in floor (independence ⇒ zero information). It does
  not address "any k eventually": pvi is a log-probability, so sampling budget never enters; a floor from
  the random-init family would be the analogue of the paper's "no usable information ⇒ correlations ≈ 0".
- **Transfer to our setting:** cheap — teacher-forced scoring of reference proofs with and without the
  theorem statement (empty or mismatched statement) under init / pend / r8 / r16; optional matched
  fine-tunes. Main failure modes: single-reference likelihood ignores other valid proofs (the gap the
  authors themselves flag for generation); our proofs mention hypothesis names, so a null input changes the
  token meaning — a mismatched-theorem control is better than an empty one; per-theorem estimates are noisy
  where the model has little usable information (exactly the theorems we care about).
- Related: `xu2020usable.md` (definitions), `voita2020mdlprobing.md` and `blier2018description.md`
  (order-dependent codes the authors argue against), `_screen_L3.md` rows for Hewitt et al. 2021
  (conditional probing) and Perez et al. 2021 (Rissanen data analysis).
