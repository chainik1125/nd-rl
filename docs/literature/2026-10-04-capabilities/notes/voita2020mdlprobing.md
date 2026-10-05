---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - voita2020information
---

# MDL probing: what a representation "encodes" = description length of the labels given it

Paper: [@voita2020information] (Voita & Titov, EMNLP 2020)
Source: arXiv 2003.12298v1 (HTML rendering read: abstract, Sec. 1-4, 6; appendix not read)

## Learnings

- **Problem: accuracy cannot separate trained from random.** Probe accuracies "do not substantially favour
  pretrained representations over randomly initialized ones", and accuracy on real labels and on random
  control tasks can be similar (Abstract). Earlier work had to shrink the probe or its training data to see a
  gap (Abstract; Sec. 1).
- **Change of measure.** "the measure of interest changes from probe accuracy to the description length of
  labels given representations"; "the description length evaluates 'the amount of effort' needed to achieve
  the quality", where effort is "(i) size of a probing model, or (ii) the amount of data needed" (Abstract).
- **Codes.** Data code under an agreed model: L_p(y|x) = −Σ log2 p(y_i|x_i), "the Shannon-Huffman code" (Sec.
  2.1, Eq. 1), i.e. the cross-entropy. Floor/ceiling references: uniform code n log2 K bits and the prior
  code H(y) (Sec. 2.1). The gain over H(y) is bounded: "the compression is limited by the mutual information
  (MI) between inputs ... and outputs" (Sec. 2.1).
- **Online (prequential) code** (Sec. 2.2.2, Eq. 4): send the first block with a uniform code ("Alice starts
  by communicating y_{1:t1} with a uniform code"), then repeatedly train the probe on everything sent so far
  and code the next block with it; blocks at "0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 percent of
  the dataset" (footnote 4). "The online code is related to the area under the learning curve" (Sec. 1;
  Sec. 2.2.2).
- **Splitting the code into data and model parts.** Model cost can be read off the online code "by
  interpreting the difference between the cross-entropy of the model trained on all data and online
  codelength as the cost of the model" (Sec. 2.3). Variational (two-part, bits-back) code: KL(β‖α) plus the
  expected data code (Sec. 2.2.1, Eq. 3). The two codes agree (Table 2), and "the ability of a probe to
  achieve good quality using small amount of data or using a small probe architecture reflect the same
  property: the strength of the regularity in the data" (Sec. 2.3).
- **Control tasks.** Codelengths for control tasks are "substantially larger than for the linguistic task
  (at least twice larger)" (Sec. 3.2). At ELMo layer 0, accuracy is 93.7 (linguistic) vs 96.3 (control) but
  the online code is 173 vs 302 kbits (Table 2): "for layer 0, accuracy for the control task is higher, but
  the code is twice longer than for the linguistic task" (Sec. 3.2).
- **Random-init models.** "codelength shows large difference between trained and randomly initialized
  representations" (Sec. 4); e.g. PoS layer 1: accuracy 97.8 vs 95.7 but online code 192 vs 294 kbits,
  trained vs random (Table 6). "compression bounds for the randomly initialized model are closer to those of
  context-agnostic Layer 0 than representations from the trained model" (Sec. 4.2).
- **Stability.** "In striking contrast to accuracy, MDL results are stable across settings" (Sec. 3.3);
  "accuracy is wrong for 8 out of 10 settings, MDL is always correct" (Fig. 3 caption).

## Evidence and limitations

- Evidence: Tables 2, 3, 6; Fig. 2-3; ELMo representations, PoS (Penn Treebank) plus 7 edge-probing tasks.
- Codelengths are theoretical: "we do not consider practical implementations of transmission algorithms"
  (Sec. 2, footnote 2).
- The online code is defined relative to a fixed learner (architecture, optimiser, seeds, block schedule);
  the paper shows robustness across 10 probe settings (Sec. 3.3), not across learning algorithms in general.
- Appendix (10 settings, MLP-1/MLP-2 sizes) not read.

## Connections and questions

- **Definition offered:** "how much a representation encodes property Y" (*[review]* paraphrase; the source says "how well
  pretrained representations encode some linguistic property") = the minimum description length
  of the labels given the representation, estimated by an online (prequential) code: final fit (data code)
  plus "amount of effort" (model cost = online code − full-data cross-entropy). A short code means the
  regularity is already there and a small learner / few examples suffice.
- **New vs better access:** not about RL, but the structure is exactly the one we need. The paper tells
  "already encoded" from "the probe learned it itself" by **the model part of the code**, compared against
  two nulls: control tasks (labels the representation cannot encode, so the probe must memorise) and
  random-init representations. Turned into a rule for us: treat the fine-tune on top of a checkpoint as the
  probe. For a target behaviour D (e.g. RL's accepted proofs, or the excluded-middle family), compute the
  prequential code of D when fine-tuning from pend and when fine-tuning from init (and earlier pretraining
  checkpoints). **Elicited** if pend's model cost is small in absolute bits and a small fraction of init's;
  **created/taught** if pend's model cost is comparable to init's (pend saves only the generic syntax bits).
  A natural threshold: the RL run's own information budget (bits of reward feedback it actually received,
  see `blier2018description.md` and the LoRA-capacity argument in `_screen_L3.md`).
- **Null / floor:** two explicit nulls — uniform code (n log2 K) and random-init representations — plus
  control tasks. For us the random-init checkpoint's prequential code is the "teach from scratch" ceiling,
  and the "any k eventually" objection is answered because the measure is bits, not success-at-k: a random
  model codes a 40-token proof at ~40 × log2|V| bits, a checkpoint that already "has" it at a few bits.
- **Transfer to our setting:** first block coded by the base model itself (so the first term is the
  teacher-forced −log2 π_pend(y | t) that Dan proposed), then fine-tune pend on growing prefixes of D (e.g.
  1, 2, 4, ..., 256 proofs) and code the next block teacher-forced. Cost: ~10 short fine-tunes × a few
  orderings × checkpoints; minutes each for a 9.5 M-param model. Failure modes: (1) the code depends on the
  fine-tuning recipe (lr, epochs), so all checkpoints must share it; (2) RL's own proofs were selected for
  high π_r(y), so D must be fixed independently of the model being scored (e.g. shortest known proofs);
  (3) renaming / near-duplicate proofs inflate apparent learnability (dedupe by canonical key).
- Related: `blier2018description.md` (prequential codes for deep nets), `whitney2020evaluating.md` (SDL,
  ε sample complexity), `xu2020usable.md` and `ethayarajh2022vusable.md` (usable information),
  `greenblatt2024passwordlocked.md` (demonstrations needed to unlock), `hu2023passuntil.md`.
