---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hewitt2019designing
---

# Control tasks and selectivity: a null model for 'the representation already encodes it'

Paper: [@hewitt2019designing] (Hewitt & Liang, 2019)
Source: arXiv 1909.03368v1 (HTML rendering read in full: Abstract, Sec. 1-6, Tables 1-2; no appendix exists)

## Learnings

- **The problem.** A probe is a supervised model trained to read a property off a frozen representation. "as
  long as a representation is a lossless encoding, a sufficiently expressive probe with enough training data
  can learn any task on top of it" (Sec. 1). So high probe accuracy alone does not show that the property is
  in the representation: the probe may have learned the task itself.
- **Control task (the null).** "control tasks, which associate word types with random outputs" (Abstract): for
  each word type v a behaviour C(v) is drawn independently at random, and every token of v gets that output
  (Sec. 2.1, Eq. 1). Same input and output space as the real task; label marginals matched ("so the marginal
  probability of each label is similar", Sec. 2.1 fn. 2). "By construction, these tasks can only be learned by
  the probe itself" (Abstract). Randomness is per type, not per example: "randomness is applied at the
  type-level rather than at the example-level" (Sec. 5.1), so the control task has strong but non-linguistic
  structure and its ceiling is "the fraction of tokens in the evaluation set whose types occur in the training
  set" (Sec. 2.3).
- **Selectivity (the measure).** "Selectivity is defined as the difference between linguistic task accuracy and
  control task accuracy" (Fig. 2 caption); a good probe "should be selective, achieving high linguistic task
  accuracy and low control task accuracy" (Abstract; Sec. 6).
- **Main numbers (ELMo layer 1, Penn Treebank).** MLP probe on part-of-speech: "97.3 accuracy ... compared to 92.8
  control task accuracy, resulting in 4.5 selectivity"; linear probe: "97.2 accuracy, and 71.2 control task
  accuracy, for 26.0 selectivity" (Sec. 1; Table 1). With selectivity-chosen hyperparameters (rank-10 for PoS,
  Table 1 caption) linear reaches 97.0 / 64.0 / 33.0 and MLP-1 97.2 / 80.6 / 16.6 (Table 1, bottom). Default
  probes "are over-parameterized and needlessly high-capacity" (Sec. 3.6).
- **Complexity control.** Rank / hidden size, training-set size and weight decay raise selectivity at little
  accuracy cost; dropout and early stopping mostly do not (Sec. 3.5, Fig. 4). Regularising to close the
  train-test gap "is insufficient if one is interested in selectivity" (Sec. 3.2). Restricting probe data rests
  on "the intuition that general rules can be learned more sample-efficiently than memorization" (Sec. 3.2), and
  for PoS "learning each linguistic task requires fewer samples than our control task" (Sec. 3.5).
- **Random-representation baseline vs selectivity.** Proj0, an "untrained BiLSTM run on the non-contextual
  character CNN word embeddings of ELMo" (Sec. 4.1), gets 96.3 PoS accuracy with a linear probe vs 96.6 for ELMo2,
  but selectivity 20.6 vs 31.4 (Table 2). "it might be thought that ELMo2 encodes nothing about part-of-speech,
  since it doesn't beat the Proj0 random representation baseline" (Sec. 4.2); with selectivity, ELMo2 probes "must
  rely on emergent properties of the representation" (Sec. 4.2). Likewise ELMo1 beats ELMo2 on accuracy by 0.6
  but loses on selectivity by 5.4: "the linear probe on ELMo2 achieves selectivity of 31.4, compared to
  selectivity of 26.0 for ELMo1" (Sec. 4.2).
- **Framing.** Probes are not thermometers: "we suggest that probes be thought of as craftspeople" (Sec. 6).

## Evidence and limitations

- Evidence: Table 1 (probe families × default / dropout / selectivity-tuned), Fig. 4 (five complexity controls),
  Table 2 (Proj0, ELMo1, ELMo2), one representation family (ELMo), two tasks (PoS, dependency edges), PTB dev set.
- Selectivity is a difference of two accuracies with no significance test, no seeds reported in the tables and
  no principled threshold; hyperparameters were hand-picked by selectivity (Table 1 caption), which can overfit.
- Internal inconsistencies: Sec. 3.5 says "the bilinear probe achieves 16.7 selectivity", but Table 1 gives 6.6
  for the default bilinear probe and 16.7 with 0.4 dropout; the Table 1 caption says all selectivity-designed PoS
  probes use rank 10, while Sec. 3.5 says "We chose rank constraints of 10 and 45, respectively" for linear and MLP.
- The control task only nulls *type memorisation*. It says nothing about whether the model *uses* the decoded
  property (no causal test); a selective probe can still read a feature the network ignores. Later work
  (MDL probing, `voita2020mdlprobing.md`; amnesic probing and the Ravichander et al. critique, rows in
  `_screen_L6.md`) addresses these gaps.

## Connections and questions

- **Definition offered:** 'the representation encodes property Y' = a low-capacity probe decodes Y from frozen
  activations with high accuracy **and** high selectivity, i.e. well above what the same probe reaches on a
  matched control task whose labels are random per type. The quantity is accuracy(Y) − accuracy(control), computed
  on held-out data at fixed probe family and hyperparameters.
- **New vs better access:** not posed for training stages, but the measure gives a clean decision rule when
  applied to checkpoints. With the probe family fixed, compare selectivity on the **init**, **pend** and **r16**
  residual streams: 'already represented before RL' if S_pend is well above S_init (bootstrap over theorems) and
  close to S_r16; 'built by RL' if S_pend ≈ S_init while S_r16 is high. Representation present in pend but
  behaviour absent is exactly the 'elicitable' case; representation absent in pend and present in r16 is a
  candidate 'created' case. The paper's own Proj0 comparison is the template: accuracy alone would have called
  ELMo2 empty.
- **Null / floor:** two nulls, both needed here: (1) the control task (what the probe can learn by memorising
  surface types), (2) a random-weights representation (Proj0; our init checkpoint). It does not answer 'any k
  solves it eventually' directly, but the logic is the same: a capacity-matched learner on random structure is
  the floor, and only the margin above it counts.
- **Transfer to our setting:** concrete case = the excluded-middle schema. Label each proof state (goal +
  context) by a semantic property that is not a surface feature, e.g. 'classically provable but not
  intuitionistically provable' (decidable for propositional logic) or 'the reference proof's next step is
  by_contra / double-negation elimination'. Control task: give each **formula skeleton** (canonical key up to
  atom renaming) an independent random label with the same marginal, so the only way to fit it is to memorise
  skeletons; evaluate on held-out theorems whose skeletons recur (as word types recur in PTB). Collect residual
  activations at the goal's last token for all 6 layers of init, every pretraining checkpoint, pend, r8, r16
  (one forward pass per state; ~10^4 states × 384 dims), fit rank-constrained linear probes (seconds each),
  report selectivity per layer and checkpoint with theorem-level bootstrap CIs. Cost: minutes on one GPU, CPU
  for the probes. Main failure modes: (a) decodable ≠ used — the base may linearly encode 'needs reductio' yet
  never act on it, so this must be paired with a causal test (steering / ablation, see
  `venhoff2025base.md`, `ward2025repurposes.md`); (b) the label correlates with surface features (¬¬ in the goal,
  formula depth), so the control task must be built on those surface features, not only on skeleton identity;
  (c) selectivity has no natural threshold — use the init checkpoint and shuffled-label probes for a CI-based
  rule rather than a fixed number.
- Related notes: `voita2020mdlprobing.md` (MDL replaces accuracy and selectivity with codelength; its control
  tasks follow this paper), `whitney2020evaluating.md`, `xu2020usable.md` (usable information under a bounded
  probe family), `jain2023mechanistically.md` (linear readout of a capability across fine-tuning),
  `ward2025repurposes.md`, `venhoff2025base.md`.
