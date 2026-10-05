---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - aghajanyan2020intrinsic
---

# Intrinsic dimensionality explains the effectiveness of LM fine-tuning

Paper: [@aghajanyan2020intrinsic] (Aghajanyan, Zettlemoyer, Gupta, ACL 2021)
Source: arXiv 2012.13255v1 (HTML rendering read: full main text, Sec. 1-6; appendix not read)

## Learnings

- **Headline.** "by optimizing only 200 trainable parameters randomly projected back into the full space,
  we can tune a RoBERTa model to achieve 90% of the full parameter performance levels on MRPC"; "pre-training
  implicitly minimizes intrinsic dimension"; "larger models tend to have lower intrinsic dimension after a
  fixed number of pre-training updates" (Abstract).
- **Method (after Li et al. 2018).** Fine-tune only θ^d in θ^D = θ^D_0 + P(θ^d) with a fixed random
  projection P (Fastfood transform, Eq. 1-2), starting from θ^d = 0 (the pretrained weights). d90 = the
  smallest d reaching a "satisfactory solution", which Li et al. "defined ... as being 90% of the full
  training metric" (Sec. 3). Structure-aware variant SAID adds one learned scale per layer (Eq. 3). Finding
  d90 is "a continuous relaxation of the sparsification problem" (Sec. 2; Sec. 3).
- **Numbers.** SAID d90: RoBERTa-Large 207 (MRPC) and 774 (QQP); BERT-Base 1608 and 8030 (Table 1). The
  estimate is an upper bound: "it is likely that the true intrinsic dimension is much lower" (Sec. 4.2).
- **Interpretation as description length.** The intrinsic vector encodes the task relative to the
  pretrained weights, so d is read "as the minimal description length of the task within the framework
  dictated by the pre-trained representations" (Sec. 5); MRPC then needs "less than a kilobyte of data to
  encode a complex natural language task within the framework provided by RoBERTa" (Sec. 5).
- **Pretraining trajectory.** Re-training RoBERTa-Base and measuring d90 every 10k updates (binary search
  between d = 100 "and a maximum of 4 million"), "the intrinsic dimensionality of RoBERTa-Base monotonically
  decreases as we continue pre-training" (Sec. 5.1, Fig. 2). Early checkpoints have no d90 at all: "Unable
  to compute means either we could not fine-tune the full checkpoint to accuracy above majority class or
  stabilize SAID training" (Fig. 2 caption). Harder tasks (ANLI) have larger d90 at every checkpoint.
- **Scale.** Across ~a dozen pretrained models, d90 on MRPC falls with parameter count: "the more parameters
  we have in the model, the less we need to represent a task" (Sec. 5.2, Fig. 3).
- **Generalisation.** Lower d90 correlates with higher evaluation accuracy and smaller relative
  generalisation gap (Fig. 4-5); a compression bound L0(f) ≤ L̂0(f) + O(√(d/m)) (Thm. 1, Eq. 5), which
  "only apply to pre-trained methods trained with the intrinsic dimension subspace method; research has yet
  to show that standard SGD optimizes in this low dimensional space" (Sec. 5.3.1).

## Evidence and limitations

- Evidence: Table 1, Fig. 1-5; GLUE-style classification tasks only, encoder models, 90%-of-full criterion
  (relative, so a task the full fine-tune solves poorly gets a low bar).
- Random subspaces give upper bounds; d90 depends on learning-rate search, projection type and on what
  "full solution" is used as the reference (the checkpoint's own full fine-tune, Sec. 5.1).

## Connections and questions

- **Definition offered:** the complexity of a task *relative to a pretrained model* = d90, the smallest
  random-subspace dimension in which fine-tuning reaches 90% of full fine-tuning; read as the task's
  description length within the model's "framework".
- **New vs better access:** not discussed as such, but it gives a size-of-update rule: if RL's gain on a
  target set can be reproduced by fine-tuning pend inside a d-dimensional random subspace with d ≪ the
  d90 needed for a control task the model cannot already do (or ≪ d90 from init / early checkpoints), the
  gain was already "within a few hundred numbers" of pend — elicited in the description-length sense. A
  gain that needs d comparable to learning a genuinely new task from that checkpoint is created. The
  pretraining-trajectory design (d90 per checkpoint, Fig. 2) can be copied directly with our pretraining
  checkpoints: when does the excluded-middle schema become cheap (small d90) to elicit?
- **Null / floor:** early checkpoints simply have no d90 (full fine-tuning cannot exceed majority class);
  for us, the random-init checkpoint is that null, and a control task (e.g. a schema absent from
  pretraining) calibrates "small". The "any k" objection is replaced by "how many parameters must move".
- **Transfer to our setting:** D ≈ 9.56 M, so a dense projection is impossible beyond tiny d (D × d
  floats) — use Fastfood / sparse projections or per-layer low-rank (LoRA) as a proxy; train in the subspace
  either by SFT on r16's accepted proofs (distillation target) or by re-running EI restricted to the
  subspace; criterion: 90% of r16's pass@1 gain on the RL targets or on the excluded-middle family. Cost:
  a binary search over d (≈ 8-10 runs) per checkpoint × criterion; each run minutes for SFT, hours if EI.
  Failure modes: the 90% relative criterion and the reference "full solution" are arbitrary; SFT on r16's
  proofs measures how compressible r16's behaviour is relative to pend (distillation), not what EI could
  discover; random subspaces underestimate compressibility.
- Related: `mukherjee2025subnetworks.md` (RL updates are sparse but full-rank), `shenfeld2025razor.md`
  (KL to base as the size of the change), `blier2018description.md` (the intrinsic-dimension code appears in
  its Table 1), `whitney2020evaluating.md` (samples rather than parameters), `_screen_L3.md` row for Li et
  al. 2018.
