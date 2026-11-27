---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M3 steer, Sonnet 5.5)
papers: [venhoffBaseModelsKnow2025]
reading_level: full text, all 45 pages incl. appendices A-G (the taxonomy tables G.1-G.9, pp22-45, were read but are not used), arXiv v4 (7 Jul 2026, ICML 2026 proceedings version). Versions v1-v3 not read. Tables extracted cleanly; no TeX read. Code (github.com/cvenhoff/thinking-llms-interp) not read.
---

# Venhoff et al. 2025/2026, "Base Models Know How to Reason, Thinking Models Learn When" (arXiv 2510.07364v4)

Page numbers are PDF pages (`=== page i/45 ===`).

## Warning: v4 is not the steering paper the earlier review described

v4 introduces SAE taxonomies and "constructive model diffing". Its steering vectors are **optimised by gradient**, not built by difference of means. The difference-of-means construction is in the earlier workshop paper (Venhoff et al., "Understanding reasoning in thinking language models via steering vectors"), which v4 cites (p5, p12) and Ward et al. summarise (Appendix A, p7 of that paper). I did not read the workshop paper.

## How the vector is built (category vectors, section 3.3 p5; Appendix E.2 pp19-20)

1. Thinking-model rollouts (vLLM, T 0.6, top-p 0.95, max 2,048 tokens) on 9,794 training questions (MATH, Natural Reasoning, SciBench, TheoremQA; 995 held out) (p19).
2. Teacher-force the rollout through the **base** model; positions where the base model's greedy token differs from the thinking model's token are "disagreement positions" (cap 64 per example-category) (p19).
3. Label each disagreement position with an SAE category computed from the **thinking model's** hidden state at the SAE layer (argmax latent) (p5, p19).
4. Jointly optimise, **in the base model**, a matrix `V in R^{K x d}` of per-category vectors and a small MLP (Linear(d,512), GELU, K heads Linear(512,1), Softplus) that predicts a non-negative coefficient alpha per position from the base residual `h_p`. The applied shift is `alpha(h_p, c) * V_c`. Loss: cross-entropy on the thinking model's next token at disagreement positions, per-category mean (p19). AdamW, lr 1e-2 for vectors and 1e-3 for the MLP, wd 0.01, grad clip 1.0 (MLP), up to 10 epochs, batch 4.
- **Layer:** "approximately 37% of model depth, which prior work suggests is most causal" (p5); in practice layer 27 of 64 for 32B (42%) (App. G.8, p40). SAE layer chosen by a grid (Figure 2, p4).
- **Selection:** three independent vector sets per pair; best by gap recovered on a 788-question holdout mix (p19-20). Selection is on train-split data, not the benchmarks.

## How it is applied (section 3.5-3.6 pp5-6)

- The base model generates greedily; at each generated token, "we check whether the base- and the thinking model disagree in their next token prediction. If not we generate the jointly predicted tokens. If they disagree, ... we obtain the SAE activations at that token position and select the highest activating category" and steer with it (p5). So **the thinking model is run alongside** to decide when and with which category; the coefficient MLP gives strength (p6). The steering is sparse: 5-12% of tokens for ORZ pairs (Table 2, p7); 12-29% for distilled and mixed.
- Base models get a bare "Answer the following question: Q: ... A:" prompt, no chat template (p20).

## Cross-model construction

- The vectors live in the **base** model's residual stream and are applied to the base model. No vector computed from one model's activations is added to another model. The thinking model contributes (a) the target tokens, (b) the SAE category label and gate. This is not "final minus pre-RL activations".
- Closest relevant point: a vector set trained to make the base model reproduce the thinking model's tokens recovers about 76% of the gap for RL-trained pairs (Open-Reasoner-Zero, direct RL from base) and about 11% for SFT-distilled pairs (abstract p1, Table 1 p7). This is evidence for "RL changes are mostly orchestration of existing mechanisms", for models of 0.5-32B. It is not evidence that a mean activation difference works.

## What was measured

- Table 1 (p7): GSM8K, MATH500, a 1,000-question Hendrycks-MATH holdout, accuracy graded by Claude Sonnet 4.6 (kappa 0.88 vs humans, App. C p17), 3 judge repetitions per rollout, "mean +- std over all rollout-judge pairs". Base and hybrid greedy (so their +- is judge noise only, my inference); thinking models 3 rollouts at T 0.6. Token budget 2,048 (the 32B ORZ pair is therefore truncated at 2k tokens). "Rec.%" = (hybrid - base)/(thinking - base).
- RL pairs: ORZ-32B (base Qwen2.5-32B): GSM8K 69.5 / 95.5 / 97.2 (base / hybrid / thinking), recovery 93.6%; MATH500 61.1 / 81.8 / 95.8, 59.8%; Hendrycks 65.7 / 80.6 / 95.9, 49.2%. ORZ-7B: 87.8 / 49.7 / 49.1%. ORZ-1.5B: 71.3 / 82.7 / 88.3%. ORZ-0.5B: 83.0 / 95.3 / 107.4%.
- SFT pairs: R1-Distill-1.5B and 8B recover -3 to +6% (negative values); 14B 15-22%; 32B 12.5-31.8%. QwQ-32B 14-29%.
- No seeds across vector training runs are reported (best of three is selected); no prompt-level confidence intervals; no multi-seed pass@k.

## Controls (section 3.8, p7; Figure 4 p8; two RL pairs, ORZ-1.5B and 32B, MATH500+GSM8K average)

- Full pipeline 77% / 77%; random category 20% / 28%; **random norm-matched vectors** (real MLP, real category choice) -28% / -13%; MLP magnitude only (random vectors and random category) -8% / -1%; random positions -8% / 28%. The random-vector control sets "norm-matched random directions" (p7) and used the trained coefficient MLP, which produced "confident coefficients along meaningless directions" that harm the model. So random-direction controls here lower accuracy; a strength matched to the real vector's effective strength damages the base.
- Coefficients are large: mean MLP alpha about 10-11 for ORZ-32B versus 0.7-1.8 for smaller RL pairs (Table 5, p22); units are relative to the learned vector norms, which are not reported.

## Ambiguous or unsupported

- Whether the "gap recovered" survives sampling at T 1.0 with long budgets (only greedy base/hybrid, 2k tokens). Whether the vectors need the thinking model run in lockstep (they do, here); a static vector without the gate was not tested (the closest, "random positions" and "MLP magnitude only", are ablations of the gate, not a static steering).
- p9 admits "selecting appropriate steering strengths is a known challenge"; strength is learned on the training split's next-token loss, not tuned for accuracy.
- Benchmarks are mathematical; the taxonomy is biased to math (p9). No instruction-following test.
- The claim that RL "teaches heuristics" and SFT teaches mechanisms rests on the recovery gap and a training-loss comparison (Figure 3 p6) across different base families; confounds (family, size, data) are not controlled except two size-matched pairs.

## How our setting differs

- Ours would be one static vector (mean activation difference, final minus `dpo`), no thinking model run in lockstep (we could only afford it if both models were resident), no coefficient MLP, no gate: strictly weaker than Venhoff's hybrid. If even the full hybrid at 32B recovers 49-60% of a math gap, a static vector for IFBench at 4-7k tokens should be expected to do less.
- Our pair is one RL lineage (the final model is `dpo` plus RL, like ORZ is base plus RL, but starting from an already SFT+DPO model); the "RL = orchestration" conclusion is the supportive prior.
- Our response lengths (4-7k tokens) are far above their 2,048 cap.

## Changes relative to the earlier description

- The earlier review described steering by activation difference. In v4 the vectors are optimised, per-category, on the base model, gated by the thinking model's SAE; recovery is on math, not instruction following. There is no model-minus-model mean difference anywhere in the paper.
- Reported recoveries are 76% (RL) and 11% (SFT) on average over three math sets; the ORZ-7B and 32B pairs on MATH500 are 49.7% and 59.8%, not "about 76%".

## Not read

- arXiv v1-v3; the workshop paper; the code; the cited work on rank-1 LoRA (Ward et al. 2025b) mentioned on p8.

## Correction after audit (2026-11-27 01:56 PT)

Appended after the auditor's Turn 2 in `discourse/steer.md`; the text above is unchanged.
- "Layer 27 of 64 for 32B (42%) (App. G.8, p40)" was wrong as a steering layer. G.8's "Layer 27" is the SAE taxonomy layer; App. E.2 (p19) distinguishes "the SAE layer" from "the steering layer", and the steering layer per model is not reported. Only "approximately 37% of model depth" (p5) is stated, and the SAE layers listed in App. G (4 of 28, 20 of 28, 38 of 48, 27 of 64) cannot all be that.
- "About 11% for SFT pairs" repeats the abstract. From Table 1 (p7) the mean recovery of the four R1-Distill pairs over the three sets is 8.5%; it is 10.7% if QwQ-32B is included (my computation). The abstract's 11% is therefore not the R1-Distill-only figure.
