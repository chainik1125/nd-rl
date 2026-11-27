---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M2, Sonnet 5.5)
papers: [cai2025predictabilityrldynamics]
reading_level: full text, all 48 pages incl. appendices A-E and figure pages (extracted text read in full; figures read as text captions only, bar heights not recoverable except where I rendered pages 1, 4, 5 as images), arXiv v3 (22 Feb 2026, ICLR 2026 camera-ready). v1 and v2 not read.
---

# Cai et al. 2025, "On Predictability of RL Dynamics for LLMs" (arXiv 2510.00553v3): full-text reading note

Source: `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/papers/cai2025predictabilityrldynamics/` (PDF + `.txt`). "p. N" = the page
marker of the extracted text (= PDF page). The abs page lists v1, v2, v3. The earlier review read v3 in part (first 100k of 128k characters);
this reading covers all 48 pages. Pages 10-16 are references, 17 ethics, 35-48 are figure pages (R2 heatmap, t-SNE and PLS plots,
no numbers in the extracted text).

## 1. The exact rank-1 construction

What the paper says (p. 3, Sec. 2.1):
- Eq. 1-2: "performing SVD on dW ... dW = sum_i sigma_i u_i v_i^T ... The Rank-1 update matrix is then defined by retaining only the largest singular value: dW^(1) = sigma_1 u_1 v_1^T."
- Eq. 3: "To ensure consistency in update strength, we rescale dW^(1) to match the L2 norm of dW: dW^hat^(1) = alpha dW^(1), alpha = ||dW||_2 / ||dW^(1)||_2. The evaluation model is then obtained by adding dW^hat^(1) to the base model."
- dW is defined (p. 2, Sec. 1) as "the parameter difference between the RL-trained model and the base model". "Base" = the checkpoint RL started from (for DAPO-32B the base is Qwen2.5-32B, App. B p. 20-21; for the R1-Distill DAPO row it is the distilled model, p. 20).
- Rank-k%: "only the top k% of singular subspaces are retained" (p. 3). No formula for how the rescale applies to it.

Settled:
- **Rescaling**: yes, always, to the L2 norm of the full dW (alpha >= 1). The text writes "||.||_2". For a matrix the spectral norm ||.||_2 equals sigma_1 for both dW and dW^(1), which would make alpha = 1 and the rescale void; so the only reading that makes Eq. 3 do something is the Frobenius norm. This is my inference; the paper never says "Frobenius". Fig. 5(a) (p. 6) reports the "unscaled Rank-1" as a "percentage" of the update norm: 19.3 % for DAPO (Table 7, p. 21: "Rank-1 fraction" 19.3 %, DPO 11.42 %, offline SFT 7.7 %, offline DIST 12.6 %). If that is a norm fraction alpha is about 5; if it is an energy fraction (sigma_1^2 / ||dW||_F^2) alpha is about 2.3. Not stated which. Either way the evaluated rank-1 model carries the full update's norm per matrix, so it is not "a rank-1 piece of the update" in size, it is a rank-1 matrix amplified 2-5x.
- Scaling experiment (p. 4, Fig. 4b): lambda*alpha*dW^(1); "performance increases rapidly with lambda and then saturates, peaking at lambda = 0.7, slightly below the norm magnitude of the full update". lambda is chosen on the plotted benchmark (MATH-500, p. 5 axis).

Not settled by the paper (I looked for each):
- **Which parameters.** The word "module" is never defined. Sec. 3 and Fig. 15 (p. 35) list "MLP Gate, MLP Up, MLP Down, Attn Q, Attn K, Attn V, Attn O" per layer: so the analysed objects are the 7 matrix types, as in our 448. Embeddings are treated separately (Fig. 5b, Fig. 8, 9, pp. 5, 25): "RL methods cause minimal distribution shift of the embedding space." Nothing says whether embeddings, lm_head and norms are (a) left at base, (b) set to the RL values, or (c) excluded from "adding dW^hat^(1) to the base model" because they are not low-rank in the same way. Sec. 3.2 (p. 7) does say that in the sliding-window experiment "only the Rank-1 updates of the selected modules are injected into the base model, while other modules remain unchanged", which shows that a partial injection is implemented; for the main experiment the text is silent.
- **All at once or one at a time.** Main results (Fig. 1, 3, Tables) read as all matrices at once ("adding dW^hat^(1) to the base model", p. 3). The only subset experiment is the R2-sorted sliding window (p. 7, Fig. 6c) with windows of about 1/3 of the modules, which is subset-at-once, not one matrix at a time. Fig. 4a "Impact of i-th Subspace" is one singular index at a time (all matrices, same index), not one matrix at a time.
- **Checkpoint pair.** Base -> fully trained for Property 1; base -> checkpoint t for the "across the RL process" curve (Fig. 3b, p. 4: "during the early stages of training, the performance of the Rank-1 Subspace is slightly lower than that of the fully trained model"). Fig. 3b's y-axis starts at 45 %, which is read as the base MATH-500 accuracy (Fig. 4a labels "BASE" at about 45 %, p. 5, from the rendered image).
- **Precision of the differences.** Not stated. Training is "Float16 precision" (p. 21 for SFT, p. 22 for RL: "effective batch size of 512 under Float16 precision"); the dtype of the saved checkpoints, of the SVD, or of the added rank-1 matrix is never given. No mention of bf16 or fp32 anywhere in the 48 pages. (Searched the text for bf16, bfloat, fp32, float32.)
- **Sign handling and trajectory alignment.** For the linear-dynamics fit, u1 vectors from different checkpoints are stacked (Eq. 5-6, p. 34); an SVD vector is defined only up to sign, and the paper never says how signs are aligned across checkpoints. For AlphaRL this matters (Sec. 3 below).

## 2. How "recovers 99.17 % of the reasoning capability" is measured

- Where the number is: Sec. 1 p. 2: "for Property 1, the Rank-1 Subspace alone recovers an average of 99.17% of the reasoning capability." The abstract (p. 1) says "recovering over 99% of performance gains". Neither a table nor a figure in the 48 pages lists the 99.17 % components; Fig. 1, 3a, 7 show bars only. I could not reproduce it from the text.
- Metric: "Relative Accuracy, defined as the ratio between the accuracy of the Rank-1 reconstructed model and that of the fully trained model" (p. 4). It is a ratio of accuracies, not a fraction of the gain over base. The base accuracy is needed to convert: Fig. 3b/4a suggest MATH-500 base about 45 % vs full about 80 %, so 99 % ratio = about 97-98 % of the gain there; for benchmarks where the base is near zero (AIME24) the two coincide; where the starting accuracy is already high (GSM8K: the 5 %-trained checkpoint is at 78.5 % of a final 94.4 %, Table 3 p. 18; the base row itself is not tabulated) a 99 % ratio can correspond to a much smaller share of the gain. The average over benchmarks and models in the headline therefore mixes both.
- Which variant: Figure 1 (p. 1) compares the full model with the "Rank-1% Subspace" (top 1 % of singular directions: its caption and text say 1 %), not Rank-1; Fig. 3a (p. 4) shows Rank-1 and Rank-k% on MATH-500 only. Fig. 7 (p. 24, image only) is said to show the six benchmarks. So the 99.17 % may be rank-1, rank-1 % or a mixture; the text does not say. The Rank-1 bars of Table 8 (p. 21, MATH-500): DAPO 100.00 % at every rank from 1 to 90 %.
- Benchmarks (p. 4): "six reasoning benchmarks, including AIME24, AIME25 and MATH-500", Table 1 adds MINERVA, GPQA, GSM8K. 
- Models: Sec. 2.2 (p. 3) "eight models, covering five RL algorithms" (PPO, RLOO, GRPO, Dr.GRPO, DAPO) plus DIST and SFT; the abstract says "8 LLMs and 7 algorithms"; Sec. 1 (p. 2) says "13 diverse LLMs (ranging from 7B to 32B parameters ...) trained with 10 advanced training algorithms". App. B (pp. 20-21) lists base/trained pairs: Qwen2.5-7B (PPO, Open-Reasoner-Zero-7B), Qwen2.5-Math-7B (Dr.GRPO, Oat-Zero), Qwen3-8B-Base (RLOO, GRPO, DAPO, SFT, DPO), R1-Distill-Qwen-7B (DAPO), Llama3.1-8B-Instruct cold-started (DAPO), GLM-4-9B (DAPO), Qwen3-14B-Base (DAPO), Qwen2.5-32B -> "DAPO-Qwen-32B" (DAPO), Qwen3-4B-Base (Spiral, Kuhn Poker). The counts 8 / 13 and 7 / 10 do not agree anywhere.
- Samples and seeds: p. 8 (Sec. 4.2) "32 sampled responses per question under temperature T = 0.6", average accuracy reported. For the Property-1 experiments (Sec. 2) the number of samples and the temperature are not stated. No seeds, no repeated runs, no intervals anywhere (searched for seed, error bar, standard deviation). One training run per (model, algorithm).
- Benchmark sizes: not stated; AIME24/25 have 30 problems each (my knowledge), so one problem = 3.3 points. Table 1 values like 28.54, 24.17 are consistent with many samples per problem; the intervals over problems are not given.

## 3. AlphaRL extrapolation (Sec. 4, pp. 7-9; App. A pp. 18-20; App. E p. 34)

- **Formula.** There is no closed-form equation of the extrapolation in the main text. Sec. 4.1 (p. 7): they build "scaled Rank-1 vectors, where each vector is scaled by the product of alpha^(t) and sigma_1^(t)" (that is, u^_t = alpha_t sigma_1^(t) u_1^(t), the column of dW^hat^(1), App. E p. 34), "record the relative accuracy of the corresponding checkpoints and fit the vectors with their relative accuracies using a single-component PLS regression, establishing a linear relationship between them. Given a target relative accuracy y*, AlphaRL obtains the corresponding update vector through inversion. Finally, this update vector is combined with the left singular vector v1 to form the new Rank-1 update for each module." Only u is extrapolated; v1 is a single fixed vector (the text says "left singular vector v1"; in their notation v1 is the right one, Eq. 4 p. 34: Dy = alpha sigma_1 u_1 <v_1, h>; which checkpoint's v1 is used is not said).
- **Which early checkpoints.** Table 1 / Table 3 label rows "Training 5 %, 10 %, 20 %, 30 %, 40 %, 50 % + AlphaRL" and the caption (p. 8) says "The prediction is based on Rank-1 vectors and their corresponding accuracies extracted from checkpoints between the Base model and the current training stage". So all checkpoints from base to the row's stage are used. How many checkpoints exist per run is not stated except that "We save the checkpoint after each training batch" (p. 22).
- **How the coefficient is chosen, and on what data.** The "coefficient" is the target y*: "We use the model's test accuracy on MATH-500 and set the target accuracy to y* = 1. The update vectors obtained through inversion are then applied to all datasets for evaluation." (p. 7). So: the regression is fitted on MATH-500 test accuracy of the intermediate checkpoints, expressed as relative accuracy (to the fully trained model, so the final model's MATH-500 accuracy is a known input), y* = 1 means "reach the fully trained model's MATH-500 accuracy", and the benchmarks reported (including MATH-500 itself) are the same test data. No dev/held-out split is described. The abstract's "without ... hyperparameter tuning" (p. 1) is true in the sense that y* is fixed, but the construction uses the final model's score on the evaluation benchmark.
- **Results (Table 1, p. 8; Tables 3-4, pp. 18-19; Qwen3-8B-Base, 14B, GLM-4-9B, R1-Distill-7B, Llama3-8B).** Example DAPO Qwen3-8B-Base average over six benchmarks: full 53.38, 10 % trained 40.55, 40 % trained 46.30, 10 % + AlphaRL 46.47, 40 % + AlphaRL 53.31. Qwen3-14B: 40 % + AlphaRL 59.18 vs full 60.87. "Up to 2.5x speedup", ">96 %" (p. 1-2) = the 40 % + AlphaRL row against the full model (about 100 % for DAPO, 96 % for GRPO/RLOO, p. 8). The 10 %-step rows are lower (e.g. GRPO 43.43 vs 51.52 full = 84 %).
- **Anomalies I found in the tables.** Table 4 R1-Distill-Qwen-7B "Training 20 % + AlphaRL" row: components 60.00, 52.25, 95.50, 57.85, 48.25, 95.50 but Avg. printed 48.23 (the components average about 68.2; the neighbouring rows are consistent), a typo; Table 4 "Training 50 %+AlphaRL" 73.75 vs fully trained 73.65. Table 1 DAPO block is identical to the Qwen3-8B-Base block in Table 3 (same numbers, no independent replication). Several AlphaRL rows exceed the fully trained model on single benchmarks (GPQA 49.25 vs 48.23), with no intervals, so differences of 1-2 points are not interpretable.
- **Extra settings.** Kuhn Poker (Tables 5-6, p. 20): AlphaRL win rate at "Training 40 %" 59 % vs full model 43 % and rank-1 44 %, MATH-500 75 % vs 64 %: extrapolation overshoots the same-stage model by a lot, and rows at 80 % and 100 % are "-" (not run). Interpretable as "extrapolation beyond the observed steps lifts performance", not as accuracy of prediction.
- **Checkpoint lineage.** The paper trains single clean runs (one lineage, one optimiser state, fixed learning rate 1e-6 with warm-up, p. 22) and uses a base-to-t difference for all t. Nothing about restarts.

## 4. Linearity claims (Property 2)

- "the linear rate metric R2 exceeding 0.96" (p. 2, Sec. 1): refers to the single illustrated module of Fig. 6b (label "R2 = 0.960", p. 7 caption; Attention).
- "the linearity of Rank-1 Subspace's evolution exhibits an average R2 of 0.914" (p. 2): average over 13 models (their count), over what set of modules is not said; Fig. 15 (p. 35) is an R2 heatmap over modules and layers, MLP mid-to-high layers higher, self-attention lower (p. 6).
- "predictions of its later state based on early-stage states achieve an average error of less than 5%" (pp. 2-3): the error definition is not given; sliding-window analysis Fig. 6c says R2 and prediction accuracy both decrease as the window moves (p. 7).
- What R2 measures here: a one-component PLS of accuracy on the stacked (standardised) u1 vectors over T checkpoints (Eq. 6-10, p. 34), in-sample, no cross-validation. With the number of features (5,120 to 27,648) far above the number of checkpoints T, a one-component PLS can fit any monotone accuracy curve; and accuracy rises monotonically with training, so any smoothly drifting vector gives a high in-sample R2. The paper does not run a control such as shuffling checkpoint order or using a random trajectory. This is my reading, not a statement of the paper.
- Fig. 6a / 16 etc.: t-SNE of PCA-50 projections "smooth, nearly linear" (p. 6): qualitative.

## 5. Is low rank specific to RL? The paper contradicts itself

For specificity:
- p. 3 (Sec. 1): "Crucially, control experiments with alternative training paradigms like supervised fine-tuning and distillation on the same models yielded neither property, demonstrating that these phenomena are distinctive characteristics of the RL process for LLMs."
- p. 4: "In contrast, SFT and DIST exhibit a strong dependence on subspace rank, requiring more subspaces to achieve performance gains."
- p. 5 (Sec. 2.3): "a universal approximate low-rank property of dW in RL, which is completely absent in SFT and DIST."
- p. 5: update norms of DIST and SFT are "one to two orders of magnitude larger than those for RL" (Fig. 5a); Table 7 (p. 21): average update norm offline DIST 21.24, online DIST 1.46, offline SFT 10.75, offline DPO 0.79, online RL DAPO 0.01.
Against:
- p. 23 (App. C): "Both On-Policy Distillation and DPO exhibit clear low-rank structures in their update patterns, demonstrating that the phenomena identified in this work are not unique to RL. Notably, On-Policy RL ... shows an even more pronounced low-rank property."
- Table 7 (p. 21): rank-1 fraction offline DIST 12.6 %, online DIST 10.45 %, offline SFT 7.7 %, offline DPO 11.42 %, online DAPO 19.3 %; rank-1 % fraction 29.3, 24.78, 27.7, 24.13, 36.5 %. DAPO is higher but the others are not near zero, and "the average update norm" differs by 80x between DPO and DAPO, so rank-1 amplification differs hugely.
- Table 8 (p. 21, MATH-500 relative accuracy of the rank-1 reconstruction): offline DIST 1.5 %, online DIST 56.4 %, offline SFT 2.5 %, offline DPO 62.25 %, DAPO 100 % (rank 1), rising for DPO to 71.5 / 74.5 / 79 / 83.25 / 88.25 / 88.25 / 93.25 % at ranks 1 %, 10 % ... 90 %. So an offline DPO update (our relevant comparison: a preference-trained change) is "62 % at rank 1" and online distillation "56 %", i.e. partly low-rank by the paper's own measure. The offline DIST and SFT rank-1 rows (1.5 %, 2.5 %) are far below the base accuracy (the base is about 45 % on Qwen3-8B-Base MATH-500 per Fig. 3b/4a, though Table 8's base differs per row), so those rows show that a norm-matched rank-1 matrix of a very large update (norm 21, 10.75) destroys the model; this is not evidence that SFT directions are high-rank in the way RL directions are low-rank.
- The relevant confound the paper itself raises: on-policy sampling and a KL-type constraint (p. 23, citing Shenfeld et al.) explain concentration. That predicts "specific to on-policy, small-update training", not "specific to RL". Our run is an RL run after a DPO model, so it is inside the favourable class, but nothing in the paper tests a lineage with a prior preference-optimisation stage whose own delta sits in the starting point.
- Section 2.3 also says that, whatever the update source, the embedding shift is minimal for RL and global for SFT/DIST (Fig. 5b, 8, 9, pp. 5, 25), which is a second mechanism claim with only a t-SNE as evidence.

## 6. Precision

Not discussed. No bf16/fp32 statement for checkpoints, for dW, or for the SVD (see Sec. 1 above). Training is "Float16" (p. 21-22); checkpoints from Open-Reasoner-Zero, Oat-Zero and DAPO-Qwen-32B are third-party releases (App. B p. 20-21) whose dtype is not given. Consequence for our setting: differences of bf16 weights are quantised to the bf16 grid (65 % of Olmo's weights move by <= 1 ulp); the paper gives no evidence that its own differences were free of this, nor that its conclusions survive it.

## 7. Where the paper is ambiguous, unsupported or internally inconsistent

1. "L2 norm" of a matrix in Eq. 3 (spectral vs Frobenius; spectral makes alpha = 1).
2. Which tensors are in dW (7 matrix types per layer vs everything), and which of embeddings / lm_head / norms the rank-1 model carries.
3. 99.17 %: source not tabulated; ratio of accuracies, not share of gain; rank-1 vs Rank-1 % unclear; benchmark average mixes base-high and base-low sets.
4. Model/algorithm counts: 8 + 7 (abstract), 8 + 5 + DIST + SFT (Sec. 2.2), 13 + 10 (Sec. 1).
5. No seeds, no intervals, no number of samples for Sec. 2.
6. AlphaRL's coefficient is chosen with MATH-500 test accuracy and the final model's score; applied to benchmarks including MATH-500.
7. R2 is in-sample, one PLS component, no control; "less than 5 % error" undefined.
8. RL-specificity contradicted by App. C and Tables 7-8.
9. Lowest early-checkpoint extrapolations are noticeably worse (Table 1, 10 % rows).
10. Table 4 typo (avg 48.23), identical DAPO numbers in Tables 1 and 3.
11. Code release: "Our model and code will be available" (p. 1), "We have already released the complete code" (p. 17). I did not look at the repository (GitHub link in abstract), so the construction questions above might be answerable from code: untested.

## 8. How our setting differs

- 32B thinking model, long chains (mean 4.5k-6.9k tokens, up to 32k), IFBench strict checker (an instruction-following verifier, not math); the paper's models are 4B-32B (one 32B), math/reasoning benchmarks, plus Kuhn Poker.
- The RL run starts from a DPO model (`dpo`), not from a pretrained or distilled base; the prior DPO delta is inside the base.
- 2,300 labelled steps at LR 2e-6 with optimiser restarts, a released lineage with an abandoned branch; the path is not straight (straightness 0.10, cos about -0.2 between consecutive 50-step changes), against Cai's single clean run with a monotone accuracy curve.
- Released checkpoints are bf16; 65 % of weights are within one ulp of `dpo`.
- Our base accuracy is not near zero (pass@1 29 % on 218 arm-A prompts) but the reach prompts are at 0 % at `dpo`: the gain (0 -> 69 %) is of the kind where the "base" is a pass rate of 0, which makes the ratio and share-of-gain metrics coincide there.

## 9. Not read / not done

- v1 and v2; the released code; figure data (bar heights in Fig. 1, 3, 7, 10, 11; the R2 heatmap values). Pages 35-48 contain no extractable numbers. I rendered pages 1, 4, 5 as images (and 7, 24, 26 without reading them in detail): so Fig. 12-14 and 6c numbers are not checked.
- Equations came through intact in the text extraction (Eqs. 1-10); no TeX source was needed.

## Correction after audit (2026-11-27 02:10 PT, reader-implementer; appended, nothing above was edited)

Re-checked against the PDF (page images and text) after the auditor's Turn 2:
1. **99.17 % sentence (Sec. 2 above).** The sentence on p. 2 reads "the Rank-1 Subspace alone recovers an average of 99.17% of the reasoning capability", so it names rank-1. My "may be rank-1, rank-1 % or a mixture" was too loose. What stands: no table or figure in the paper lists the components, and Figure 1 (p. 1), the paper's headline figure, shows Rank-1 %, not rank-1.
2. **Fig. 7 (p. 24), which I had not read.** Confirmed on the rendered page: in the MATH-500 panel "Top Rank-100%" is about 26 % (DIST) and about 57 % (SFT) relative accuracy (AIME24 about 44 / 47 %, GSM8K about 62 / 61 %). Keeping all singular directions reconstructs dW exactly, so a faithful reconstruction would give 100 %. This is the best evidence in the paper on "which parameters": changed tensors outside the decomposition (embeddings, which DIST and SFT move a lot, Fig. 5b) were probably left at base. Alternatives I cannot exclude: a dtype effect in the reconstruction. It remains an inference. The 32B model (DAPO*) at Top Rank-1 is the weakest on hard sets (about 87 % AIME24, 91 % AIME25 relative). Fig. 3a, Fig. 7 and Table 8 report the same experiment with different numbers (e.g. SFT Rank-90 %: 46.75 in Table 8, about 57 in the figures).
3. **">96 %" (Sec. 3 above, "96 % for GRPO/RLOO").** From Table 1 averages: GRPO 49.42 / 51.52 = 95.9 %, RLOO 48.52 / 50.82 = 95.5 %. The text's "96 %" (p. 8) is rounded up, and the abstract's ">96 %" holds only for DAPO (53.31 / 53.38).
4. **"Nothing in the paper tests a lineage with a prior preference-optimisation stage" (Sec. 5 above) was too strong.** App. B (p. 20) has DAPO after a cold-start on Llama3.1-8B-Instruct, and Table 4 (p. 19) reports AlphaRL on it. Llama 3.1 Instruct was DPO-trained (outside knowledge, from the Llama 3 report, not from this paper). But Table 4 has no rank-1 reconstruction numbers; p. 18 asserts the rank-1 structure holds there without a table.
5. **Tables 1 and 3.** The identical DAPO numbers are the same run shown twice (Table 3 adds stages), not an anomaly in the data; the real point stands, there is no independent replication.
6. **Omitted in the note:** Fig. 14 / p. 28: "using only the first 20 tokens ... allows the base model to approach the performance of the full parameter update", i.e. the gains in those RL runs are largely an opening-style switch; Fig. 4a / Fig. 11: lower-ranked directions, each rescaled to the full norm, also lift accuracy, and there is no random-direction control.
