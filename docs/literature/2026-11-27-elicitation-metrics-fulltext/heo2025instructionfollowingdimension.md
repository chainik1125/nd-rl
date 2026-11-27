---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M3 steer, Sonnet 5.5)
papers: [heo2025instructionfollowingdimension]
reading_level: full text, all 19 pages incl. appendices A.1-A.7, arXiv v5 (28 Mar 2025, ICLR 2025 version). Extracted text read end to end; no TeX source needed (no garbled equation; the only formula is inline). Code repository not read.
---

# Heo et al. 2025, "Do LLMs 'know' internally when they follow instructions?" (arXiv 2410.14516v5): steering half

Page numbers are PDF pages (`=== page i/19 ===`).

## How the vector is built

- **Not a difference of means and not between models.** "The direction D is the weight of a linear probes trained on all IFEval-simple dataset" (p6, section 3.1). It is a logistic-style probe weight predicting success (1) versus failure (0) of one model's own response, within one model.
- **Probe training** (p4, section 2.2): "A simple linear model was trained on instruction-following success outcome, optimized for 1000 epochs with AdamW, a 0.001 learning rate, and 0.1 weight decay." No normalisation of the probe weight is stated; no bias handling is stated.
- **Data:** IFEval-simple, 5 instruction types x the same 100 GPT-4-written tasks = 500 prompts per model (p3, section 2.1). Greedy decoding labels (p4, A.4 p17).
- **Position:** "first token", defined on p4 as the representation of LLM(x1..xn), i.e. the state after the whole prompt (the position that produces the first response token). Middle and last token (after m/2 or m response tokens) are also probed.
- **Layer:** the probes in Tables 1-3 use early (14 for 7B/Mistral/Phi, 16 for 13B), middle (26 / 32) and last (32 / 40) layers (p4). For the steering table, "This adjustment was applied to the representations in the last layer of the model, as it was more robust to variations in alpha" (p6). The paper does not say which layer's probe supplied D for the steering experiment. Unsupported: that D comes from a last-layer probe is my reading, not stated.

## How it is applied

- "R_updated = R_original + alpha x D ... applied to all input representations, including both success and failure cases" (p6). "We focused on the representation of the first token, which corresponds to the input embedding before any response generation" (p6).
- **Ambiguity that matters.** A last-layer addition at the last prompt position can only change the logits of the first generated token (no later layer reads it). The paper never says whether the addition is repeated at every decoding step. The reported gains (+2 to +6 points of strict-ish success on whole responses) are hard to explain by a first-token-only logit shift for keyword or ending instructions; the likelier reading is that it is added at every step, but this is a guess.
- **Strength:** alpha chosen "for each model and instruction type using a validation set comprising 10% of the instruction data" (p7). Reported values: 0.3 (Llama-2-chat 13B and 7B), 0.1 (Phi-3), 0.15 (Mistral-7B) (p7). One value per model is reported though the text says per model and instruction type. The scale of D (unnormalised probe weight) relative to the residual norm is not given, so alpha is not interpretable across models.
- Note p6 footnote 3: training the probe on 70% and applying RE to the other 30% "were similar but slightly worse"; the headline numbers apply the probe to the same data it was trained on (in-sample).

## What was measured

- Metric: IFEval success rate (SR) from the official functions; A.4 (p17) says the "success rate (loose) metric" with greedy decoding. Quality ratio QR = share of responses scoring above 7 (GPT-4, 0-9 scale) among responses that follow the instruction (p7). Transition ratios SCR (false to true) and SPR (true to true) in Figure 3 (p6).
- Models: Llama-2-chat-7B, 13B, Mistral-7B-Instruct-v0.3, Phi-3-mini-128k. All 3.8-13B, non-thinking.
- Samples: 500 prompts per model (5 types x 100 tasks). No prompt-bootstrap or binomial interval. "Standard deviations are across three runs with different random seeds" (Table 4 caption, p6); with greedy decoding and a fixed D the only thing a seed can change is the random direction (and possibly probe training). The original SR has +-0.00 for that reason (interpretation).

## Table 4 (p6), success rate (SR), last layer

| model | original | random | inst-follow | change vs original | random change |
|---|---|---|---|---|---|
| Llama-2-7B | 0.57 +- 0.00 | 0.55 +- 0.00 | 0.59 +- 0.00 | +2 pts | -2 |
| Llama-2-13B | 0.61 +- 0.00 | 0.54 +- 0.12 | 0.65 +- 0.02 | +4 | -7 |
| Mistral-7B | 0.58 +- 0.00 | 0.56 +- 0.02 | 0.64 +- 0.02 | +6 | -2 |
| Phi-3-mini | 0.71 +- 0.00 | 0.63 +- 0.04 | 0.74 +- 0.01 | +3 | -8 |

QR is 0.87 / 0.92 / 0.95 / 0.76 original, nearly unchanged with D, lower for random on Mistral (0.86).

- **Check I ran:** the original SR equals the mean over the five instruction types in Table 10 (p18) for Llama-7B (0.569), Llama-13B (0.610), Mistral (0.585) but not Phi-3: Table 10 averages to 0.655 against Table 4's 0.71. An unexplained inconsistency (possibly Phi's keywords:frequency, which Table 3 marks "-"); it was not resolved.
- Binomial standard error at n=500, p~0.6 is about 2.2 points, so +2 and +3 are within one standard error if the comparison is unpaired; the paper does not pair.
- Reverse RE (A.7, p19): -alpha D gives 0.54 (Mistral) and 0.60 (Phi), "worse than random directions ... but the difference is not significant" and alpha was not tuned there.

## Controls

- **Random direction:** the paper says only "we compared it against random directions" (p7) and "outputs using a random directions" (Table 4 caption). How the directions were drawn (Gaussian? unit norm? matched norm to D? how many draws), and the scale alpha used for them, are not stated anywhere in the 19 pages (searched for every use of "random"). The 0.12 standard deviation for Llama-13B shows large sensitivity to the draw. Unsupported claim: that random directions are an equal-norm control.
- No control from a different layer, no shuffled-label probe, no mean-difference baseline.

## Other findings relevant to us

- The probe does not generalise across instruction types: AUROC 0.50-0.55 (Table 1, p4; Table 3 p5; expanded Tables 7-8 pp17), and the authors conclude there may be no "global" instruction-following dimension (p5, A.3). A probe trained on one instruction type's success within a model is not a general "comply" direction. The steering table's D is trained on all five types at once.
- The direction is closest in cosine to prompt rephrasing (Figure 4, p9; 20 prompts x 5 rephrasings, forbidden-keyword type only, two models).

## Ambiguous or unsupported

1. Position and repetition of the addition (first token only or every step); layer of D's probe; norm of D; how random directions were built; the Phi-3 base-rate inconsistency; whether QR gain is meaningful (computed over a changing denominator, successes only).
2. In-sample probe and in-sample alpha selection (10% validation, but D itself is trained on all data).
3. No error bars over prompts; seeds are not defined.

## How our setting differs

- Our vector is a mean activation difference between two checkpoints of one 32B thinking model, not a success/failure probe weight within one model. Heo gives no support for a checkpoint-difference vector; it is evidence only that a small residual push can move IFEval-style compliance by a few points on small chat models.
- Responses are 4-7k tokens of thinking then an answer; Heo's responses are short. A one-position addition cannot be expected to persist over thousands of tokens; any steering must be added at every generated position, and Heo does not test that.
- Olmo 3 blocks apply RMSNorm to each sublayer output before the residual add, so a residual addition is not absorbed by a following norm in the same way as in pre-norm Llama-2/Mistral; strength must be set against the residual norm of our model, not Heo's alpha.
- Our checker is IFBench strict (official strict checker), Heo's is IFEval loose. Heo's effect sizes (2-6 points) are on a base rate of 57-71%; ours is 29% at `dpo`.

## Changes relative to the earlier note (`notes/heo2025-if-dimension-and-gepa.md`, REVIEW.md section 7)

- "Steering arm = mean activation difference s2300 minus t" is our design, not Heo's: Heo's D is a probe weight.
- The earlier note called the random-direction control a clean comparison; it is under-specified (no norm, no draw count).
- Earlier numbers (0.57 to 0.59 etc.) are correct; the Phi baseline disagrees with the per-type table.
- The earlier review's phrase "first-token representation" is right but hides that the intervention's persistence over decoding is unspecified.

## Not read

- The code repository (github.com/apple/ml-internal-llms-instruction-following), where the answers to the ambiguities above would be. IFEval-simple data files. The cited representation-engineering papers (Zou et al. 2023; Rimsky et al. 2023).

## Correction after audit (2026-11-27 01:56 PT)

Appended after the auditor's Turn 2 in `discourse/steer.md`; the text above is unchanged. On whether the addition is repeated at every decoding step I wrote that "the likelier reading is every step". The text (p6) supports neither reading; that sentence is a guess and should not be used as evidence. Also: the leave-one-type-out AUROCs reach 0.10-0.13 in places (Table 7, p17), which fits a probe that encodes instruction type times type base rate (Table 10, p18) and strengthens the caution that there is no global instruction-following direction.
