---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M1, Claude Sonnet 5.5)
papers: [karan2025reasoningwithsampling]
reading_level: full text, all 20 pages incl. appendix A.1-A.3 and references, arXiv 2510.14901v1 (16 Oct 2025); PDF extracted with pymupdf and read in full with page markers; equations (1)-(14) were legible in the extraction, TeX source not needed
---

# Karan & Du 2025, "Reasoning with Sampling: Your Base Model is Smarter Than You Think" (arXiv 2510.14901v1)

Page numbers are the PDF pages (`=== page i/20 ===`). Extracted text:
`/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/papers/karan2025reasoningwithsampling/karan2025reasoningwithsampling.txt`.

## What the paper measures

Whether single-shot (pass@1) accuracy of a base model can be lifted to GRPO's level by sampling alone, from the target
distribution p^alpha (p = the base model's sequence distribution), using a Metropolis-Hastings (MH) sampler and no
training, data or verifier (p. 1-2). Benchmarks: MATH500, HumanEval, GPQA-Diamond, AlpacaEval 2.0 (p. 8). Models:
Qwen2.5-Math-7B, Qwen2.5-7B, Phi-3.5-mini-instruct (p. 8). RL baseline: GRPO trained on the MATH train split (p. 8,
Table 1 row "GRPO (MATH)").

## Exact construction

- Target (Sec. 4.1, Eq. 2): p^alpha, alpha in [1, inf). Low-temperature sampling is Eq. 3, tau = 1/alpha.
  Proposition 1 (p. 4): "Low-temperature sampling does not sample from the power distribution p^alpha." The conditional of
  p^alpha weights token x_t by a SUM OF EXPONENTS of future-path probabilities (Eq. 7), low temperature by an EXPONENT OF
  SUMS (Eq. 8). Example 1 (p. 5) is a 2-token toy; Appendix A.1 (p. 16, Prop. 3, Eq. 14) gives a condition under which
  the two pick different tokens. These are existence arguments for a difference; the paper gives no measurement of how
  large the difference is for an LLM.
- Sampler (Sec. 4.2-4.3, Eq. 9, Algorithm 1, p. 6-7): MH with a random-resampling proposal (pick uniform index m, resample
  the suffix from p_prop). Autoregressive blocking: intermediate targets pi_k proportional to p(x_{0:kB})^alpha (Eq. 10-11).
  Per block: initialise with p_prop, then N_MCMC resampling steps with accept ratio Eq. 9 / line 7.
- Hyperparameters (p. 8): T_max = 3072 tokens, B = 3072/16 = 192, N_MCMC = 10 (p. 11), alpha = 4.0, proposal p_prop = the
  base model at temperature 1/alpha. AlpacaEval uses proposal temperature tau = 0.5 (p. 8). HumanEval pass@k plot uses alpha = 1.67
  "(this temperature exhibits slightly better results at earlier k)" (p. 17).
- Cost (Eq. 12, p. 7; p. 11): expected generated tokens E ~ N_MCMC T^2 / (4B); multiplier = N_MCMC T / (4B) over plain
  decoding. With N_MCMC = 10, T = 679, B = 192: 8.84x (p. 11). The multiplier grows linearly in T at fixed B.
- alpha ablation (Fig. 6 left, p. 11): "an intermediate alpha = 4.0 outperforms other values", "relatively stable beyond
  alpha >= 2.0". N_MCMC ablation (Fig. 6 right): accuracy rises to N_MCMC = 10, then flat, "no more than 3-4% between
  N_MCMC = 2 and N_MCMC = 10". Values are read from figures in the paper, not stated as tables.

## The low-temperature baseline (the point that matters for us)

Table 1 (p. 9) has a row "Low-temperature" for every model. The text never states its temperature (grep of the whole
extraction: the row appears only in Table 1). Ji et al. (Table 5, p. 28) state 0.25 = 1/alpha for the same numbers, and
the Qwen rows of Ji's Table 1 are identical to Karan & Du's, so tau = 0.25 is a probable reading, not a stated fact in
this paper. The "Base" row's decoding temperature is also not stated.

Table 1 (accuracy as fraction; Base / Low-temperature / Power sampling / GRPO):

| Model | MATH500 | HumanEval | GPQA | AlpacaEval 2.0 |
|---|---|---|---|---|
| Qwen2.5-Math-7B | .496 / .690 / .748 / .785 | .329 / .512 / .573 / .537 | .278 / .353 / .389 / .399 | 1.61 / 2.09 / 2.88 / 2.38 |
| Qwen2.5-7B | .498 / .628 / .706 / .740 | .329 / .524 / .622 / .561 | .278 / .303 / .318 / .354 | 7.05 / 5.29 / 8.59 / 7.62 |
| Phi-3.5-mini-instruct | .400 / .478 / .508 / .406 | .213 / .585 / .732 / .134 | .273 / .293 / .364 / .359 | 14.82 / 18.15 / 17.65 / 16.74 |

Reading (my arithmetic on their table, no CI exists): on Qwen2.5-Math-7B MATH500 low temperature alone closes
(.690-.496)/(.785-.496) = 67 % of the base-to-GRPO gap and power sampling 87 %. So in this paper low temperature recovers
most of the headline gain on most cells; power sampling adds a further part. The paper does NOT show that low temperature
"alone" is insufficient; it shows that it is distinct in theory (Prop. 1) and a few points lower in practice. Exceptions:
Qwen2.5-7B AlpacaEval 2.0 (low-T 5.29 < base 7.05) and, in Ji et al., an RL-tuned model where low T hurts (see the Ji note).
The Phi-3.5-mini-instruct row is an instruct model, not a base model, and GRPO there lowers HumanEval (.213 -> .134) and
barely moves MATH500 (.400 -> .406) while sampling moves them a lot, so "matches GRPO" there is partly a statement about
a weak GRPO run (p. 8: Phi hyperparameters were hand-picked "that avoid training instabilities").

## Verbatim quotes we might rely on

- "This is the question of distribution sharpening [He et al., 2025, Shao et al., 2025, Yue et al., 2025]: that is,
  whether the posttrained distribution is simply a 'sharper' version of the base model distribution, instead of placing
  mass on reasoning traces the base model is unlikely to generate." (p. 1) The paper treats sharpening as the
  hypothesis; it cites, not tests, it.
- "Then if RL posttrained models really are just sharpened versions of the base model, we should be able to explicitly
  specify a target sampling distribution that achieves the same effect." (p. 4) A sufficiency test, not a proof of
  sharpening: matching RL's accuracy with a sharpened sampler is compatible with, but does not establish, that RL is
  sharpening.
- "A common misconception is that sampling with (3) over T tokens is equivalent to sampling from p^alpha; however, this is
  false in a subtle yet crucial way" (p. 4).
- "Across base models of different families, our sampling algorithm achieves massive, near-universal boosts in single-shot
  accuracies" (p. 8); "on MATH500, which is in-domain for RL-posttraining, power sampling achieves accuracies that are on par with those obtained by GRPO" (p. 8).
- "GRPO samples are heavily concentrated at the highest likelihood peak" (p. 9, Fig. 4: log-likelihood under the base model).
- Length: "Qwen2.5-Math-7B averages a response length of 600 tokens, while GRPO averages 671 tokens. Surprisingly, power
  sampling achieves a similar average length of 679 tokens" (p. 10).
- Diversity: GRPO pass@k "tapers off for large k" (p. 10, Fig. 5); on HumanEval GRPO "exhibits better pass@k than the
  base model until k = 16" (p. 18), which they attribute speculatively to multiple valid code answers.
- Compute: "our method incurs roughly the same inference cost as one epoch of GRPO training, assuming 8 rollouts per sample" (p. 11).

No quote on instruction following or long reasoning traces exists: the extraction contains zero hits for
"instruction-follow", IFEval, IFBench; the longest budget is 3,072 tokens and the mean MATH500 response 679 tokens.

## Ambiguous or unsupported

- No error bars, no confidence intervals, no seed count anywhere (0 hits for "seed", "standard dev", "confidence interval").
  MATH500 has 500 problems, HumanEval 164, GPQA-Diamond 198, so differences of 2-4 points (e.g. the N_MCMC ablation) are
  within sampling noise at those sizes; this is my arithmetic, the paper says nothing.
- Temperatures of the "Base" and "Low-temperature" rows unstated (above).
- "Base" is Qwen2.5-Math-7B and Qwen2.5-7B (base models) but Phi-3.5-mini-instruct (already instruction-tuned). The
  claim "base model is smarter than you think" is thus supported for two base models and one instruct model.
- The RL comparators are GRPO runs trained for this paper on the MATH train split with Shao et al.'s settings, one run per
  model, no seeds. Out-of-domain "wins" over GRPO (HumanEval, AlpacaEval) compare against a model trained on MATH only.
- p^alpha MCMC accuracy depends on N_MCMC and B; B = T_max/16 is tied to T_max = 3072. No result at long T.
- Only single-shot (pass@1) accuracy and pass@k up to 16.
- Fig. 6 and Fig. 4 values are graphical; I could not read exact numbers.

## How our setting differs

- Model: a 32B thinking model (Olmo 3.1 Think 32B line) vs their 3-7B math/code models; our "base" `dpo` is a post-trained
  (SFT + DPO) thinking model, so the "base" is already an instruct-like policy, closest to the Phi-3.5-mini-instruct row,
  not to the base-model rows.
- Length: our responses average 4,400-6,900 tokens (p90 about 10-11k), up to a 32,768 cap; theirs 600-700 tokens, cap 3,072.
  Eq. 12 makes MCMC cost linear in T at fixed B.
- Task: IFBench strict checker (constraint following) vs math/code/MC-QA/AlpacaEval.
- RL: the Olmo RL run used verifiable rewards on math, code, instruction-following and chat (Olmo 3 model card /
  technical report p. 38), many steps with a published sampling setting of T 1.0, top_p 1.0 (technical report p. 47), vs
  one GRPO run on MATH.
- Low temperature behaves differently for thinking models: the Olmo 3 report (footnote 71, p. 114) says "both thinking
  models degenerate quickly when evaluated with low temperatures", so tau = 0.25 is not a safe baseline for us.

## What I did not read / check

- I read the proofs in A.1 but did not verify Prop. 3 beyond following the algebra.
- I did not run or read the authors' code, and did not check Table 1 against their repository.
- The "Website" and "Code" links on page 1 were not followed.
- Figures were read through their captions and axis text only (the extraction has no figure pixels).

## Correction after audit (2026-11-27 02:10 PT, reader-implementer, Claude Sonnet 5.5)

Appended after the auditor's Turn 2 (`discourse/temp.md`); the text above is unchanged. I rendered PDF page 11 as an image
and checked each point.

1. **Fig. 6 values are printed on the bars; I wrote "could not read exact numbers" and that was wrong** (I had only the
   text extraction, which has no bar labels). Left panel, MATH500, alpha = 1.0 / 2.0 / 4.0 / 10.0 / inf: Qwen2.5-Math-7B
   45.0 / 70.0 / 74.8 / 72.0 / 71.0; Qwen2.5-7B 35.8 / 69.4 / 70.6 / 69.2 / 69.0; Phi-3.5-mini-instruct 38.4 / 48.2 / 50.8 /
   48.2 / 48.4. alpha = 1.0 is plain base sampling, but Table 1 "Base" MATH500 is 49.6 / 49.8 / 40.0, so the same sampler
   gives 4.6 / 14.0 / 1.6 points different results in two places of the same paper, unexplained. This is in-paper evidence
   that run-to-run or set-up variation is as large as several of the claimed effects.
2. **Low-temperature tau.** Right panel (N_MCMC = 0 points): about 69.0 (Qwen2.5-Math-7B) and about 62.8 (Qwen2.5-7B), equal to
   the Table 1 low-temperature MATH500 values .690 and .628. With zero MH steps the sampler is the proposal alone (base model
   at 1/alpha, p. 8), so tau = 0.25 is supported inside this paper too, not only by Ji p. 28.
3. **Table 1 Base HumanEval (.329) and GPQA (.278) are identical for Qwen2.5-Math-7B and Qwen2.5-7B.** My table reproduced
   them without comment; likely a copy error (Ji reuses them).
4. **GRPO provenance.** I wrote "GRPO runs trained for this paper". Page 8 says only "we use the implementation of GRPO in
   Shao et al. [2025], which posttrains these models on the MATH training split"; Ji p. 28 says the checkpoint is Shao et
   al.'s released one. Neither settles who trained it.
5. My arithmetic "low-T closes 67 %" stands, but note the Base row itself is uncertain by the point in 1.
