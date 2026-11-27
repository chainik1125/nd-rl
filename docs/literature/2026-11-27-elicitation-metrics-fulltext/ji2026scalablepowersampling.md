---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M1, Claude Sonnet 5.5)
papers: [ji2026scalablepowersampling]
reading_level: full text, all 33 pages incl. appendices A-C and references, arXiv 2601.21590v2 (12 Oct 2026, ICML 2026 proceedings header); PDF extracted with pymupdf and read in full with page markers; the long bias/jackknife proofs (pp. 14-22) were read but not independently re-derived; TeX source not needed (equations legible)
---

# Ji, Tutunov, Zimmer, Bou Ammar 2026, "Scalable Power Sampling" (arXiv 2601.21590v2)

Page numbers are PDF pages (`=== page i/33 ===`). Extracted text:
`/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/papers/ji2026scalablepowersampling/ji2026scalablepowersampling.txt`.

## What the paper measures

Whether the power distribution p^alpha of Karan & Du can be sampled without MCMC: a token-level lookahead estimator
("scalable power sampling") compared with standard decoding, low-temperature, Best-of-N, self-consistency, MCMC power
sampling and GRPO(MATH), on MATH500, HumanEval, GPQA-Diamond (p. 6, Table 1). Models: Qwen2.5-7B, Qwen2.5-Math-7B,
DeepSeek-Math-7B, the RL-tuned DeepSeek-Math-7B-RL (p. 6-7), and Qwen2.5-32B (p. 8, Table 3).

## Exact construction

- Notation (Eq. 1, p. 2): low-temperature p_alpha^(low.temp)(x_t | ...) proportional to p^alpha(x_t | ...); power
  p_alpha^(pow)(x_t | ...) = sum over futures of p^alpha(full sequence) / normaliser.
- Theorem 3.1 (Eq. 2, p. 3; proof App. A.1, p. 13): p^(pow)(x_t | .) = p^alpha(x_t | .) zeta_t(x_t) / sum_{x'} p^alpha(x' | .)
  zeta_t(x'), with zeta_t(x') = sum_{x_{t+1:T}} p^alpha(x_{t+1:T} | q, x_{0:t-1}, x'). So power = low temperature times a
  per-token "future quality" factor; they coincide on the last token (p. 3). This is an exact identity; the approximation
  enters only in estimating zeta.
- Estimator (Eq. 3-4, p. 3): zeta_hat = mean over M_t rollouts of p^(alpha-1)(rollout | prefix, x'), sampled from the base model; ratio
  estimator biased O(1/M_t) (Lemma 3.2, p. 4); jackknife correction (Eq. 5-6) makes it O(1/M_t^2) (Lemma 3.3).
- Algorithm 1 (p. 5): per step, top-K_t candidates, M_t rollouts each; practical Algorithm 2 (App. B, p. 22-23) works in
  blocks of B tokens, L_t candidate blocks, top-K_t, M_t rollouts of lookahead H_t tokens. Truncation to top-K_t and
  finite H_t add bias that they do not quantify (App. B.1, p. 24-25: "difficult to quantify without imposing further
  assumptions").
- Hyperparameters (p. 6, Table 5 p. 28): alpha = 4, B = 192, K_t = M_t = 8, H_t = B = 192 (p. 33), T_max = 3072, seed 0
  "for all main results" (p. 27-28); low-temperature baseline "temperature = 0.25 = 1/alpha"; Best-of-N N=32 by base
  log-likelihood; self-consistency 16 samples; MCMC power sampling as in Karan & Du (N_MCMC = 10).
- Implementation: vLLM v0.6.3 on one GPU (p. 27). Cost: "typically about 2.5-3.5x slower than standard decoding ... but remains up to 10x faster than MCMC" (p. 28); wall clock per prompt: MCMC 2.5 min vs 0.22 min on
  Qwen2.5-Math-7B MATH500 (p. 8); Qwen2.5-32B: 0.36 vs 4.1 min per prompt (p. 8).

## What they compare, and the results that matter to us

Table 1 (p. 6), pass@1 fractions (MATH500 / HumanEval / GPQA):

| Model | Base | Low-temp (0.25) | MCMC | Ours | GRPO(MATH) |
|---|---|---|---|---|---|
| Qwen2.5-7B | .498/.329/.278 | .628/.524/.303 | .706/.622/.318 | .708/.756/.349 | .740/.561/.354 |
| Qwen2.5-Math-7B | .496/.329/.278 | .690/.512/.353 | .748/.573/.389 | .758/.604/.409 | .785/.537/.399 |
| DeepSeek-Math-7B | .362/.415/.333 | .366/.427/.430 | .424/.470/.345 | .464/.487/.364 | .492/.524/.333 |

The Qwen2.5-7B and Qwen2.5-Math-7B rows for Base, Low-temperature, MCMC and GRPO are numerically identical to Karan & Du's
Table 1; I read this as re-use of their numbers, not a re-run (the paper does not say). The new rows are Best-of-N,
self-consistency and "Ours".

- Table 2 (p. 7), DeepSeek-Math-7B-RL (already GRPO'd): Base .492/.524/.333; Low-temperature .412/.524/.303. Low
  temperature LOWERS MATH500 (-8.0 points) and GPQA on an RL-trained model; power sampling .494/.530/.349, "Ours"
  .502/.549/.364. Quote: "GRPO post-training has already sharpened the token distribution and reduced answer
  diversity, making local temperature scaling and answer-level voting less effective." (p. 7)
- Table 3 (p. 8), Qwen2.5-32B (base): Base .577/.392/.376; Low-temp .682/.507/.414; MCMC .738/.567/.383; Ours
  .748/.793/.429. This is the only 32B result in either paper and is a base model with no GRPO comparator.
- Diversity (p. 8): duplicate-answer rate over 8 rollouts on 100 MATH500 prompts: 58.3 % low-temperature, 62.5 % ours,
  83.3 % GRPO.
- Ablations (p. 8, Fig. 4; p. 29, Fig. 6): robust to K_t, M_t; sensitive to alpha, best at 4-5, alpha = 8 degrades; "alpha = 1
  performs poorly on most tasks" with an exception on GPQA DeepSeek-Math-7B where "low-temperature decoding can be particularly strong". Jackknife ablation:
  average drop 4.29 % without it (p. 33, Table 9).

## Verbatim quotes we might rely on

- Abstract (p. 1): "growing evidence suggests that its gains arise primarily from distribution sharpening rather than the
  acquisition of new capabilities." Cited, not tested, in this paper; the intro cites He, Song, Yue, Gai, Ni et al. (p. 1).
- "Under this view, the reasoning trajectories required to solve complex problems already exist within the base model; RL
  primarily reshapes the probability mass." (p. 1)
- "Overall, our results reproduce the conclusions of Karan & Du (2025) regarding the effectiveness of MCMC-based power
  sampling, suggesting that a substantial fraction of the gains attributed to reward-based post-training can be recovered
  through principled sampling from the base model alone." (p. 7) The word "reproduce" is applied to numbers that appear copied, see above.
- "Relative to standard decoding, low-temperature sampling generally improves performance, but the gains are modest in
  some settings, suggesting that purely local token-level sharpening has limited ability to extract higher-quality
  trajectories across tasks." (p. 6) Compare their own Table 1, where low T supplies +13 to +19 MATH500 points on the Qwen models.
- "We demonstrate that latent reasoning capabilities in base models are accessible via distribution sharpening." (p. 9)
- Abstract: "matches or surpasses one-shot GRPO without relying on any external rewards" while the body compares
  to GRPO trained on the MATH training split (p. 6, Table 1 row "GRPO (MATH)"; Appendix C.1 p. 28 checkpoints from Shao et al.). I could not tell what "one-shot" means (single-shot evaluation, or one-example RL); the checkpoints cited are full MATH-split GRPO.
- "All experiments are run on the same single GPU using vLLM v0.6.3" (p. 27); max generation "Tmax = 3072 tokens" (p. 6).

Nothing on instruction following (0 hits for "instruction-follow", IFEval, IFBench) or on responses over 3,072 tokens.

## Ambiguous or unsupported

- Single seed (0) for all main results, no error bars, no CIs (0 hits). Benchmarks of 164-500 prompts, so differences of
  1-2 points (Ours vs MCMC on MATH500 .708 vs .706) are noise.
- The abstract's "over 10x" latency saving: measured per prompt on their implementation at T_max 3072; the 2.5-3.5x slowdown
  over plain decoding is stated without a table (p. 28).
- "Reproduce Karan & Du" is not a re-run (identical numbers).
- Bias bounds are worst-case with exponent |V|^{(T-t+1)(alpha-1)} (Appendix A.5, p. 19-22), explicitly "quite loose"; Lemma 3.2/3.3 give rates, not practical error. Truncation (top-K_t) and lookahead (H_t) bias are unquantified (p. 24).
- The key hyperparameter K_t = M_t = 8 with H_t = 192 is tuned on Qwen2.5-Math-7B (Fig. 4); the "validation subset" for other models is not described (p. 29).
- Which models are base: Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B, Qwen2.5-32B are pretrained models; Qwen2.5-Math-7B is already math-continued-pretrained; DeepSeek-Math-7B-RL is RL-tuned (Table 2).

## How our setting differs

- 32B thinking model with 4-7k-token traces (cap 32,768) vs 7B base and 3,072-token cap; lookahead rollouts of
  H_t = 192 tokens every B = 192 are a small slice of a 5,000-token trace, and the cost formula sum(L_t B + K_t M_t H_t)
  (p. 23) scales linearly with the number of blocks, so we would have about 5,000/192 = 26 blocks per response.
- IFBench strict checker, a constraint-following score determined by a few specific tokens, vs accuracy of a final answer.
- Our `dpo` is already post-trained; Table 2 is the nearest analogue (low T hurt an RL-tuned model) and `s2300` is an RL-trained model.
- Official Olmo evaluation uses T 0.6, top_p 0.95 (technical report p. 37, 114) and warns thinking models degenerate at low T (p. 114, fn 71); their low-T baseline is tau = 0.25 (our setting would loop).
- Ji et al.'s Algorithm 2 samples candidate blocks and lookahead rollouts "from the base model" p(. | ...) (p. 22-23; the sampling temperature of these draws is not stated) and keeps the top-K_t by likelihood; Karan & Du's proposal is the base model at T = 1/alpha. Whether the two use the same proposal temperature is unclear from the text.

## What I did not read / check

- I did not re-derive Lemma 3.2/3.3 or the A.5 constants; I followed them but did not verify.
- I did not read their GitHub code or check Table 1 against it.
- Figures 1-6 were read through captions and tick labels only.

## Correction after audit (2026-11-27 02:10 PT, reader-implementer, Claude Sonnet 5.5)

Appended after the auditor's Turn 2; the text above is unchanged.

1. **Cost.** I wrote the token formula (p. 23) but treated the "2.5-3.5x slower than standard decoding" (p. 28) as the cost.
   p. 28 says "We report wall-clock generation time per prompt", and p. 5 says "the total inference burden is bounded by
   O(sum K_t M_t) generated sequences". Per block of B output tokens, Algorithm 2 generates L_t*B candidate tokens plus
   K_t*M_t*H_t rollout tokens (p. 23). With K_t = M_t = 8 and H_t = B = 192 that is at least 64 extra blocks of tokens
   per output block, so at least about 65x the tokens of plain decoding (L_t is never given, so more). The 3x is a
   single-prompt latency on a lightly loaded GPU. On a throughput-bound pod we pay tokens. My Turn 1 line "Ji-style at 3x
   about $6.5" was wrong.
2. **DeepSeek-Math-7B row (Table 1, p. 6).** I tabulated it without remark. Low temperature .362 to .366 on MATH500 closes
   (.366-.362)/(.492-.362) = 3 % of the GRPO gap, while "Ours" (.464) closes 78 % and MCMC (.424) 48 %. On GPQA low T
   (.430) beats Ours (.364) and GRPO (.333). So the papers' own data contain a case where low temperature does nothing and
   power sampling does most of the work, and others where low T alone does the work. "Power sampling is only interesting
   where low T gives partial recovery" (Turn 1 section 8) is not supported by the papers.
3. **Algorithm 2 weighting (auditor's point, my reading).** Candidate blocks are drawn from p, cut to the top-K_t by
   likelihood, then selected with weights p^alpha * zeta_hat (eqs. 19-21, p. 23) with no division by the proposal
   probability. Selection among candidates that are themselves drawn from p gives roughly p^(alpha+1) * zeta_hat in the
   sampling-importance-resampling reading. I follow the logic but did not verify it numerically; Theorem 3.1 and the lemmas
   concern the token-level Algorithm 1. It matters only for a follow-up.
4. I said Ji's low-T baseline needs "tau = 0.25 would loop for us" as fact; that is an inference from the Olmo report
   (footnote 71), not something either paper tests.
