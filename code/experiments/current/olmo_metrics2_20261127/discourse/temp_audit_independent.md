---
written_on: 2026-11-27
written_by: agent:claude (auditor for M1 `temp`, Claude Opus 5.5)
papers: [karan2025reasoningwithsampling, ji2026scalablepowersampling]
reading_level: "Karan & Du: full text, all 20 pages incl. appendix, arXiv 2510.14901v1 (the only version). Ji et al.: full text, all 33 pages incl. appendices A-C, arXiv 2601.21590v2 (12 Oct 2026, latest). Both model cards and generation_config.json files in full. Olmo 3 report (118 pp.): decoding passages only, pages listed under Sources."
---

## Auditor (Claude Opus 5.5), independent read, 2026-11-27 01:51 PST

Independence: before writing this I opened none of `docs/literature/2026-11-27-elicitation-metrics-fulltext/`,
the code folder's `discourse/` files (other than writing this one), `tools/temp_baseline_inventory.py`, or
`docs/literature/2026-11-25-elicitation-metrics/`.

### Sources and what I read

- `karan2025reasoningwithsampling`: PDF md5 `c9a13a6c73f1981fe37d90b57d631f98`, extracted to
  `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/karan2025reasoningwithsampling.txt`, pages 1-20 read. arXiv
  lists only v1 (16 Oct 2025). Figures 4 and 6 (pp. 9, 11) also read from rendered page images
  (`audit/png/karan_p9.png`, `karan_p11.png`) because the extracted text has no bar values.
- `ji2026scalablepowersampling`: PDF md5 `3ced72d1eb013fba38cd86161b22140f`, extracted to `audit/ji2026scalablepowersampling.txt`,
  pages 1-33 read. v2 (12 Oct 2026) is the latest; it carries the ICML 2026 header. Figures 2-4 (pp. 7-8) read from
  rendered images.
- Model cards: `modelcards/think31/README.md`, `modelcards/dpo/README.md` and both `generation_config.json`, read in full.
- Olmo 3 report (`modelcards/tr/olmo3.txt`, 118 pp.): I read only the decoding passages, on pp. 37 (evaluation settings
  and variance), 40 (Table 16), 47 (offline filtering, "matching how we sample during RL training"), 52 (Table 24, IF
  results), 64 (RL-Zero evaluation), 108 (repetition filter), 109 (Table 49, RL hyperparameters) and 114 (A.8.1 and
  footnote 71). I saw single grep lines on pp. 57, 97 and 115 (tool-use evaluation at T 0, base-model evaluation sweep,
  safety evaluation at T 0.7), which are not relevant here.
- To check sizes and statistical power, I also read Dan's stored grade rows (read only):
  `ext/nd-takehome/artifacts/olmo_ifcurve/rows/{F_dpo_ifb,F_s0100_ifb,F_s2300_ifb,M_s0500}.csv`, plus
  `olmo_ifcurve/lengths.csv` and the docstring of `olmo_ifladder/sample.py`. The script is
  `/tmp/claude-10001/.../scratchpad/pvar.py`, which is scratch only and not kept.

Equation note: the extracted text garbles the display equations (Karan eqs. 3-9 and 12, Ji eqs. 2-6 and 17-22), but
the symbols can still be followed. I checked them against the page images and did not need the TeX source.

---

### Q1. Construction: what is sampled, with which hyperparameters, against which baselines

**Target distribution (both papers): the sequence-level power distribution p^α.** Karan p. 4, §4.1, eq. 2: "One
natural way to sharpen a distribution p is to sample from the power distribution pα." Low-temperature sampling is
eq. 3, "ptemp(xt|x0 . . . xt−1) = p(xt|…)^α / Σ p(x′t|…)^α … where the temperature is τ = 1/α". Proposition 1
(p. 4) says that "Low-temperature sampling does not sample from the power distribution pα". The difference is a sum
of exponents (eq. 7, p^α) against an exponent of sums (eq. 8, low temperature). Observation 1 (p. 5): "The power
distribution upweights tokens with few but high likelihood future paths, while low-temperature sampling upweights
tokens with several but low likelihood completions." Appendix A.1 (p. 16, Proposition 3) formalises this with
"positive/negative pivotal tokens".

**Karan's sampler (MCMC).** It is Metropolis-Hastings (eq. 9, p. 6) with a "random resampling" proposal: "select
a random t ∈[1, T] and resample the sequence starting at index t using pprop" (p. 6). Algorithm 1 (p. 7) applies
this block by block to intermediate targets πk ∝ p(x0:kB)^α (eqs. 10-11). For each block it extends by B tokens with
pprop, then runs N_MCMC steps. Each step picks m uniformly in {1,…,(k+1)B}, resamples from m to (k+1)B, and accepts
with min{1, πk(x′)/πk(x) · pprop(x|x′)/pprop(x′|x)}. Expected cost (eq. 12): E_tokens ≈ N_MCMC·T²/(4B), which is
N_MCMC·T/(4B) times plain decoding.

Hyperparameters, p. 8 §5.1: "Tmax = 3072 … block size B = 3072/16 = 192 … α = 4.0 coupled with a proposal LLM pprop
chosen as the base model with sampling temperature 1/α". For AlpacaEval the proposal is τ = 0.5. N_MCMC = 10 comes
from p. 11 ("Plugging in our experimental parameters NMCMC = 10, T = 679 … B = 192 … 8.84×"). In the pass@k appendix
(A.2, p. 17) the exponent is "α = 1.67 for HumanEval (this temperature exhibits slightly better results at earlier
k)". α = 1.67 is the power analogue of T = 0.6, and the paper itself slips into calling α a "temperature".

**Karan's baselines** (Table 1, p. 9): "Base", "Low-temperature" and "GRPO (MATH)".
- *The low-temperature baseline's temperature is never stated in Karan.* My inference that it is τ = 0.25: in Fig. 6
  right (p. 11), the N_MCMC = 0 points are about 69.0 (Qwen2.5-Math-7B) and about 62.8 (Qwen2.5-7B). These equal the
  Table 1 low-temperature MATH500 values 0.690 and 0.628. With zero MH steps the sampler is just the proposal, the base
  model at τ = 1/α = 0.25. Ji's Table 5 (p. 28) also states "Low-temperature: temperature = 0.25 = 1/α", and Ji's
  Table 1 shows the same low-temperature numbers.
- *Neither paper states the decoding of the "Base" row* (temperature and top_p are not given). Neither paper uses
  top_p anywhere.
- GRPO: "we use the implementation of GRPO in Shao et al. [2025], which posttrains these models on the MATH training
  split" (p. 8).

**Ji's approximation.** Theorem 3.1 (p. 3, eq. 2) writes p_pow(xt|prefix) as proportional to p(xt|prefix)^α·ζt(xt),
where ζt(x′) = Σ_future p(future|prefix,x′)^α. That is, "the power distribution can be expressed as a scaled version
of the low-temperature distribution". ζ is estimated by Monte Carlo over base-model futures, ζ̂ = (1/M)Σ p^{α−1}(future_r)
(eq. 3). The plug-in ratio estimator (eq. 4) has O(1/M) bias (Lemma 3.2). A jackknife (eqs. 5-6) reduces that to
O(1/M²) (Lemma 3.3).

The algorithm actually run is the batched Algorithm 2 (p. 23; App. C.1, p. 27: "Our implementation follows the batched
scalable power sampling method in Algorithm 2"). It works as follows:
- Sample L_t candidate blocks of B tokens from p.
- Keep the top K_t by base likelihood.
- Run M_t rollouts of H_t tokens from each kept block.
- Weight the blocks by p^α(block)·ζ̂ with the jackknife (eqs. 17-21), and sample one.
- Choose the last block by low temperature among the top-K (eq. 22).

Hyperparameters (Table 5, p. 28; p. 6; App. C.3, p. 33): α = 4, Tmax = 3072, B = 192, H_t = 192, K_t = M_t = 8, seed 0,
vLLM v0.6.3. **L_t is never given.**

**Ji's baselines** (pp. 5-6, Table 5):
- "Base" (standard decoding, settings unstated).
- Low temperature at 0.25.
- Best-of-N with N = 32, ranked by base log p(x|q).
- Self-Consistency over 16 samples by majority vote. How this is done for HumanEval code is undefined.
- MCMC power sampling with Karan's settings (α = 4, B = 192, N_MCMC = 10, proposal at τ = 0.25).
- GRPO (MATH) checkpoints: the Qwen2.5-Math-7B "…lr5e-7-kl0.00-step150" checkpoint from Shao et al. (footnote 2,
  p. 28), and DeepSeek-Math-7B-RL.

**My objection to Algorithm 2 (my interpretation, not stated in the paper).** The candidate blocks are drawn from p
and then weighted by p^α·ζ̂ with no division by the proposal probability, after a top-K-by-likelihood cut. For
192-token blocks every candidate is distinct. By the usual sampling-importance-resampling argument, the block that is
picked is then distributed roughly as p^{α+1}·ζ̂ restricted to the top-K of the L_t draws, not as p^α. Theorem 3.1
and Lemmas 3.2-3.3 concern Algorithm 1 over the full vocabulary. Ji acknowledges the extra bias from top-K and from
H_t (App. B.1, pp. 24-25) but not this weighting issue. So the "Power Sampling (ours)" rows come from a heuristic
sharpening sampler; they are not a consistent estimate of p^α.

---

### Q2. Evidence

**Models.**
- Karan: Qwen2.5-Math-7B and Qwen2.5-7B (pretrained bases), and Phi-3.5-mini-instruct (a post-trained 3.8B model), with
  GRPO-on-MATH versions of each (p. 8).
- Ji: Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B and DeepSeek-Math-7B-RL (p. 6), plus Qwen2.5-32B (Table 3, p. 8,
  no GRPO row).

Nothing in either paper is a thinking model, a long-chain-of-thought model, or a model that went through
SFT, DPO and then RL.

**Benchmarks.**
- MATH500 (500 problems), HumanEval (164), GPQA Diamond (198).
- Karan only: AlpacaEval 2.0 (805 prompts, length-controlled win rate judged by GPT-4-turbo).
- Tmax is 3072 tokens throughout. Mean lengths are about 600-700 tokens: Karan p. 10 ("Qwen2.5-Math-7B averages …
  600 tokens, while GRPO averages 671 … power sampling … 679"); Ji p. 8 (640 / 682 / 679 / 667).

**Samples, seeds, error bars.**
- Table 1 in both papers is "single-shot" (Karan p. 8: "We evaluate all of our methods and baselines single-shot";
  Ji: pass@1).
- pass@k is shown up to k = 16, for Qwen2.5-Math-7B only (Karan Figs. 5, 7-9; Ji Figs. 3, 5).
- Karan gives no seeds. Ji: "we set the random seed to 0 for all main results" (p. 27).
- **Neither paper has an error bar, interval or repeat anywhere.**
- Binomial standard errors I computed: about 2 points on MATH500, about 3.4 on GPQA (198 questions), about 3.9 on
  HumanEval (164). Several "outperforms" claims sit within one SE. Examples: Karan GPQA, Qwen-Math, power sampling
  0.389 vs GRPO 0.399; Ji Table 2, MCMC vs Ji's method on DeepSeek-RL, 0.494 vs 0.502.

**Headline claims and the tables or figures behind them.**

| Claim | Support |
|---|---|
| Karan: power sampling "nearly match[es]" GRPO on in-domain MATH500 | Table 1, p. 9 |
| Karan: beats GRPO out of domain | Table 1 (HumanEval, AlpacaEval) |
| Karan: diversity kept | Figs. 5, 7-9 |
| Karan: α robustness | Fig. 6 left |
| Karan: N_MCMC scaling | Fig. 6 right |
| Karan: likelihoods and confidences | Fig. 4 |
| Ji: matches or beats MCMC and GRPO | Table 1, p. 6 |
| Ji: power sampling helps on the RL model | Table 2, p. 7 |
| Ji: works at 32B | Table 3, p. 8 |
| Ji: "over 10×" latency gain | Fig. 2, p. 7 |
| Ji: α / K / M ablation | Figs. 4 and 6 |
| Ji: H_t / B ablation | Table 8, p. 33 |
| Ji: jackknife ablation (−4.29 points on average) | Table 9; I checked the arithmetic and it is right |

Ji's "over 10×" is the maximum ratio, not a typical one: 2.52/0.24 and 2.50/0.22 min/prompt on MATH. On DeepSeek GPQA
it is 0.63/0.25 = 2.5×.

**What the papers' own numbers say about plain low temperature.** I computed these from the tables. Fraction of
GRPO's in-domain MATH500 gain recovered by low temperature at τ = 0.25:

| Model | Base → low T → power → GRPO | Low T recovers | Power recovers |
|---|---|---|---|
| Qwen2.5-Math-7B | 49.6 → 69.0 → 74.8 → 78.5 | (69.0 − 49.6)/(78.5 − 49.6) = **67 %** | 87 % |
| Qwen2.5-7B | 49.8 → 62.8 → 70.6 → 74.0 | **54 %** | 86 % |
| DeepSeek-Math-7B (Ji) | 36.2 → 36.6 → MCMC 42.4 / Ji 46.4 → 49.2 | **3 %** | 78 % (Ji's method) |

- Phi-3.5-mini-instruct: GRPO gains only 0.6 points (40.0 → 40.6), so no ratio is meaningful. Low temperature alone
  gives +7.8 on MATH500 and +37.2 on HumanEval for this *post-trained* model.
- Qwen2.5-32B: low temperature +10.5 on MATH500 (57.7 → 68.2).
- Low temperature on the **RL model** (Ji Table 2, DeepSeek-Math-7B-RL): MATH500 0.492 → 0.412, GPQA 0.333 → 0.303,
  HumanEval unchanged at 0.524. Ji, p. 7: "low-temperature sampling degrades performance on MATH500 and GPQA".

**Claims that no table or figure supports.**
- **"RL's gains are sharpening."** Karan treats it as a question ("This is the question of distribution sharpening",
  p. 1) and a premise ("if RL posttrained models really are just sharpened versions of the base model, we should be
  able to…", p. 4). The only in-paper evidence is Fig. 4 (GRPO samples sit at high base likelihood and confidence).
  On p. 10 it then asserts "the observation that RL-posttraining strongly sharpens the base model distribution at the
  expense of diversity [Song et al., 2025]", backed by a citation and Fig. 5.
- Ji's abstract: "growing evidence suggests that its gains arise primarily from distribution sharpening". Citations
  only.
- Ji p. 1: "These findings strongly support the hypothesis that the reasoning capabilities attributed to RL are
  largely latent within the base model". This is an interpretation of Karan's numbers.
- Ji p. 7: Table 2 is "consistent with the view that GRPO already performs a form of distribution sharpening". It is
  consistent with that view, but it is not a test of it.
- Neither paper compares the RL policy with p^α directly, for example by KL divergence, by the likelihood of GRPO
  samples under p^α, or by overlap of samples.
- Karan p. 11: power sampling "incurs roughly the same inference cost as one epoch of GRPO training". This is
  arithmetic, not a measurement.
- Ji p. 8: duplicate-answer rates of 58.3 / 62.5 / 83.3 %. No table, 100 examples.
- Ji p. 7: "our results reproduce the conclusions of Karan & Du". Ji's Table 1 Qwen rows (Base, Low-temperature, MCMC,
  GRPO) are digit-for-digit identical to Karan's Table 1. They were reused, not reproduced, or at least no
  independent rerun is reported. The new data are the DeepSeek rows and Qwen2.5-32B.

**Instruction following: nothing in either paper.** The nearest is Karan's AlpacaEval (general helpfulness judged by
an LLM). There, low temperature *lowered* Qwen2.5-7B's score (7.05 → 5.29) and raised Qwen-Math (1.61 → 2.09) and Phi
(14.82 → 18.15).

**Long reasoning traces: nothing.** Tmax is 3072 and mean lengths are about 650 tokens. MCMC acceptance rates and
mixing are never reported.

**Model card and report (decoding).**
- Both cards: "We evaluated our models on the following settings. We also recommend using them for generation: …
  temperature: 0.6, top_p: 0.95, max_tokens: 32768". Both `generation_config.json` files say T 0.6, top_p 0.95,
  max_new_tokens 32768.
- The DPO config lacks `"do_sample": true`, which the 3.1 Think config has. My understanding of HF `generate()` (not
  checked here) is that it would then decode greedily. This does not matter for vLLM.
- The cards' first code snippet uses `do_sample=True, top_k=50, top_p=0.95` at the default T, which is inconsistent
  with their own recommendation.
- Report p. 37: "use a 32K max context length, a sampling temperature of 0.6 and top-p of 0.95". Table 16 (p. 40)
  gives 0.6 / 0.95 for every task.
- Report p. 114, footnote 71: **"We find that both thinking models degenerate quickly when evaluated with low
  temperatures (as used in OLMo 2), while instruction models can be evaluated at this higher temperature."**
- RL rollouts:
  - Table 49, p. 109: "Sampling temperature 1.0" for 32B Think RL (response length 32,768, group size 8).
  - p. 47: "We sample with a temperature of 1.0 and top-p of 1.0, matching how we sample during RL training".
  - p. 64: RL-Zero is evaluated "with … temperature 1.0 to encourage diversity as we report pass@k".
- Report p. 52, Table 24 (also in both cards): IFBench, 32B Think DPO **34.4**, 3.1 Think **68.1**, at T 0.6 /
  top_p 0.95. The metric (strict or loose) is not stated near the table.
  - For comparison, Dan's T 1.0 draws over all 300 IFBench prompts (my computation): dpo strict 28.0 / loose 33.8,
    s2300 strict 67.6 / loose 69.6.
  - So the developers' own card is close to being a dpo-at-T-0.6 cell already. If their metric is strict, T 0.6 /
    top_p 0.95 adds about +6 points at dpo, about 16 % of the 39.6-point gap. If it is loose, it adds about +1. In
    both cases it adds about 0 at the final model. Harness, system prompt and metric differ, so this is a prior, not
    evidence.

---

### Q3. What a faithful test on our checkpoints looks like

**Mapping the papers' design onto ours.** Their design holds the pre-RL model and prompts fixed and changes only the
sampler. It compares the result with the RL model at its standard decoding, using single-shot accuracy, and includes
a plain low-temperature baseline. Ji also runs the RL model under each sampler.

Our faithful analogue:
1. dpo at T ∈ {lower values}, top_p 1.0. Pure per-token temperature is exactly Karan eq. 3 with α = 1/T. The papers
   use no nucleus truncation.
2. The full response (thinking and answer) sampled at that T, as the papers temper the whole output.
3. Same template, prompts, max tokens (32,768 plays the role of Tmax), and checker.
4. Target: s2300 at T 1.0 / top_p 1.0, its standard and RL-native setting, already stored.
5. Ji's Table 2 analogue: s2300 at the same lowered T.
6. Quantities to report:
   - R(T) = [pass@1(dpo,T) − pass@1(dpo,1.0)] / [pass@1(s2300,1.0) − pass@1(dpo,1.0)], with a paired prompt
     bootstrap. This is the "recovered fraction", matching the papers' comparison.
   - G(T) = pass@1(s2300,T) − pass@1(dpo,T), the decoding-matched gap.

   If s2300 gains as much as dpo from lower T, the dpo gain is a generic decoding effect, not "recovering RL's gain".

**Departures that cannot be avoided, or should not be avoided.**
1. **Temperature.** The papers' low temperature is τ = 0.25. Footnote 71 of the Olmo report says thinking models
   "degenerate quickly" at low temperature, and our responses can run to 32,768 tokens. T 0.6 and 0.8 are α = 1.67 and
   1.25, much milder than α = 4. The papers have *no* low-temperature data at these τ: Karan Fig. 6 sweeps the power
   exponent at α ∈ {1, 2, 4, 10, ∞} only.
2. **No power sampling** (not approved). So M1 tests what the papers call the *weak* baseline. Ji p. 6: "purely local
   token-level sharpening has limited ability to extract higher-quality trajectories".
3. **dpo is not a base model.** It went through SFT and then DPO, and DPO is itself a likelihood-ratio sharpening
   toward preferred responses. The papers' closest case is Phi-3.5-mini-instruct.
4. **Traces are 7-30× longer.** Mean 4,448 tokens at dpo on arm A, p90 9,765, p99 19,618 (Dan's rows), against 3,072
   at most. Per-token sharpening compounds over thousands of tokens.
5. **Task and checker.** IFBench strict checker instead of maths answer match. IF prompts were in Olmo's RL data ("IF-RLVR …
   sampled from IFEval … and IFBench-Train", report p. 47). So IFBench is roughly in-domain for this RL, as MATH500 is
   for GRPO(MATH).
6. **The RL being "recovered" is different.** Here it is 2,300 steps of multi-domain RLVR at T 1.0. In the papers it
   is GRPO on the MATH split (Ji names a step-150, lr 5e-7 checkpoint).
7. **Sampling and statistics.** 2 draws per arm-A prompt, compared with stored draws (16 at dpo), instead of 1
   sample per question. This is harmless with per-prompt means and a paired bootstrap; the papers have no intervals
   at all.

---

### Q4. Weakest points that matter for our use

**Karan**
1. No error bars, seeds or repeats, and one sample per question.
2. **Internal inconsistency.** The α = 1.0 bars in Fig. 6 left (MATH500: 45.0, 35.8, 38.4) should equal base sampling,
   but Table 1 Base is 49.6, 49.8 and 40.0. That is a 14-point gap for Qwen2.5-7B, far beyond the about-2-point
   binomial SE, and it is unexplained. Either run-to-run variation is very large or the setups differ.
3. Table 1 gives identical Base HumanEval/GPQA values (0.329 / 0.278) for two different models, which looks like a
   copy error.
4. The low-temperature τ and the Base decoding are not stated.
5. The comparator RL is short, maths-only GRPO on 7B models.
6. The "sharpening" premise is assumed, not tested.
7. MCMC is untested above 3,072 tokens. Mixing time is discussed only in theory (p. 6).

**Ji**
1. The headline Qwen rows are reused from Karan.
2. No error bars, seed 0 only.
3. Algorithm 2 does not target p^α (Q1), and L_t is unspecified.
4. **H_t = 192 lookahead.** For our traces, the IF verdict is in the final answer, 2-4 thousand tokens after early
   decisions, far outside the rollout horizon. So the "planning" correction cannot see the outcome it is meant to plan
   for.
5. The bound constants grow as |V|^{(T−t+1)(α−1)} (App. A.5), which the paper itself calls loose.
6. **Cost is reported as single-prompt latency, not tokens.** Ji's own complexity (p. 23) is
   O(Σ[L_t·B + K_t·M_t·H_t] + L_w·B) tokens, which is at least 64× plain decoding per block at K = M = 8, H = B
   (plus L_t). Throughput-bound serving, as in our runs, pays the token count, not the latency. "about 2.5–3.5×
   slower than standard decoding" (p. 28) does not transfer to our setting.

**Both, for M1**
1. No IF, no long traces, and no data on mild temperatures.
2. A plausible mechanism points the *other way* for IF (my hypothesis, not in either paper). Many constraints (no
   commas, an exact word count, all lowercase) require choices *against* the model's default high-likelihood style.
   Sharpening toward dpo's modes could lower compliance on those constraint families while raising it where failures
   are random slips.
3. Ji's own data break the inference "low temperature fails, so power sampling fails". On DeepSeek-Math-7B MATH500,
   low temperature gives +0.4 while power sampling gives +10.2.

---

### Q5. Predictions

**What the papers imply.**
- *dpo:* positive effect from lower T if dpo behaves like a base model (+0.4 to +19.4 MATH500 points at τ = 0.25). The
  effect should shrink as RL proceeds and turn neutral or negative at the RL'd model (Ji Table 2: −8 MATH500, −3 GPQA,
  0 HumanEval).
- At T 0.6/0.8 the papers give no magnitude.
- The card anchor (Q2) bounds dpo's T 0.6 / top_p 0.95 gain at roughly 0-6 points, about ≤ 16 % of the gap.

**My predictions (guesses, stated so they can be checked).**
- dpo arm-A pass@1 at T 0.6: 29-35 % (stored T 1.0: 29.0).
- s0100: the same as dpo, since it is essentially the same model (31.5 vs 29.0 at T 1.0, CIs overlap).
- s0500: +0 to +4.
- s2300: within ±2 of 68.3.
- Reach prompts at dpo: ≤ 1 % of draws.
  - The reach prompts were *selected* as dpo 0/16 at T 1.0. Sharpening toward dpo's modes pushes mass further from
    responses that dpo almost never produces, unless those responses are high-likelihood minority branches (Karan's
    pivotal-token case).
- There may be family-level offsets: count and format slips improve, against-habit constraints get worse.

**What would contradict the papers.**
1. Lower T *reduces* dpo pass@1 beyond noise. This contradicts their general "low temperature helps pre-RL models"
   pattern (weakly, because τ and models differ).
2. A large gain at s2300 from lower T contradicts the pattern of Ji's Table 2.
3. A gain that *grows* with RL step contradicts "already-sharpened models benefit less".

None of these outcomes can contradict their power-sampling claim, which M1 does not test.

**Is a null interpretable?** Yes, as a bound on the cheapest intervention:
- With 2 draws per prompt, the arm-A MDE at dpo is about 3.3 points (Q7). That is about 8 % of the 39.3-point dpo →
  s2300 gap.
- On reach, 0/384 gives a one-sided 95 % bound of about 0.8 % of draws (rule of three, ignoring clustering by
  prompt), against 69.4 % at s2300.

It is *not* evidence that "RL's gain is not sharpening", for four reasons:
1. Low temperature is not p^α (Karan Prop. 1).
2. α ≤ 1.67 is mild.
3. Long traces turn per-token sharpening into degeneration risk rather than better plans.
4. A null on low temperature need not mean a null on power sampling (the DeepSeek-Math-7B case).

A null must also be checked against offsetting effects before it is read as "no effect": loops and truncations versus
gains, and constraint families moving in opposite directions (Q6).

---

### Q6. Controls and reference points

**Necessary**
1. **s2300 at the same lowered T (arm A).** This is Ji Table 2's analogue and the only way to compute the matched gap
   G(T). Without it, a dpo gain cannot be told apart from a generic decoding effect. It is cheap because reach at
   s2300 is not needed.
2. **Pure temperature as the main arm (top_p 1.0).** This keeps the arms aligned with the papers' eq. 3. The card
   setting couples T 0.6 with top_p 0.95; separating them needs one extra cell at dpo (T 0.6, top_p 0.95, arm A).
   That cell is necessary *if* the card setting is reported as a result. Running the card cell only, and attributing
   it to "temperature", would be a confound.
3. **Length and repetition-loop accounting, per cell.**
   - n_tok mean / p50 / p90 / p99, % `finish=length`, % with no `</think>`.
   - An n-gram repetition flag. The Olmo report's own filter on p. 108 uses "heavily repeated (10x+) sentences,
     paragraphs, or (50x+) phrases".
   - Primary pass@1 counts truncated draws as failures. A secondary analysis excludes them.
   - Also a running budget guard. Footnote 71 makes loops at low T a real cost risk: a loop to 32,768 tokens costs
     about 7× a typical draw.
4. **Paired analysis.** Same prompts in every cell, prompt bootstrap of differences, dpo first, pass@1 leading.
5. **Reach-set selection bias.** The reach set is defined by dpo's 0 of 768 stored T 1.0 draws (2026-11-25 summary
   README, lines 29-30). The stored dpo reach baseline is therefore 0 % *by construction*, and any fresh dpo draws
   regress upward. Either compare dpo T 0.6 reach with a *fresh* dpo T 1.0 reach cell, or state that s0100's fresh
   T 1.0 rate (0.1 % of 1,536 draws) is the proxy. Run the fresh cell only if dpo T 0.6 reach comes back small but
   non-zero (a conditional control).
6. **No setting chosen on reach.** T values fixed in advance ({0.6, 0.8}, with 0.6 from the card) need no split. If
   the write-up names a "best T", choose it on half of the 176 arm-A non-reach prompts and report on the other half.
   42 reach prompts are inside arm A, so "chosen on arm A" is not allowed.
7. **Prompt identity check instead of a sampling control.** Dan's meta records the sha256 of each prompt's token ids.
   The modified sampler (T and top_p are module constants, so it must be modified) should reproduce those hashes and
   settings, and log T and top_p in the meta. This is cheaper than a fresh T 1.0 arm-A cell and catches the realistic
   failure (a template or tokenisation drift).
8. **Baseline construction at s0500.** Dan's `M_s0500` rows have 4 to 32 draws per prompt. `sample.py` has a
   sequential stop-after-K mode, so draws per prompt can depend on outcomes. The T 1.0 baseline must use fixed-n
   draws (or the first k chunks in index order that every prompt has). Otherwise per-prompt rates are biased. I did not
   check which jobs make up "8 per prompt elsewhere"; my naive mean over `M_s0500` is 40.8 against the stated 41.2.

**Nice to have**
1. A per-constraint-family breakdown. It is free, and it detects offsetting effects.
2. pass@2 (unbiased estimator), and reach "any pass in 8" as the diversity readout.
3. Dose-response with T 0.8 at dpo.
4. **Likelihood diagnostic: Karan Fig. 4 on our data. This is the one I value most per dollar.**
   - Prefill-score Dan's stored s2300 draws (all draws, and passing reach draws in particular) and dpo's own stored
     draws under dpo. Compare total and per-token log-likelihood.
   - If s2300's passing responses sit in dpo's high-likelihood region, sharpening-type samplers could in principle
     reach them.
   - If they sit below dpo's own typical samples, no α > 1 sampler of dpo concentrates mass on them.
   - This tests the sharpening premise directly, which neither paper does.
   - It is prefill only. Reach s2300: 1,536 draws × about 6.5 k tokens ≈ 10 M prefill tokens; prefill is much faster
     than decode. It can share the dpo model load.
   - vLLM's `prompt_logprobs` path, or HF, needs checking.
5. s0100 cells. Its arm-A gain over dpo (2.5 points) is below the MDE, so s0100 adds a replicate of dpo, not
   information about RL's gain.

---

### Q7. Minimum cells, cost, and when power sampling would be worth proposing

**Assumptions.**
- 1,500 tok/s on one H200, from CONTEXT's example: 1,000 × 4,448 tokens in 50 min gives 1,483 tok/s. At $5.29/h
  this is about $0.98 per million generated tokens.
- Mean tokens per draw at T 1.0: dpo 4,448; s0100 4,347; s0500 about 4,840 (from `M_s0500`); s2300 6,277; dpo reach
  about 4,480 (Dan's `lengths.csv`).
- Model download and load about 10 min per checkpoint.
- A +30 % reserve for loops and length growth at lower T.

**The lead as written in CONTEXT** (T ∈ {0.6, 0.8} × {dpo, s0100, s0500} × (218 × 2 + 48 × 8)) costs about
22.4 M tokens. That is about 4.1 h, roughly **$22 + $2.6 for loads ≈ $24.5 (about $31 with the reserve)**, against
the $10 split. Range at 1,400-1,700 tok/s: about $21-27 before the reserve.

**Proposed minimum (core cells).** All at top_p 1.0. For the 42 reach prompts inside arm A, the first 2 of the 8
reach draws double as arm-A draws.

| Cell | Checkpoint × T | Prompts × draws | Draws | Tokens | Cost |
|---|---|---|---|---|---|
| C1 | dpo, T 0.6 | arm A × 2 + reach × 8 | 736 | about 3.3 M | about $3.2 |
| C2 | s0500, T 0.6 | arm A × 2 + reach × 8 | 736 | about 3.6 M | about $3.5 |
| C3 | s2300, T 0.6 | arm A × 2 | 436 | about 2.7 M | about $2.7 |

Core total: about 9.6 M tokens ≈ 1.8 h + 3 loads ≈ **$12 (about $15 with the reserve)**. A trimmed version drops
reach at s0500: about $10.6.

Add-ons, in priority order:

| Cell | Checkpoint × T | Prompts × draws | Cost | Purpose |
|---|---|---|---|---|
| C4 | dpo, T 0.6, top_p 0.95 (card) | arm A × 2 | about $1.9 | Separates top_p |
| C5 | dpo, T 0.8 | arm A × 2 | about $1.9 | Dose |
| C6 | dpo, T 1.0, fresh | reach × 8 | about $1.7 | Only under the condition in Q6.5 |

The likelihood diagnostic (Q6 nice-to-have 4) costs about $1-2 if it shares C1's load.

**Power** (from Dan's stored rows; these assume a homogeneous effect, and a heterogeneous effect widens them).
- Mean within-prompt Bernoulli variance at dpo is 0.055: 107/218 prompts pass 0 of 16 and 35 pass 16 of 16.
- Paired SE for 2 new draws vs 16 stored ≈ √((0.055/2 + 0.055/16)/218) ≈ 1.2 points. 95 % half-width about ±2.3;
  **MDE (80 % power) about 3.3 points**.
- s0500 (variance 0.085, 8 stored): MDE about 4.4. s2300 (0.097): MDE about 4.7.
- 4 draws at dpo would bring the MDE to about 2.5 at double the cost. I do not think that is worth it.

**When a power-sampling follow-up would be worth proposing.**
1. *Budget gate (Leon's rule).* The sweep moves dpo arm-A pass@1 by at least the MDE, with a paired CI excluding 0,
   *and* the matched gap G(T) shrinks. The gain must be specific to the pre-RL model, not equally present at s2300.
2. *Scientific caveat to put in front of Leon.* This gate is a budget choice, not something the papers imply: a null
   on low T does not predict a null for power sampling (Ji, DeepSeek-Math-7B). The better scientific gate is the
   likelihood diagnostic. Power sampling is worth proposing only if s2300's passing reach responses are not in dpo's
   low-likelihood tail.
3. *Feasibility and cost at our lengths.* Karan's MCMC costs N_MCMC·T/(4B) times plain decoding.
   - At the papers' B = 192 and T ≈ 4,450 that is about 58×: one draw per arm-A prompt at dpo ≈ 56 M tokens ≈ $55.
     Two draws ≈ $110.
   - Only block sizes 5-10× larger than any tested are affordable. B = 2048 gives about 5.5×, about $6 for 218 single
     draws. But Ji's Table 8 shows accuracy falling as B grows (B = 768: 70.4 vs 75.8 at 192).
   - Ji's Algorithm 2 at K = M = 8 costs at least 64× in tokens, and its H = 192 horizon cannot see the IF verdict.
   - So CONTEXT's "about 10x" is achievable only outside every tested setting. Any proposal must say so.
4. *Pilot before any proposal.* About 20 dev prompts (arm-A non-reach), α = 4, B = 2048, N_MCMC = 4, proposal at τ =
   0.6 rather than 0.25 (footnote 71). Measure:
   - the MH acceptance rate (if it is about 0, the chain is just the proposal);
   - loop and length rate;
   - tokens per sample;
   - whether vLLM 0.11 returns the raw T = 1 log-probabilities the MH ratio needs. I believe it has a logprobs-mode
     setting but have not verified that.

   Cost: about 0.2 M tokens if no draw loops. The multiplier grows with length (16× at 32,768 tokens with this B), so
   each draw that loops to the cap costs about 0.5 M tokens and all 20 looping would cost about 10 M. Run it with a
   loop/overrun abort; budget about $2 plus a load. Without acceptance rates at long lengths, any cost estimate for the
   full run is a guess.

END OF INDEPENDENT READ
