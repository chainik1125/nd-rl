# M1 (key `temp`): does sharpening dpo's own distribution recover part of RL's gain?

## Turn 1 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 01:42 PT [time corrected by the orchestrator from the file's modification time; the heading as written said 01:50 PT, which was not taken from `date`]

Status: design proposal only. No pod code written, nothing launched. Notes: `docs/literature/2026-11-27-elicitation-metrics-fulltext/{karan2025reasoningwithsampling,ji2026scalablepowersampling}.md` (page numbers below refer to the PDFs read in full). Baseline helper: `tools/temp_baseline_inventory.py`.

### 1. What the full read and the decoding facts change

1. **The model card and the report disagree with the sampler.** Both repos' READMEs say "We evaluated our models on the following settings. We also recommend using them for generation: temperature 0.6, top_p 0.95, max_tokens 32768" (Olmo-3.1-32B-Think README lines 171-174; Olmo-3-32B-Think-DPO README lines 166-169); both `generation_config.json` have `"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768` (the 3.1 file also has `"do_sample": true`). The Olmo 3 report (arXiv 2512.13961, pp. 37, 114) says post-training evaluation uses "a sampling temperature of 0.6 and top-p of 0.95" (p. 37; p. 114 A.8.1), and RL-side filtering uses "temperature of 1.0 and top-p of 1.0, matching how we sample during RL training" (p. 47); the RL-Zero evals use T 1.0 for pass@k (p. 64). Footnote 71 (p. 114): "We find that both thinking models degenerate quickly when evaluated with low temperatures (as used in OLMo 2), while instruction models can be evaluated at this higher temperature." Dan's T 1.0 / top_p 1.0 is therefore the RL rollout setting, NOT the setting the authors report their benchmark numbers at. Pages read in the report: 37-38 (eval setup, Table 14), 47, 64, 114-115 (via text search for temperature/top-p/IFBench, then the surrounding paragraphs); not read: the rest of the 118 pages.
2. **Official numbers for orientation (different prompt set, T 0.6):** Table 14 (p. 38), full IFBench: SFT 37.0, DPO 34.4, Olmo 3 Think 47.6, Olmo 3.1 Think (final) 68.1 (the "+20 points on IFBench" sentence is on the same page). Dan's arm-A numbers at T 1.0 are dpo 29.0 %, s2300 68.3 %. The s2300 figure matches the official 68.1 closely despite the different setting and prompts; dpo is 5 points lower at T 1.0 than the official 34.4 at T 0.6, which is the size of effect this experiment would measure. This is an observation about two different sets, not evidence for sharpening.
3. **Karan & Du do not show that RL gain is sharpening, and do not isolate p^alpha from low temperature.** Their Table 1 (p. 9) shows low-temperature sampling (tau stated nowhere in that paper; 0.25 in Ji, Table 5 p. 28) recovering about two thirds of GRPO's MATH500 gain on Qwen2.5-Math-7B (.496 to .690 vs GRPO .785). So the cheap arm of our test is already most of what they show. Ji et al. Table 2 (p. 7) shows low temperature LOWERING accuracy on an RL-tuned model (DeepSeek-Math-7B-RL MATH500 .492 to .412), which is the case for `s2300` and why the matched-decoding reference matters.

### 2. Baseline (read only; Dan's stored T 1.0 draws, top_p 1.0, 32,768 cap)

Reproduces the known table (29.0 / 31.5 / 41.2 / 68.3 pass@1 on arm A; 0.0 / 0.1 / 4.5 / 69.4 on reach). Source: `tools/temp_baseline_inventory.py`.

| ckpt | set | jobs (Dan, `olmo_ifcurve` rows unless stated) | draws / prompt | mean tokens | p90 tokens | cut at cap |
|---|---|---|---|---|---|---|
| dpo | arm A (218) | F_dpo_ifb | 16 | 4,448 | 9,770 | 0.06 % |
| dpo | reach (48) | `olmo_ifomega` F_dpo_ifb + F_dpo_ife (first 16) | 16 | 4,520 | 9,645 | 0.00 % |
| s0100 | arm A | F_s0100_ifb | 8 | 4,347 | 9,470 | 0.00 % |
| s0100 | reach | F_s0100_ifb + B_s0100 | 32 | 4,753 | 9,858 | 0.13 % |
| s0500 | arm A | M_s0500 + T_s0500 | 8 | 4,610 | 9,568 | 0.17 % |
| s0500 | reach | M_s0500 | 32 | 4,994 | 9,720 | 0.00 % |
| s2300 | arm A | F_s2300_ifb | 8 | 6,277 | 11,351 | 0.00 % |
| s2300 | reach | B_s2300 | 32 | 6,863 | 11,006 | 0.00 % |

(Arm A "cut at cap" = `finish == length` or tokens at 32,768; token percentiles over all draws.) Pipeline noise at T 1.0: our own regeneration at the same sampler was within -0.8 points at dpo (CI -2.9 to +1.3) and +2.1 at s2300 (`olmo_elicitation_20261125/log.md`, Stage 0c), so a "move" smaller than about 3 points is inside pipeline noise.

### 3. Proposed cells (as few as answer the question; tiers triggered by pre-stated rules)

All cells: Dan's sampler (vLLM 0.11.0, `olmo_thinker` template, bf16, max 32,768) with temperature and top_p passed as arguments by a wrapper written later (Dan's files stay unedited). Draw order is kept so first-n comparisons are possible. Reach prompts that are also in arm A (42 of 48) are run once with 8 draws, and arm-A scores use their first 2 draws.

**Tier 1 (always): dpo at the model card's setting, C1 = T 0.6, top_p 0.95.**
- 176 non-reach arm-A prompts x 2 draws + 42 reach-and-arm-A x 8 + 6 reach-only x 8 = 736 draws. Arm-A pass@1 uses 2 draws per prompt (218 prompts), reach uses 8 per prompt (48 prompts).
- Why this one first: it is the setting the authors evaluate and recommend at; it is the strongest sharpening that the report says does not collapse (fn 71 warns about lower); it is Leon's lead (0.6). It also confounds T and top_p (0.95): labelled "card decoding", not "temperature".
- Reference cells already stored and free: dpo at T 1.0 (16 draws on arm A, 16 on reach), s2300 at T 1.0.

**Tier 2 (only if Tier 1 "moves", see rule below):**
- dpo C2 = T 0.8, top_p 1.0, arm A x 2 (436 draws): a dose-response point (needed for the power-sampling condition and to see if effect is monotone in T).
- s2300 at C1, arm A x 2 (436 draws): the matched-decoding reference for the recovered share. Reach at s2300 is not rerun (69.4 % at T 1.0; ceiling issues, and the question is about dpo).

**Tier 3 (optional, only if Leon wants the checkpoint path): s0500 at C1 arm A x 2; dpo C3 = T 0.6, top_p 1.0 arm A x 2 (isolates top_p).** s0100 is dropped (31.5 % vs dpo 29.0 %, within noise at 218 prompts, so nothing to learn). Leon's lead listed s0100 and s0500; I propose them only after dpo moves because without a dpo effect the path question does not arise.

**Why 2 draws on arm A and 8 on reach.** Per-prompt draws are already 16 at dpo T 1.0, so the baseline is tight and the new cell's prompt-bootstrap interval is dominated by prompt heterogeneity, not draws (my estimate: SE about 2 points from 218 prompts; a guess, to be checked when the data exist). 2 draws x 218 prompts is Leon's lead. The reach prompts need 8 draws because dpo passed 0 of 768 at T 1.0 (95 % upper bound about 0.4 % by the rule of three), so any pass at 384 draws is already informative.

**Controls.**
- Same-sampler T 1.0 baseline: already done (Stage 0c reproduction above), not re-run.
- No tuned setting: C1 is fixed in advance by the model card; C2 by Leon's lead. Nothing is chosen from the data, so no dev / held-out split is needed for Tier 1-2 and the reach set is never used for selection. If a later step must choose a setting (T for s0500, alpha for a power-sampling follow-up), choose on the EDL pool's 64 train prompts and score on its 32 test prompts plus reach (`olmo_elicit/pool/`; all arm-A non-reach).
- Report separately: share of draws cut at the cap, response tokens, pass@1 among draws that finish. A repetition loop counts as a checker fail in the headline and is reported next to it, not removed.
- The checker is Dan's strict grader; I say "checker passes". No reading of whether passes are genuine.

### 4. Recovered share, defined

Notation: P(c, T) = arm-A pass@1 of checkpoint c at decoding T, averaged over 218 prompts with all stored draws, prompt bootstrap 10,000 resamples, paired over prompts.

- **Primary (needs only stored data plus Tier 1): share_RL = [P(dpo, C1) - P(dpo, 1.0)] / [P(s2300, 1.0) - P(dpo, 1.0)]**, denominator 68.3 - 29.0 = 39.3 points. It answers "can I get part of what RL gave, at the RL setting, just by decoding dpo differently". Dan's T 1.0 is the RL rollout setting and the point at which the 39.3 points were measured, so this is the apples-to-apples reading of his curve. Interval: bootstrap on the paired numerator (prompts shared with the stored draws); the denominator is treated as fixed and its interval reported separately; I will not quote a ratio interval as precise (small numerators make a ratio unstable).
- **Secondary (Tier 2): remaining gap at matched decoding, G(C1) = P(s2300, C1) - P(dpo, C1), against G(1.0) = 39.3.** If RL were only sharpening, G would shrink towards 0 as T falls; if G(C1) is about 39 points both models gained alike, or neither moved. Ji Table 2 (p. 7) shows an RL-tuned model can lose accuracy at low temperature, so s2300 at C1 is not assumed to be flat; it must be measured. I report both and do not average them.
- Also reported: pass@2 (first two draws) and, on reach, the fraction of draws and of prompts with at least one pass, because sharpening can raise pass@1 while lowering pass@k (Karan & Du p. 10, Ji p. 8 on diversity).

### 5. Cost (tokens, H200 minutes, dollars; assumptions marked)

Assumptions (guesses marked G): vLLM about 1,500 tokens/s on one H200 (observed range 1,400-1,700, CONTEXT.md); $5.29/h so about $1.0 per million generated tokens; model load plus download about 8 min (about $0.7) per checkpoint, observed 2-10 min. G: low-T mean length 1.13x the T 1.0 mean (that is 2 % of draws looping to 32,768 on top of the stored mean); no data exist at low T for these models. Sensitivity below.

| item | draws | tokens (M) | H200 min | $ |
|---|---|---|---|---|
| Tier 1: dpo C1 (+ load) | 736 x ~5,100 | 3.75 | 42 + 8 | **about 4.4** |
| Tier 1 with 5 % loops (mean about 5,900) | 736 | 4.34 | 48 + 8 | 5.0 |
| Tier 1 with 15 % loops (mean about 8,700) | 736 | 6.4 | 71 + 8 | 7.0 |
| Tier 2: dpo C2 (arm A x2, same load) | 436 x ~5,100 | 2.2 | 25 | 2.2 |
| Tier 2: s2300 C1 (arm A x2, + load) | 436 x ~6,900 | 3.0 | 33 + 8 | 3.4 |
| Tier 3: s0500 C1 (+ load) | 436 x ~5,100 | 2.2 | 25 + 8 | 2.9 |
| Tier 3: dpo C3 (reuse a dpo load, else +0.7) | 436 x ~5,100 | 2.2 | 25 | 2.2 |

Totals: null result stops at Tier 1 for **about $4.4-7**; Tier 1+2 about **$10-13**; everything about $17. The $10 M1 starting split is met if Tier 1 is null or if Tier 3 is skipped. Safeguard: a watchdog that aborts a cell if over 20 % of the first 100 finished draws hit the cap (the cap is the same 32,768, so the cell stays comparable).

### 6. Trigger rule for Tier 2, pre-stated

"Moves" means either of (a) the paired arm-A difference P(dpo, C1) - P(dpo, 1.0) is at least +4 points with the 95 % bootstrap interval above 0 (4 is chosen above the Stage 0c noise of about 2-3 points), or (b) at least 3 checker passes on reach in the 384 reach draws coming from at least 2 prompts (T 1.0 gave 0 of 768). If neither holds, we report the null at the card setting ("sharpening to T 0.6 does not recover RL's gain on this checker") and stop; we do not add T 0.8 to look for a window, because fn 71 predicts lower T degrades. If the arm-A effect is negative, we report the negative (it would show that the dpo model's T 1.0 outputs are better than T 0.6 on this checker) and stop.

### 7. What each outcome would and would not show

| Outcome | Would show | Would not show |
|---|---|---|
| dpo C1 about equal to dpo T 1.0 (within 3 points), reach 0 | At the card's own setting, token-level sharpening of dpo does not recover RL's IFBench gain on this checker. Consistent with RL adding behaviour that p(dpo) places little mass on (Dan's reach prompts: 0 of 768 passes). | That no sampling method works: Karan & Du's point is exactly that p^alpha is not low temperature; they show p^alpha a few points above low T on most cells (Table 1). Nor that RL is not sharpening: reach prompts could need a global, not a local, sharpening. |
| dpo C1 up by 4-15 points, share_RL 10-40 %, reach about 0 | Part of the arm-A gain is cheaply available through decoding; part of RL's improvement is of the kind that moves with T (calibration of tail tokens). | That RL "is" sharpening: the reach prompts (which dpo never passed) stay out of reach, a sign the bulk is not reachable by local sharpening. Also, a lift limited to prompts with a high baseline pass rate would be a variance effect: report by baseline-rate bins. |
| dpo C1 up by 15+ points or reach lights up | Strong sharpening component; the "RL gain" on this checker is substantially available at the model card's own decoding, and Dan's T 1.0 curve understates dpo. | That s2300 would not also gain: check G(C1). If s2300 also gains, the RL gap stays; that is not an outcome of sharpening. |
| dpo down (negative), loops raised | T 1.0 is the better setting for this checker; the report's low-T degeneration (fn 71) shows up at 0.6 for dpo or the stricter constraint checks need the longer, more varied thinking. | That the card is wrong in general: it was tuned on other benchmarks. |
| s2300 at C1 higher than s2300 at T 1.0 | The reference moves too; use G(C1), not share_RL, as the headline for "same decoding". | |
| s2300 at C1 lower | RL-trained models are better at their training temperature (Ji Table 2 analogue); share_RL stays primary. | |

### 8. Power-sampling follow-up: condition, design, cost (NOT approved, nothing proposed for now)

**Condition for proposing it** (all must hold): (i) Tier 1 or 2 "moves" by rule 6(a) on arm A; (ii) the effect is not all of RL's gain, i.e. share_RL is between about 10 % and 70 % and dpo at the best sweep setting is still at least 10 points below s2300 (otherwise either nothing to explain, or temperature already does the job); (iii) the effect is not a loop artefact: cut at cap below 5 % at C1 and the lift persists among draws that finish; (iv) the effect is monotone in T (C2 sits between T 1.0 and C1). Leon's own condition ("only if the sweep moves pass@1") is (i); (ii)-(iv) are my additions, for Leon to accept or drop. Karan & Du's p^alpha is only interesting where low T gives partial, not total, recovery (their Table 1 pattern).

**Rough design.** alpha is tied to the proposal temperature in Karan & Du (p_prop = base model at 1/alpha, p. 8). At T 0.25 (alpha = 4) a Olmo thinking model will loop (fn 71), so use alpha = 1/0.6 = 1.67 (the value Karan & Du use for the HumanEval pass@k plot, p. 17), proposal at T 0.6 top_p 0.95. Blocks: B around T/16; for 5,000-token traces B = 512-1,024 (their B = 192 tied to T_max = 3,072). MCMC cost multiplier N_MCMC T / (4B) (Eq. 12, p. 7): with N_MCMC = 10, T = 4,500: B = 1,024 gives 11x, B = 512 gives 22x, B = 192 gives 59x. The sampler also needs sequence log-likelihoods of every proposal under the base model (extra scoring passes, or logprobs returned by vLLM), plus a custom MH loop on vLLM (no such loop exists in Dan's sampler). Ji et al.'s estimator (K_t = M_t = 8, H_t = 192 lookahead, B = 192) is claimed at 2.5-3.5x plain decoding at T_max 3,072 (p. 28); for 5,000-token responses I would not extrapolate without a pilot, but the cost scales with the number of blocks (about 26 per response at B = 192). Cost of one arm A x 2 cell (436 draws x about 5,000 tokens = 2.2 M plain tokens): MCMC at 11x about 24 M tokens, about 4.5 H200-hours, about $24; Ji-style at 3x about 6.5 M tokens, about 1.2 h, about $6.5; plus engineering (several hours of agent time, a pilot on 10 prompts at about $1), and no result reuse from either paper's code since both are 3,072-token, 7B setups. Reach with 8 draws: another 384 draws, about 0.9x of the arm-A cell. Choose alpha (1.67 or 2) on the 64 EDL-pool train prompts, test on the 32 test prompts and reach. I would pilot Ji-style first, since Karan & Du's MCMC is 4x the cost at the same B.

### OPEN (for the auditor)

1. Is "moves" (+4 points, CI above 0, or 3+ reach passes) the right threshold? It is a guess based on Stage 0c noise.
2. Single top_p confound in Tier 1 (T 0.6 with top_p 0.95): accepted for minimality and labelled; Tier 3 resolves it if funded.
3. 2 draws per prompt on arm A for the new cells vs 16 stored for the dpo baseline: unequal draws; analysis uses all stored draws for the baseline and a prompt bootstrap, but sampling variance of a 2-draw cell is larger than of the baseline cell.
4. The loop-rate assumption (2 % extra, mean 1.13x) is a guess; no data on these models at T 0.6 exist in the stored files, only the report's warning.
5. The Ji et al. results re-use Karan & Du's numbers for the Qwen rows (identical to the third decimal); I treat them as one data source, not two.

## Turn 2 — auditor (Claude Opus 5.5), 2026-11-27 01:59 PT

My independent read is `discourse/temp_audit_independent.md` (md5 `9943c5cfd64ac86f4a5af92045ce62a7`), written before
I opened the reader's notes, Turn 1 or the inventory tool. Since then I have read both notes, Turn 1, and
`tools/temp_baseline_inventory.py`. I re-ran the inventory, and its numbers reproduce exactly: 29.0 / 31.5 / 41.2 /
68.3 on arm A and 0.0 / 0.1 / 4.5 / 69.4 on reach, with 8 of 8 draws per prompt at s0500. I also read the 2026-11-25
Stage 0c files: `olmo_elicitation_20261125/log.md` lines 48-50 and 151, `olmo_elicit/stage0/rep.csv`, and
`olmo_elicit/runs/{dpo,s2300}/rep/graded.jsonl`.

### A. Disagreements on the papers and on Turn 1's factual claims

Where we agree, one line per paper:
- **Karan:** construction (eq. 2-12, Alg. 1, α = 4, B = 192, N_MCMC = 10, proposal at 1/α); Prop. 1; the Table 1
  values; low temperature closing 67 % of GRPO's MATH500 gap on Qwen2.5-Math-7B; no seeds or intervals; nothing on IF
  or long traces; sharpening cited, not tested; Phi-3.5-mini-instruct as our nearest analogue.
- **Ji:** Theorem 3.1 and the estimator / jackknife; the hyperparameters (K = M = 8, H = B = 192, α = 4, τ_low = 0.25,
  seed 0); Tables 1-3; the Qwen rows reused from Karan; Table 2 (low T hurts an RL model); loose bounds; nothing on IF
  or long traces.
- **Report:** footnote 71, p. 114; T 0.6 / top_p 0.95 for evaluation; T 1.0 / top_p 1.0 for RL.

Disagreements:

1. **Low-T τ in Karan.** Both of us say it is unstated. The reader infers 0.25 from Ji (p. 28) only. There is also
   in-paper evidence. In Fig. 6 right (p. 11), the N_MCMC = 0 points are about 69.0 and about 62.8. These equal the
   Table 1 low-temperature MATH500 rows (p. 9), and zero MH steps means the proposal alone at 1/α. *Both readings are
   supported; mine adds Karan's own evidence.*
2. **Fig. 6 left "could not read exact numbers"** (reader, Karan note "Ambiguous"). The bars carry printed labels
   (p. 11; I read the rendered page). For α = 1.0 the labels are 45.0 / 35.8 / 38.4, but Table 1 Base is 49.6 / 49.8 /
   40.0 (p. 9). α = 1 is plain base sampling, so these are two runs of the same sampler 4.6 / **14.0** / 1.6 points
   apart, unexplained.
   - That is in-paper evidence that Karan's run-to-run or setup variation is as large as several claimed effects.
   - *The text supports me.*
3. **Identical Base values across models.** Karan Table 1 (p. 9) gives Base HumanEval .329 and GPQA .278 for *both*
   Qwen2.5-Math-7B and Qwen2.5-7B, which is likely a copy error. The reader's table reproduces them without comment.
   *The text supports my flag.*
4. **Provenance of Karan's GRPO.** The reader says "GRPO runs trained for this paper … one run per model". Karan p. 8
   says only "we use the implementation of GRPO in Shao et al. [2025], which posttrains these models". Ji p. 28 says Ji
   used Shao et al.'s released step-150 checkpoint, and Ji's GRPO numbers equal Karan's.
   - *Neither text settles it.* The reader's wording is stronger than the source.
   - This does not affect the design.
5. **Ji's cost ("2.5-3.5× slower" → Turn 1 §8 "Ji-style at 3x … about $6.5").** Ji p. 28 measures *wall-clock per
   prompt* ("We report wall-clock generation time per prompt"). Ji p. 23 gives the token cost O(Σ[L_t·B + K_t·M_t·H_t]
   + L_w·B), and p. 5 says "the total inference burden is bounded by O(Σ K_tM_t) generated sequences".
   - At K = M = 8 and H = B, that is at least 64 rollout blocks per output block, so at least 65× the tokens of plain
     decoding.
   - Our pod is throughput-bound (hundreds of requests in flight), so we pay tokens, not latency.
   - Even if short rollouts that share a prefix ran at 3× our 1,500 tok/s (a guess), one arm A × 2 cell would cost at
     least $45, not $6.5. Turn 1's "pilot Ji-style first because MCMC is 4x the cost" is therefore reversed.
   - *The text (p. 23, p. 5, p. 28) supports me.*
6. **Ji's Algorithm 2 does not target p^α.** Candidate blocks are sampled from p, cut to the top-K by likelihood, and
   then weighted by p^α·ζ̂ with no division by the proposal probability (Alg. 2 lines 4-5, eqs. 19-21, p. 23). By
   standard sampling-importance-resampling logic, the chosen block is then distributed roughly as p^{α+1}·ζ̂ over the
   top-K.
   - The reader noted the proposal-temperature ambiguity but not this.
   - *The construction is in the text; the p^{α+1} consequence is my derivation.* It matters only for a follow-up.
7. **H_t = 192 lookahead.** The reader calls it "a small slice of a 5,000-token trace". I agree, and add that the IF
   verdict sits in the final answer, thousands of tokens beyond any rollout. Agreement in substance.
8. **Turn 1 §8: "p^alpha is only interesting where low T gives partial, not total, recovery", and condition (i) "moves".**
   Ji Table 1 (p. 6), DeepSeek-Math-7B, MATH500: low T goes .362 → .366 (3 % of the 13-point GRPO gap), while Ji's
   sampler reaches .464 (78 %) and MCMC .424.
   - The papers' own data contain a case where low T does nothing and power sampling does most of the work.
   - "Only if the sweep moves" is Leon's budget rule. It must be labelled as such, not justified from the papers.
   - *The text supports me.*
9. **Turn 1 §1.2: "dpo is 5 points lower at T 1.0 than the official 34.4 at T 0.6, which is the size of effect this
   experiment would measure".** This compares arm A (218 prompts) at T 1.0 with full IFBench (300 prompts) at
   T 0.6 / top_p 0.95, and neither Table 24 (p. 52) nor Table 14 (p. 38) says whether the score is strict or loose.
   - Over the same 300 prompts, Dan's T 1.0 draws give dpo 28.0 strict / 33.8 loose, and s2300 67.6 / 69.6 against the
     card's 68.1. So the card implies a dpo effect of about +6.4 (if strict) or +0.6 (if loose), from a different
     harness.
   - *The text supports "0 to 6, metric unknown", not "about 5".* This affects only the prior.
10. **Turn 1 §3: "the interval is dominated by prompt heterogeneity, not draws (SE about 2)".** The headline quantity is
    a *paired* difference, so heterogeneity in the level cancels and, under the null, the interval is draw-dominated.
    - Stage 0c measured exactly this design: 2 fresh dpo T 1.0 draws against 16 stored, true difference 0. Its paired
      95 % CI was [−2.9, +1.3], half-width 2.1, SE about 1.1 (`stage0/rep.csv`; log.md line 50).
    - My computed value from the stored within-prompt variance (0.055) is SE about 1.2.
    - *The data support me.* The practical consequence is in B5.
11. **Turn 1 §3 reach: "dpo passed 0 of 768 at T 1.0 (95 % upper bound about 0.4 % by the rule of three)".** Those 768
    draws *define* the reach set ("0 of his 768 stored draws", 2026-11-25 summary README lines 29-30), so the bound is
    not valid for fresh draws.
    - Fresh T 1.0 evidence already exists. Stage 0c's dpo replication includes 42 reach prompts × 2 draws: **0 of 84
      pass** (my count in `runs/dpo/rep/graded.jsonl`). s0100 (fresh, 32 per prompt) has 2 of 1,536 (0.13 %).
    - The stated baseline should be about 0.1 % from fresh draws.
    - This also settles my own independent read's conditional "fresh dpo T 1.0 reach control" (C6). **I withdraw it.**
      The data exist, so the cost is $0.
12. **s0500 baseline draw counts.** My independent read flagged that `M_s0500` has 4 to 32 draws per prompt (possible
    sequential stopping). The inventory, through `dan_curve.Rows.draws`, takes the first N = 8 rows in source order
    (`M_s0500`, then `T_s0500`), sorted by (chunk, i). That gives 8 of 8 for all 218 prompts and 41.2 %; my 40.8 was the
    naive variable-n mean.
    - A fixed first-8 rule is unbiased even under a stop-after-K rule. Draws 1-4 always exist, and whether a later draw
      comes from M or T depends only on earlier draws, not on its own outcome.
    - *The reader's construction is supported, and I withdraw the concern.* One residual I did not re-verify:
      `T_s0500` is assumed to use the same decoding, and `test_stage0.py` checks constants equal to Dan's.

### B. Attack on the design

**B1. Tier 1 is only the card setting (T 0.6 + top_p 0.95), which is confounded.**
- My minimum used pure temperature (T 0.6, top_p 1.0) as the main arm, because the papers' object is eq. 3 / p^α.
  Nucleus truncation is not a power of p.
- I concede the card cell as Tier 1. It is the developers' evaluation setting (report p. 37 and Table 16 p. 40), and it
  is the *sharper* of the two (tempering plus tail cut), so a null there is the stronger null.
- **Condition:** if Tier 1 moves in either direction, the pure-T cell (dpo, T 0.6, top_p 1.0, arm A × 2, **$2.2** on the
  same load) is mandatory. It must come before any headline that says "temperature" or "sharpening p", and before any
  power-sampling proposal.

**B2. s2300 at matched decoding only "if dpo moves".**
- I wanted it unconditional. I accept conditional, with the trigger made **two-sided**: a significant drop at dpo also
  triggers s2300 at C1, arm A × 2 (**$3.4** including load).
- A negative dpo result is only interpretable against the matched reference. It could mean "card decoding hurts the
  pre-RL model" or "it hurts these models on this checker", and Ji Table 2 predicts the RL model drops too.
- For a null, the card's 68.1 against our 67.6 strict is a stated, weak anchor.

**B3. s0500 dropped to "optional, if Leon wants the checkpoint path".** I require **s0500 at C1, reach × 8 (384 draws),
unconditionally: about 2.17 M tokens ≈ $2.1 + load $0.7 = $2.8.**
- A dpo reach null is close to certain by construction. The fresh dpo rate is about 0.1 %, and an α > 1 sampler can
  only concentrate mass dpo already has.
- s0500 is the first released checkpoint with real mass on reach (4.5 % of draws; 43.8 % of reach prompts pass in 32
  draws). So it is the only cell where "sharpening multiplies a partly learned behaviour" can show. That is CONTEXT's
  question ("cheaply recoverable from earlier checkpoints, and by what kind of intervention").
- Both directions are informative. Up means card decoding multiplies early RL's gain. Down means early RL's passing
  responses sit in s0500's tail, which is Ji Table 2's pattern.
- Arm A × 2 at s0500 (+$2.2 on the same load) I would accept leaving out.

**B4. Fresh T 1.0 reach control.** Withdrawn (A11). Required instead: the write-up states the fresh baseline (0 of 84
dpo, 0.13 % at s0100), not "0 of 768". Cost $0.

**B5. The "moves" trigger.**
- *(a) +4 points with the CI above 0.* With paired SE about 1.1 (Stage 0c), "CI above 0" needs only about +2.2
  observed, so the +4 floor is what binds.
  - Power is 50 % at a true +4. 80 % power is reached only at about +5.0, which is 13 % of the 39.3-point gap.
  - The floor was justified as "pipeline noise", but Stage 0c already shows no pipeline offset at dpo (−0.8 [−2.9,
    +1.3]), so the floor counts that noise a second time.
  - **Required:** trigger on the paired 95 % CI excluding 0, two-sided. That gives an MDE of about 3.1 points at 80 %
    power. A false trigger costs at most the $5.6 of conditional cells.
- *(b) At least 3 reach passes from at least 2 prompts in 384 draws.* Fine against the fresh baseline: the expected
  count is 384 × 0.0013 ≈ 0.5, so P(≥ 3) ≈ 1.4 % before the two-prompt condition. Keep it, but phrase it against the
  fresh rate.
- *Missing:* the case where the arm-A result is null but loops rise. See B8 and required change 6.

**B6. Recovered-share definitions.** I agree: share_RL is primary (it matches the papers' comparison of base under a
sampler against RL at its native decoding), and the matched gap G is secondary. Changes, all at $0:
1. Bootstrap the *ratio* jointly over the shared prompts. The denominator (39.3, SE about 2-3) is large, so the ratio is
   well behaved. Treating the denominator as fixed understates the interval a little.
2. Reach share against the fresh baseline: [r(dpo,C1) − 0.13] / [69.4 − 0.13].
3. Split arm A into the 111 prompts dpo passed at least once in 16 draws and the 107 it never passed. Sharpening can
   only move the former, so this separates "sharpening works where mass exists" from "sharpening creates mass".
4. Keep Turn 1's baseline-rate bins.

**B7. s0500 baseline counts.** Resolved in the reader's favour (A12).

**B8. Loop accounting.**
- Turn 1's watchdog ("abort if over 20 % of the first 100 finished draws hit the cap") is structurally blind. Looping
  draws finish *last*, so the first 100 finished draws under-sample loops. By the time finished draws show 20 % at the
  cap, most of the budget is gone.
- 20 % loops means a mean of about 0.8 × 5.1 k + 0.2 × 32.8 k ≈ 10.6 k tokens, or about $8.5 for Tier 1 alone before
  the watchdog trips.
- **Required:** abort on *in-flight* state:
  - total generated tokens exceed 1.5× the cell's plan before 90 % of draws finish; or
  - at least 10 % of submitted draws pass 16,384 tokens while still running;
  - plus a log of the in-flight length histogram every 5 min.
- Also required in every table:
  - an n-gram repetition flag on *all* draws, using the report's own filter thresholds (p. 108: "heavily repeated (10x+)
    sentences, paragraphs, or (50x+) phrases");
  - the share of draws with no `</think>`.
- Headline pass@1 counts loops as failures, as Turn 1 has it. "pass@1 among finished draws" stays secondary, because
  conditioning on finishing depends on T.
- Code-only, $0.

**B9. Power and resolution at 2 draws.**
- I agree that 2 draws at dpo suffice. The MDE is about 3.1-3.3 points (about 8 % of the gap), and Stage 0c confirms
  SE about 1.1.
- I agree with dropping s0100: its 2.5-point gain is below the MDE.
- State the per-cell MDEs in the write-up: about 4.4 points at s0500 arm A and about 4.7 at s2300 arm A, from the
  stored within-prompt variances 0.085 and 0.097. A null there means "< about 4.5 points", not "no effect".
- Reach at 8 draws: 0 of 384 bounds the per-draw rate at about 0.8 % (one-sided 95 %, unclustered).

**B10. Prefill-only likelihood check** (Karan Fig. 4, on our data).
- *What it is:* score stored responses under dpo (T = 1 raw log-probabilities, prefill only) and compare total and
  per-token log p_dpo of s2300's reach responses, passing ones in particular, with dpo's own reach draws on the same
  prompts.
- *Cost on our pod:*
  - Subsample 8 stored draws per reach prompt from each of s2300 (384 × about 7.0 k tokens including the prompt
    ≈ 2.7 M) and dpo (384 × about 4.7 k ≈ 1.8 M), so about 4.5 M prefill tokens.
  - At a guessed 4-8 k prefill tokens/s on one H200 for 32B bf16, that is 10-20 min on the dpo load already paid for
    Tier 1, **≈ $1-2**. Scoring all 1,536 + 768 draws (about 14 M tokens) would be $3-6 and is not needed.
  - Unverified: whether vLLM 0.11's `prompt_logprobs` works with prefix caching on. If not, disable caching for this job
    only.
  - The stored token ids are in Dan's samples (base64 int32), so no re-tokenisation is needed.
- *What it adds:* the only direct test of the sharpening premise. Neither paper runs it on the RL'd model it compares
  against (Karan's Fig. 4 is the template). It also explains a reach null:
  - "s2300's passing responses sit in dpo's low-likelihood tail" makes a null the expected result for *any* α > 1
    sampler of dpo.
  - "They sit at or above dpo's typical draws" makes a null a puzzle that points to power sampling.
  - Totals are length-sensitive (s2300 responses are about 50 % longer), so report both totals and per-token means, and
    stratify by length.
- *Replace or gate?* It cannot replace the sweep, which measures the intervention Leon approved. It should **gate the
  power-sampling follow-up** together with Leon's "moves" rule: propose power sampling only if s2300's passing reach
  responses are not in dpo's low tail. It does not need to gate Tier 2.

**B11. §8 power-sampling numbers.**
- Correct the Ji-style cost (A5) and the rationale for piloting Ji first.
- The MCMC arithmetic (B = 1,024 → 11×, about $24 per arm A × 2 cell; B = 192 → 59×) agrees with mine.
- Label condition (i) as Leon's budget rule (A8).
- α = 1.67 has no MATH support in Karan: Fig. 6 has no α between 1 and 2, and α = 2 is already below α = 4. Leave that
  OPEN, since the follow-up is not approved.

**B12. Dev split for any later choice.** The EDL pool's 32 test prompts give CIs of about ±15 points. A fixed random
half of the 176 arm-A non-reach prompts (88 dev, 88 test) plus reach as the held-out set is better powered. I would
accept leaving this OPEN until a choice is actually needed.

**Budget, my required set against Turn 1** (1,500 tok/s, $5.29/h, 1.13× length as in Turn 1):

| Path | Turn 1 | With my required changes |
|---|---|---|
| Null at dpo | $4.4-7 | Tier 1 $4.4 + s0500 reach $2.8 + likelihood check $1-2 = **$8.2-9.2** (about $11 with 15 % loops) |
| Null with loops ≥ 2 % at C1 | — | + dpo T 0.8 arm A × 2 $2.2 → **about $10.4-11.4** |
| Moves (either sign) | $10-13 | + s2300 C1 $3.4 + dpo pure-T $2.2 → **$13.8-14.8**; + T 0.8 $2.2 only if a power-sampling proposal is to be written |

The null path stays at or under about $10. The moves path goes about $4-5 over only when there is a finding to
interpret.

#### Required changes

1. Reach baseline: state the fresh T 1.0 rate (Stage 0c dpo 0 of 84 on 42 reach prompts; s0100 2 of 1,536 = 0.13 %),
   not "0 of 768" with a rule-of-three bound. Say that the reach set was selected on those 768 draws. **$0.**
2. Trigger: paired 95 % CI excluding 0, two-sided, with no +4 floor (MDE about 3.1 points instead of about 5.0). Keep
   the reach rule (≥ 3 passes from ≥ 2 prompts), phrased against the fresh rate. **$0.**
3. Any arm-A move (up or down) triggers both s2300 at C1, arm A × 2 (**$3.4**) and dpo pure-T 0.6 / top_p 1.0, arm A × 2
   (**$2.2**). No "temperature" or "sharpening" headline, and no power-sampling proposal, without the pure-T cell. All
   headlines say "card decoding". **Conditional $5.6.**
4. s0500 at C1, reach × 8, unconditional (**$2.8** including load).
5. Loop watchdog on in-flight state: tokens over 1.5× plan before 90 % of draws finish, or ≥ 10 % of submitted draws past
   16,384 tokens while running, with a 5-min in-flight length log. Repetition flag (report p. 108 thresholds) and
   no-`</think>` share in every table. **$0** (code).
6. dpo T 0.8, top_p 1.0, arm A × 2 only if C1 is null but cut or repetition share at C1 is ≥ 2% (degeneration may be
   masking a gain), or if a power-sampling proposal is to be written (monotonicity). This replaces Tier 2's
   automatic T 0.8. **Conditional $2.2.**
7. Shares: joint paired-bootstrap interval for share_RL; reach share against the fresh baseline; arm-A split into
   dpo-ever-passed (111) and never-passed (107). **$0.**
8. Prefill likelihood check, subsampled (8 stored draws per reach prompt from s2300 and from dpo, scored under dpo on
   the Tier 1 load), about 4.5 M prefill tokens. It is the scientific gate for any power-sampling proposal. **About
   $1-2.**
9. §8 corrections: Ji-style costs at least 65× tokens (at least $45 per arm A × 2 cell even with a guessed 3× throughput
   uplift), and H_t = 192 cannot see the IF verdict. Condition (i) is relabelled as Leon's budget rule, with Ji's
   DeepSeek-Math-7B row (low T 3 % against power sampling 78 % of the gap) cited as the counterexample. **$0.**
10. The new T / top_p wrapper reproduces Stage 0c's checks (template sha, every prompt-token sha256 equal to Dan's meta)
    and logs T and top_p in each job's meta. **$0.**

#### Points I would accept being left OPEN

1. Card setting versus pure T as the Tier 1 arm. I accept the card setting provided required change 3 holds.
2. Whether s0500 arm A × 2 (+$2.2) is added beside s0500 reach.
3. The exact likelihood-gate statistic (total versus per-token, percentile threshold). Also whether the check runs now or
   only before a power-sampling proposal, if vLLM 0.11's `prompt_logprobs` path proves non-trivial.
4. The dev / held-out split for any later setting choice (EDL pool 64 / 32 versus 88 / 88 of the non-reach arm A).
5. α, proposal and block size for a power-sampling follow-up, which is not approved.
6. Whether the card's IFBench numbers are strict or loose. This affects only the prior.
7. Provenance of Karan's GRPO models (trained by them or Shao et al.'s checkpoint). It does not affect the design.

## Turn 3 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 02:03 PT

Replies to Turn 2. Checks I ran for this turn (all read only): rendered Karan p. 11 as an image; counted dpo's Stage 0c reach rows
(`runs/dpo/rep/graded.jsonl`); recomputed the baseline split of arm A. Corrections to my notes are appended at the end of each
note under "Correction after audit"; Turn 1 is left as written.

### A. Reading disagreements

1. **Karan low-T tau.** Concede and verified. Fig. 6 right (p. 11) N_MCMC = 0 points are about 69.0 and 62.8 = Table 1's .690 and
   .628. tau = 0.25 is supported inside the paper.
2. **Fig. 6 bar labels.** Concede. They are printed (alpha = 1.0: 45.0 / 35.8 / 38.4 against Table 1 Base 49.6 / 49.8 / 40.0). My "could not read exact numbers" came from the text extraction only; I should have rendered the page. The inconsistency stands as in-paper evidence of large unexplained variation.
3. **Identical Base HumanEval/GPQA for two models (Table 1).** Concede; I reproduced it without comment.
4. **GRPO provenance.** Concede the wording. p. 8 says only that they "use the implementation of GRPO in Shao et al. [2025]"; Ji p. 28 names the released checkpoint. Neither says who trained it.
5. **Ji cost, p. 23 versus p. 28.** Concede in full. p. 28 is wall-clock per prompt ("We report wall-clock generation time per prompt"); the token cost is p. 23, O(sum[L_t B + K_t M_t H_t] + L_w B). With K = M = 8, H = B = 192, rollouts alone are 64 B per output block, so at least about 65x plain tokens (more, since L_t is unstated). At our 1,500 tok/s and 2.2 M plain tokens per arm-A x 2 cell that is at least about 143 M tokens, about 40; the auditor's "at least 5" assumes a guessed 3x throughput uplift for short prefix-sharing rollouts, which I cannot confirm either. Either way my ".5, pilot Ji first" was wrong and is reversed: at B = 1,024 Karan's MCMC (11x, about 4) is cheaper than Ji-style (at least 65x). Pilot MCMC first, if anything.
6. **Algorithm 2 weighting.** Agree the construction is in the text (candidates drawn from p, cut to top-K, weighted p^alpha zeta_hat, no division by the proposal). The p^(alpha+1) consequence follows from the usual resampling argument; I did not verify it numerically. Only matters for a follow-up.
7. **H_t = 192 cannot see the IF verdict.** Agreed, and I said so.
8. **DeepSeek-Math-7B row.** Concede. Ji Table 1: low T .362 to .366, 3 % of the .129 GRPO gap, while "Ours" .464 closes 78 %. My Turn 1 line "p^alpha is only interesting where low T gives partial recovery" is not supported; condition (i) "moves" is Leon's budget rule, not a result of the papers. Counter-point for balance (same table): on GPQA, DeepSeek low T (.430) beats Ours (.364) and GRPO (.333), so the papers' own numbers show both directions and no stable ordering.
9. **Card IFBench as a prior.** Concede. I compared an arm-A (218 prompt) number with a full-set number at a different harness, and the report's tables do not say strict or loose near Table 14 (p. 38; I did not read Table 24 on p. 52). "About 5 points" is withdrawn; the prior is "0 to about 6, metric unknown". I did not re-compute the auditor's 28.0/33.8 and 67.6/69.6 over the 300-prompt set.
10. **Interval: draw-dominated, not prompt-dominated.** Concede. The headline is a paired difference, and Stage 0c measured its width (2 fresh dpo draws vs 16 stored: -0.8, CI [-2.9, +1.3], half-width 2.1). My "SE about 2 from prompt heterogeneity" was wrong.
11. **Reach baseline.** Concede and verified: Stage 0c's dpo replication has 84 reach rows (42 prompts x 2) and 0 strict passes; the 768 stored draws define the reach set, so the rule-of-three bound from them is invalid for fresh draws. Fresh proxy: s0100 2 / 1,536 = 0.13 %. A side effect I checked: the 42 shared prompts were selected as 0 / 16 in the same stored draws that form the arm-A baseline, so a regression-to-the-mean upward bias exists in arm A; at the fresh rate 0.13 % it is about 42/218 x 0.13 % = 0.03 points, negligible.
12. **s0500 baseline.** Agreed, resolved. I did not re-verify that `T_s0500` used identical decoding beyond what `test_stage0.py` checks.
13. **New from this check (not raised by the auditor).** The auditor's split of arm A into dpo-ever-passed (111) and never-passed (107) is defined on the same 16 draws that form the baseline, which builds in regression to the mean (the ever-passed group's baseline is selected high, the never-passed group's low). Fix below: define the split on draws 1-8, baseline from draws 9-16. Counts: 100 prompts with at least one pass in draws 1-8 (baseline from draws 9-16: 61.5 %) and 118 with none (baseline 1.4 %).

### B. Required changes: accept or argue

| # | Change | Reply |
|---|---|---|
| 1 | Fresh reach baseline (0/84; 0.13 % at s0100) | Accept. |
| 2 | Trigger = paired two-sided 95 % CI excludes 0, no +4 floor | Accept. Stage 0c shows no pipeline offset, so a floor counts it twice. Effects below about +4 are labelled "detected, small" (under 10 % of the gap). |
| 3 | Any significant arm-A move (either sign) triggers s2300 at C1 and dpo pure-T (0.6, top_p 1.0) | Accept. All headlines say "card decoding" until the pure-T cell exists. |
| 4 | s0500 at C1, reach x 8, unconditional | Accept (the only cell where sharpening can multiply partly learned behaviour; s0500 had 4.5 % of reach draws passing at T 1.0). It is the second cut candidate under a 0 cap, see section F. |
| 5 | Loop watchdog on in-flight state, repetition flag, no-`</think>` share | Accept; specified below. Implementation caveat: I have not re-read Dan's `sample.py` this turn; if it submits all prompts in one blocking `generate` call, the wrapper must drive the engine step loop. Unverified. |
| 6 | T 0.8 only if C1 null with cut or repetition share at least 2 %, or if a power-sampling proposal is to be written | Accept. |
| 7 | Joint-bootstrap share; reach share against fresh baseline; ever/never split | Accept, with the split redefined (A13). |
| 8 | Prefill likelihood check on the Tier 1 load, gating the power-sampling follow-up | Accept; specified below (what is scored, statistic, gate). |
| 9 | Section 8 corrections | Accept (A5, A8). |
| 10 | Wrapper reproduces Stage 0c checks, logs T and top_p | Accept. |

Where I still argue: (i) the likelihood gate's thresholds are guesses and I say so (OPEN 3); (ii) s0500 arm A x 2 is left out (OPEN 2), because reach, not arm A, is the question at s0500; (iii) the Ji-style pilot is dropped, not just corrected.

### C. Revised design in full

**Common to all cells.** Dan's sampler (vLLM 0.11.0, `olmo_thinker` template, bf16, max 32,768 tokens, prefix caching) through a new wrapper (Dan's files stay unedited) that takes temperature and top_p, logs both in the job meta, and reproduces Stage 0c's checks (template sha; every prompt-token sha256 equal to Dan's meta). Draw order kept. Strict checker, "checker passes". Reach prompts that are also in arm A (42 of 48) are run once at their reach draw count; arm-A scoring uses their first 2 draws.

**Cells.**

| Id | Checkpoint | T / top_p | Prompts x draws | Draws | Status |
|---|---|---|---|---|---|
| C1 | dpo | 0.6 / 0.95 (card) | 176 arm-A non-reach x 2, 42 shared x 8, 6 reach-only x 8 | 736 | always |
| S5R | s0500 | 0.6 / 0.95 | 48 reach x 8 | 384 | always |
| PF | prefill scoring under dpo, same load as C1 | n/a | reach prompts: 8 stored s2300 + 8 stored dpo + 8 C1 draws each | 1,152 scored | always |
| C3 | s2300 | 0.6 / 0.95 | 218 arm A x 2 | 436 | if trigger A |
| C4 | dpo | 0.6 / 1.0 (pure T) | 218 arm A x 2 | 436 | if trigger A |
| C2 | dpo | 0.8 / 1.0 | 218 arm A x 2 | 436 | if trigger B |
| S5A | s0500 | 0.6 / 0.95 | 218 arm A x 2 | 436 | not planned (OPEN 2) |

**Triggers, written before any data.**
- **Trigger A** (run C3 and C4): the paired arm-A pass@1 difference at C1, mean over prompts of (mean of 2 fresh draws minus mean of all 16 stored dpo T 1.0 draws), prompt bootstrap (10,000 resamples, seed 0), two-sided 95 % interval excludes 0, either sign; or reach rule: at least 3 passes from at least 2 prompts among the 384 dpo reach draws C1 (expected under the fresh 0.13 % rate: 0.5).
- **Trigger B** (run C2): C1 arm-A interval includes 0 AND (share of C1 draws cut at the cap, or flagged repetitive, is at least 2 %); OR a power-sampling proposal is to be written (monotonicity point).
- **Stop-and-report**: if neither trigger fires, report the null at card decoding and spend nothing more. A null here bounds the arm-A effect at about the MDE (about 3.3 points at 80 % power; paired SE about 1.2), which is about 8 % of the 39.3-point gap, and says nothing about p^alpha.
- **Power-sampling proposal** may be written only if all hold: (i) Leon's budget rule: trigger A fires with a positive arm-A move at dpo; (ii) share_RL (below) between 10 % and 70 %, and dpo C1 still at least 10 points below s2300; (iii) cut-at-cap below 5 % at C1 and the lift persists among finishing draws; (iv) C2 sits between T 1.0 and C1; (v) the PF gate passes. (i) is a budget rule, not implied by the papers (DeepSeek-Math-7B, A8); (v) is the scientific gate.

**Prefill likelihood check (PF), exactly.**
- *Model:* Olmo-3-32B-Think-DPO (dpo), bf16, raw log-probabilities at temperature 1.0 over the full vocabulary (no top_p, no temperature scaling), response tokens only (prompt tokens conditioned on, not counted). Whether vLLM 0.11's `prompt_logprobs` returns raw (not post-processed) values and works with prefix caching on is unverified; sanity test before use: score 3 stored dpo T 1.0 draws and compare sum of log-probabilities with the cumulative log-probability a generation run reports for the same tokens (they must agree up to bf16 noise); if not, disable prefix caching for this job.
- *What:* for each of the 48 reach prompts, 8 stored s2300 responses (first 8 of the 32 in draw order), 8 stored dpo responses (first 8 of the 16), and 8 fresh dpo C1 responses (the same job). About 1,152 responses, about 6.5 M prefill tokens (384 x about 7.0 k + 384 x about 4.7 k + 384 x about 5.1 k including prompts, my estimate). The stored token ids are base64 int32 in Dan's samples; whether they are response-only must be checked, and prompt ids rebuilt from the template with the sha check.
- *Statistic:* per response, per-token mean log p_dpo over response tokens (primary; totals are length-confounded since s2300 responses are about 50 % longer) and the same split at `</think>` into thinking and answer. Within each prompt compare the s2300 passing responses with that prompt's 8 dpo T 1.0 responses. Primary summary S = median over prompts having at least one s2300 pass of [mean per-token LL of s2300 passing responses minus mean per-token LL of dpo T 1.0 responses], prompt-bootstrap CI. Secondary: P_low = share of s2300 passing responses whose per-token LL is below the minimum of that prompt's 8 dpo responses (about 11 % expected if no shift, with 9 exchangeable values). Reference scales: the same statistic for s2300 failing responses, and for dpo C1 responses (how far T 0.6 shifts LL by itself).
- *Gate (thresholds are guesses):* "passing responses sit in dpo's low tail" if P_low is at least 50 %; "not in the tail" if P_low is at most 25 % and S is no lower than the dpo C1 shift's negative; between, unresolved and the proposal needs a pilot. It gates only the power-sampling proposal, not Tier 2.
- *Cost:* prefill about 4-8 k tokens/s (a guess for 32B bf16 on one H200), 14-27 min, about .3-2.4 on the C1 load (no extra load).

**Loop watchdog (in-flight state).** The wrapper runs the engine step loop (or equivalent) and every 5 min logs: running, finished, waiting counts, a histogram of in-flight output lengths, total generated tokens. Abort the cell if (a) total generated tokens (finished plus in-flight) exceed 1.5x the cell's plan (draws x 5,100 for dpo, x 5,640 s0500, x 6,900 s2300) before 90 % of draws have finished, or (b) at least 10 % of submitted draws are past 16,384 tokens while still running. An aborted cell is reported as aborted, with its finished draws graded and its unfinished draws counted as failures; it is not dropped. Every table also reports: share finish=length, share without `</think>`, an n-gram repetition flag on all draws using the report's own thresholds (p. 108 per the auditor: "heavily repeated (10x+) sentences, paragraphs, or (50x+) phrases"; I have not read p. 108), token mean / p50 / p90 / p99, and pass@1 among finished draws as secondary. Headline pass@1 counts loops as failures.

**Analysis definitions.**
- Unit = prompt. B_p = mean strict over all stored T 1.0 draws (16 dpo; 8 s0500; 8 s2300); N_p = mean over the cell's first 2 draws (arm A) or all draws (reach).
- Delta(c) = mean_p(N_p - B_p) over the 218 arm-A prompts; prompt bootstrap, 10,000 resamples, seed 0, paired; same resamples for every statistic below.
- **share_RL** = Delta(dpo, C1) / D, with D = mean_p(B_p^{s2300} - B_p^{dpo}) = 39.3 points, numerator and denominator in one joint bootstrap (the ratio's interval is reported, not a fixed denominator).
- **Matched gap** G(C1) = mean_p(N_p^{s2300,C1} - N_p^{dpo,C1}) against G(1.0) = 39.3; reported beside share_RL, never averaged with it. Only if C3 runs.
- **Reach:** r(c) = share of draws that pass, prompt bootstrap over 48 prompts. dpo: r(C1) against the fresh baseline 0/84 (0.13 % at s0100 as proxy); share_reach = (r_dpo,C1 - 0.13 %) / (69.4 % - 0.13 %), joint bootstrap. s0500: r(S5R) against its stored 4.5 % (32 draws per prompt), and "any pass in 8" against the first 8 stored draws.
- **Split to separate "sharpening where mass exists" from "creating mass":** define groups on dpo stored draws 1-8 (100 prompts with at least one pass, 118 with none), baseline from draws 9-16 only (61.5 % and 1.4 %); the groups do not share draws with the baseline, which removes regression to the mean. Secondary analysis.
- Secondary: pass@2 (first two draws), baseline-rate bins, constraint-family breakdown, pass@1 among finishing draws.
- **MDEs reported with every null** (80 % power, paired): about 3.3 at dpo arm A, about 4.4 at s0500 arm A, about 4.7 at s2300 arm A (the auditor's computations from stored within-prompt variances; not re-derived by me). Reach at 8 draws: 0 of 384 bounds the per-draw rate near 0.8 % one-sided (unclustered).
- No setting is chosen from data in these cells. If one must be (T for a later cell, alpha), choose on a fixed random half of the 176 non-reach arm-A prompts (88, seed fixed in advance) and report on the other 88 plus reach; the EDL pool's 32 test prompts give about +-15 point intervals, too wide. OPEN 4.

### D. Cost table (1,500 tok/s, $5.29/h so about $0.98 per M tokens; 8 min load = about $0.7; length 1.13x the stored mean, a guess)

| Item | Draws | Tokens (M) | H200 min | $ |
|---|---|---|---|---|
| C1 (always) | 736 x 5,100 | 3.75 | 42 + 8 load | 4.4 |
| S5R (always) | 384 x 5,640 | 2.17 | 24 + 8 load | 2.8 |
| PF (always, on C1 load) | about 6.5 M prefill | n/a | 14-27 | 1.3-2.4 |
| **Always-run total** | | | | **about 8.5-9.6** |
| C3 (trigger A) | 436 x 6,900 | 3.0 | 33 + 8 | 3.4 |
| C4 (trigger A, reload) | 436 x 5,100 | 2.2 | 25 + 8 | 2.9 |
| C2 (trigger B, reload) | 436 x 5,100 | 2.2 | 25 + 8 | 2.9 |
| Trigger A total | | | | about 6.3 |

With 5 % loops (mean 5,900) the always-run total is about $9.5-10.5; with 15 % loops (mean 8,700) it would be about $12-14, which the watchdog (b) aborts before it is reached. If trigger A fires: about $15-16 in all; if B too: about $18.

### E. What each outcome shows and does not show (changes from Turn 1 only)

- A significant arm-A move at dpo C1 is labelled "card decoding", not "temperature" or "sharpening", until C4 exists. Without C3 a positive move cannot be told from a generic decoding effect (Ji Table 2: RL models can drop at low T).
- A reach null at dpo C1 is close to guaranteed by construction (fresh rate about 0.1 %); it shows nothing about sharpening. S5R is where sharpening can show: an increase over 4.5 % means card decoding multiplies a partly learned behaviour; a decrease means early RL's passing responses sit in s0500's tail (Ji Table 2 pattern).
- PF is the only direct test of the sharpening premise on our models; a "low tail" result makes an arm-A and reach null the expected result for any alpha > 1 sampler of dpo, a "not in the tail" result makes the null a puzzle that points to a sequence-level sampler.

### F. If capped at $10 for all of M1: what I would cut, in order

The always-run set is about $8.5-9.6 on paper and has no headroom for loops, and nothing conditional is funded. Cuts:
1. **dpo reach draws 8 to 4 per prompt (C1 reach: 192 instead of 384; saves about $1.0).** Costs: the fresh-rate bound on dpo reach rises from about 0.8 % to about 1.6 %, and the reach trigger becomes at least 2 passes from at least 2 prompts. Arm A is unaffected. Least interpretability lost, since dpo reach is near zero by construction.
2. **PF subsample 8 to 4 stored responses per prompt per group (saves about $0.9).** Costs: the P_low and S intervals widen (4 responses per group per prompt gives a coarser "minimum" comparison, expected-no-shift P_low 20 % instead of 11 %); the gate becomes weaker.
3. **S5R draws 8 to 4 per prompt (saves about $1.0).** Costs: at 4.5 % the cell expects only about 9 passes in 192 draws, so only a large change is detectable.
4. Last resort, drop S5R entirely: the experiment then cannot test "sharpening multiplies a partly learned behaviour", i.e. the checkpoint-path question; only dpo remains, where reach is null by construction.
After cuts 1-2 the always-run total is about $6.6-7.7, leaving about $2.3-3.4 for loops. The trigger-A cells (about $6.3) cannot fit under $10; under a hard cap, run C3 only ($3.4) and keep the headline at "card decoding", rather than C4.

### OPEN (still disputed or undecided)

1. Tier 1 arm = card decoding (T 0.6 and top_p 0.95 together). I accept the auditor's condition (C4 mandatory before a "temperature" headline); not resolved is whether a card-decoding null is the stronger null (the auditor's view: it is the sharper setting) or just a different setting from the papers' pure temperature.
2. s0500 arm A x 2 (+$2.2): not planned; the auditor would accept leaving it out; I leave it out.
3. The PF gate thresholds (P_low at least 50 % / at most 25 %), per-token versus total log-likelihood as the statistic, and whether PF runs now or only before a power-sampling proposal. Whether vLLM 0.11 `prompt_logprobs` is raw and compatible with prefix caching is unverified.
4. Dev/held-out split for any later choice (88/88 of the non-reach arm A, versus the EDL pool's 64/32).
5. alpha, proposal temperature and block size of a power-sampling follow-up (alpha = 1.67 has no direct support in Karan: Fig. 6 sweeps alpha in {1, 2, 4, 10, inf}); and whether a pilot is warranted at all, since the cost of any variant at our lengths is at least about $24 per arm-A x 2 cell (MCMC, B = 1,024) and Ji-style at least about $45-140.
6. Whether the card's IFBench numbers are strict or loose (prior only).
7. Provenance of Karan's GRPO checkpoint (does not affect the design).
8. Whether "cut share or repetition share at least 2 %" is the right trigger B, and whether the 1.5x-plan / 16,384 abort thresholds are right; both are guesses without any low-T data on these models.

## Turn 4 — auditor (Claude Opus 5.5), 2026-11-27 02:09 PT

Round 2. Checks I ran for this turn (all read only, CPU):
- Recomputed the draws 1-8 / 9-16 split from Dan's `F_dpo_ifb` rows, sorted by (chunk, i).
- Recomputed the s0500 reach-cell resolution through `dan_curve`.
- Read Dan's `olmo_ifladder/sample.py` lines 176-365, i.e. his engine loop.
- Read the vLLM 0.11.0 source copy in `metrics2/ext/vllm_v0.11.0/vllm/`: `v1/worker/gpu_model_runner.py` lines 2720-2816,
  `v1/engine/processor.py` lines 70-92, and `v1/engine/llm_engine.py` lines 257-338. The copy is partial, at 32 files;
  `sampler.py`, `kv_cache_manager.py`, the scheduler, `output_processor.py` and `metrics/loggers.py` are **not** in it.
- Decoded the first rows of Dan's stored samples.
- Ran a crude repetition detector over our Stage 0c fresh T 1.0 draws (`runs/{dpo,s2300}/rep/samples.jsonl.gz`).

### 0. Record integrity: mangled dollar amounts in Turn 3

Several dollar figures in Turn 3 lost their leading digits. They look like shell expansion of `$1`, `$2`, `$4` and `$6`
as empty positional parameters:

| Turn 3 line | Text as written | Probably meant |
|---|---|---|
| 377 | "at least about 143 M tokens, about 40" | about $140 |
| 377 | "the auditor's 'at least 5'" | $45 |
| 377 | "my '.5, pilot Ji first'" | $6.5 |
| 377 | "(11x, about 4)" | $24 |
| 394 | "under a 0 cap" | $10 |
| 431 | "about .3-2.4" | $1.3-2.4 |

Turns are never edited, so Turn 5 should carry a dated erratum restating them. Future turns should be written without
an unquoted heredoc. My own turns were written with the Edit tool and are intact.

### 1. My ten required changes, as implemented in Turn 3 §C

1. **Fresh reach baseline: implemented.** "dpo: r(C1) against the fresh baseline 0/84 (0.13 % at s0100 as proxy)"
   (§C, Analysis). "(expected under the fresh 0.13 % rate: 0.5)" (Trigger A).
2. **Trigger is a two-sided CI with no floor: implemented.** "two-sided 95 % interval excludes 0, either sign". The
   addition "Effects below about +4 are labelled 'detected, small'" (§B, row 2) is fine.
3. **Either-sign move runs C3 and C4; no temperature headline or power-sampling proposal without C4: partly
   implemented.**
   - C3 and C4 are both "if trigger A". The label rule is there: "labelled 'card decoding', not 'temperature' or
     'sharpening', until C4 exists" (§E).
   - **Missing:** the power-sampling conditions (i)-(v) in §C do not require C4. Condition (iv) reads "C2 sits between
     T 1.0 and C1", which compares pure T 0.8 with card decoding (T 0.6 *and* top_p 0.95), so it mixes top_p into a
     monotonicity test.
   - §F also says "under a hard cap, run C3 only", and then nothing stops a proposal from being written without C4.
   - Still required (R3 below).
4. **s0500 reach × 8, unconditional: implemented.** "S5R | s0500 | 0.6 / 0.95 | 48 reach x 8 | 384 | always".
5. **In-flight watchdog, repetition flag, no-`</think>` share: specified, but not implementable as written.** See §2c.
6. **T 0.8 conditional: implemented.** "Trigger B (run C2): C1 arm-A interval includes 0 AND (share of C1 draws cut at
   the cap, or flagged repetitive, is at least 2 %)". The threshold needs a baseline; see §2d.
7. **Joint bootstrap, reach share against the fresh baseline, ever/never split: implemented.**
   - "numerator and denominator in one joint bootstrap (the ratio's interval is reported, not a fixed denominator)".
   - "share_reach = (r_dpo,C1 - 0.13 %) / (69.4 % - 0.13 %), joint bootstrap".
   - The split was redefined (§2a).
8. **Prefill check: implemented in outline; three faults.** See §2b.
9. **§8 corrections: implemented.** "(i) is a budget rule, not implied by the papers (DeepSeek-Math-7B, A8)". The Ji
   pilot is dropped (§B). The dollar figures in A5 need the erratum (§0).
10. **Wrapper checks: implemented.** "reproduces Stage 0c's checks (template sha; every prompt-token sha256 equal to
    Dan's meta)" and "logs both in the job meta". One implementation fact supports this. Dan's `submit()` reads the
    module globals at call time (`SamplingParams(n=m, temperature=TEMPERATURE, top_p=TOP_P, …)`, sample.py line 279), and
    the meta logs them (line 349). So a wrapper that imports his module, sets `TEMPERATURE` and `TOP_P`, and calls `main()`
    needs no edit to his file. Whatever the wrapper does, its meta must show 0.6 / 0.95.

### 2. Attack on what is new

#### 2a. Regression-to-the-mean fix for the ever/never split: verified and accepted

I recomputed from `F_dpo_ifb` in (chunk, i) order and got the same result:

| Group (from draws 1-8) | Prompts | Baseline from draws 9-16 | Same group from draws 1-8 |
|---|---|---|---|
| At least one pass | 100 | 61.5 % | 63.2 % |
| No pass | 118 | 1.4 % | — |

All 42 reach prompts fall in the never-passed group. The ever-passed group's own draws 1-8 give 63.2 %, against 61.5 % from
draws 9-16, so the selection inflation the fix removes is about 1.7 points. The fix is correct.

The split costs resolution, which must be stated next to the result. Using 8 baseline draws and 2 new draws, the paired
SE is about 2.4 points in the ever group (within-prompt variance 0.093), so the MDE there is about 6.8. In the never group
it is about 0.8 (variance 0.012). Keep it secondary, as Turn 3 has it.

The residual bias in the headline Delta (42 reach prompts selected at 0 of 16 within the 16 baseline draws) is about
0.03 points, as Turn 3 says. It is negligible.

#### 2b. Prefill likelihood check as specified

**What is scored, and the token ids.** Verified.
- Dan's rows carry `ids` (base64 int32) that are **response-only**: `n_tok == len(ids)`, ending in EOS 100257 when
  `finish == "stop"`. I decoded one row each of `B_s2300` and `olmo_ifomega/F_dpo_ifb`. Dan's docstring line 36 says the
  same.
- `prompt_tokens` is stored, so rebuilding prompt ids from the template and checking their sha256 against Dan's meta is
  possible.
- Draws cut at the cap have no EOS. Score the EOS token consistently (always include it when present) and report per-token
  means with cut draws excluded.

**Under which model, and raw or processed log-probabilities.** What I could verify in the vLLM 0.11.0 copy:
- `_get_prompt_logprobs_dict` takes the prompt positions' hidden states, calls `self.model.compute_logits(...)`, then
  `self.sampler.compute_logprobs(logits)` and `gather_logprobs(...)` (gpu_model_runner.py lines 2785-2795).
- No temperature, top_p or penalty step sits on this path; those act in the sampler's forward pass on sampled positions.
  So prompt log-probabilities come from the raw logits whatever the request's sampling parameters.
- **Not verifiable from the copy.** `sampler.py` is absent, so I cannot see whether `compute_logprobs` depends on
  `logprobs_mode` (which is passed to `Sampler(...)`, line 255). My recollection is a plain fp32 `log_softmax`, unverified.
- PF requests should in any case set temperature 1.0 and top_p 1.0. Then raw and processed coincide on every path.

**Prefix caching.** Partly verified, and it is a real risk.
- The prompt-logprob loop fills slices only from `start_idx = request.num_computed_tokens` onward (line 2758) into a
  tensor created with `LogprobsTensors.empty_cpu(...)` (line 2753).
- If a prefix-cache hit set `num_computed_tokens > 0`, the cached positions' log-probabilities would never be written and
  would hold uninitialised memory, with no error raised.
- Correctness therefore depends on the KV-cache manager refusing cache hits for requests with `prompt_logprobs`. I believe
  vLLM v1 does this, but `kv_cache_manager.py` is not in the copy, so it is **unverified**.
- PF cannot run inside Dan's sampler anyway, which requests no log-probabilities. It needs its own `LLM(...)`, so it
  should set `enable_prefix_caching=False`. The 24 responses per prompt share only a 100-200-token prompt prefix, so
  nothing is lost.

**Other engine facts from the copy.**
- `processor.py` validates `prompt_logprobs` only `if params.prompt_logprobs:` (line 85), so 0 is not rejected. Whether
  the runner then registers a request with 0 depends on `input_batch` (not in the copy). Use `prompt_logprobs=1`, which
  returns the target token plus top-1 at no real cost, unless a probe shows 0 works.
- Memory: each prefill chunk computes `[num_logits, 100,278]` logits and an fp32 log-softmax, about 0.4 MB per token, so
  about 3.3 GB transient for an 8,192-token chunk. It is a guess that vLLM's start-up memory profiling does not reserve
  for it. Set `max_num_batched_tokens` to about 4,096 and `gpu_memory_utilization` ≤ 0.85 for PF.
- **Cost correction.** "On the C1 load (no extra load)" is not quite right, because PF is a separate process with a new
  `LLM(...)`. The weights are already on the pod's disk, so this is a reload, not a download: about 2-5 min, about
  $0.2-0.4 (guess).

**The sanity test as written cannot be run.** "Compare sum of log-probabilities with the cumulative log-probability a
generation run reports" needs generation log-probabilities, and Dan's stored draws have none: `sample.py` requests no
`logprobs`, and the rows are `n_tok, finish, text, ids`. Replacement, on the PF engine, a few minutes:
1. Generate 3 draws at T 1.0 / top_p 1.0 with `logprobs=0` and max_tokens 2,048.
2. Re-score each prompt+response with `prompt_logprobs`. The mean per-token difference must be under 0.01 nats (bf16
   batch noise).
3. Score one stored response twice in the same engine and check that every position is finite and identical.

**The statistic and the gate.** Two faults and one gap.
- *Direction fault.* "Not in the tail if P_low is at most 25 % and S is no lower than the dpo C1 shift's negative" passes
  s2300 responses that sit *below* dpo's typical per-token LL, by up to the size of the T 0.6 shift. Any α > 1 sampler of
  dpo moves mass toward *higher* LL, so targets below dpo's typical draws are moved away from.
  - **Required:** "not in the tail" needs S ≥ 0 at the point estimate, or at least a CI upper bound ≥ 0, and P_low ≤ 25 %.
  - The C1 shift is then a descriptive scale only. Scoring the C1 draws is not needed for the gate (see §2e for the cost
    saving).
- *Per-token versus total.* p^α acts on the **total** sequence likelihood. s2300's reach responses are about 50 % longer
  (6,863 vs 4,520 mean tokens), so a per-token "not in the tail" says nothing about whether p^α of dpo would put mass on
  them.
  - Per-token is the right *descriptive* statistic (Karan Fig. 4 is length-averaged). The *gate* also needs a
    length-adjusted total. Within each prompt, regress total LL on length over dpo's 8 responses and report where s2300's
    passing responses fall in the residuals.
  - **Required:** the gate passes only if both the per-token and the length-adjusted total comparisons clear it. Code
    only, $0.
- *Gap.* Compute S and P_low on the **answer segment** (after `</think>`) as well, as a pre-stated secondary. The
  checker's verdict depends only on the answer. A gate driven by thinking-trace style could pass or fail for reasons
  unrelated to constraint compliance. Code only, $0.
- The thresholds themselves (50 % / 25 %) can stay OPEN.

#### 2c. The in-flight watchdog: can it be built outside Dan's loop?

Facts from `sample.py` (lines 277-337):
- Requests are `SamplingParams(n=m, …, output_kind=RequestOutputKind.FINAL_ONLY)` with m = `--chunk` draws per request.
- The loop writes rows only for `o.finished` requests, and his 60-s progress line counts finished tokens only (`gen_tok`).
- So (i) wrapping `step()` sees nothing in flight, because FINAL_ONLY returns finished requests only, and (ii) a request
  with m = 8 (one reach prompt) writes **nothing** until its slowest draw ends. One looping draw holds 7 finished siblings
  hostage. An abort then loses them, and Turn 3's rule "unfinished draws counted as failures" would count them as failures.

What is available outside the loop:
- Dan constructs `LLM(..., disable_log_stats=False)` (line 264).
- `LLMEngine.step()` records scheduler and iteration stats and calls `do_log_stats_with_interval()` every
  `envs.VLLM_LOG_STATS_INTERVAL` seconds (llm_engine.py lines 278-284 and 332-338; I believe the default is 10 s).
- The logger's line format (average generation throughput, running and waiting counts) is in `metrics/loggers.py`, which
  is not in the copy, so it is **unverified**. Check it on the first minute of a pod log.

Given those lines, an external watchdog can do the following:
- (a) Integrate the logged generation throughput over time. That gives total generated tokens G(t), finished plus in
  flight, to about 10-s granularity. Abort if G(t) > 1.5 × plan before 90 % of draws are written. **Implementable.**
- (b) "≥ 10 % of submitted draws past 16,384 tokens while running" is **not observable** from outside, because
  per-request lengths are not logged. Replace it with a proxy:
  - waiting = 0, running ≥ 10 % of the cell's draws, and mean in-flight length (G − F)/running ≥ 16,384, where F is the
    finished tokens from Dan's progress line;
  - plus a STALL rule: no row written for 20 min while running > 0.
- An in-process route also exists without editing Dan's file. The wrapper could monkeypatch
  `vllm.v1.engine.llm_engine.LLMEngine.step` and read `self.get_metrics()` (lines 316-318) or the output processor's
  per-request states. It relies on internal APIs that are not in the copy, so I would not require it.
- **Required (code only, $0):** run C1, S5R, and any C2 / C3 / C4 with `--chunk 1`, or 2 for arm A. Rows are then written
  as each draw finishes, the finished count is real, and an abort loses only in-flight draws. Prefix caching keeps the
  repeated prompt prefill cheap. Draw identity is unchanged ("first 2 draws" = chunks 0 and 1), and every other line of
  Dan's sampler is unchanged.

#### 2d. Trigger B

- The 2 % absolute threshold has no stated baseline, and the repetition detector is not yet specified. Turn 3 quotes the
  report's thresholds via me ("I have not read p. 108"). The open-instruct implementation is not available here, so
  "phrase" is undefined.
- My crude detector flags a response if, among items of at least 20 characters, a paragraph or sentence repeats 10 or
  more times, or a word 10-gram repeats 50 or more times. On our fresh T 1.0 Stage 0c draws it flags **0.7 % at dpo**
  (3/436) and **2.8 % at s2300** (12/436). Using word 5-grams instead of 10-grams gives the same rates, and the
  sentence-repeat rule does almost all the flagging.
- So an absolute 2 % would already fire at s2300's T 1.0 baseline. At dpo it is about 1.3 points over baseline, which is
  about 2.5 SE at 736 draws, so it is reasonable there.
- **Required ($0, CPU):**
  1. Freeze the detector in code before any data.
  2. Run it on the stored T 1.0 draws of each checkpoint used. The Stage 0c draws or Dan's arm-A samples will do.
  3. Exclude IFBench "repeat"-family prompts from the flag, or apply it to the thinking segment only, because those
     prompts ask for repetition.
  4. Define trigger B relative to the same checkpoint's T 1.0 rate: C1 share ≥ T 1.0 share + 1.5 points. At dpo that is
     about 2.2 % with my detector, close to Turn 3's number, but now defined.

#### 2e. Cost and cut order under a $10 cap

Turn 3's always-run set is $8.5-9.6 before loops. It omits the PF reload (+$0.2-0.4), and it pays for scoring the C1
draws, which the corrected gate does not need. My order:

1. **C1 reach draws 8 → 4 per prompt** (544 draws instead of 736; −$1.0). Same as Turn 3's cut 1, and its trigger change
   (≥ 2 passes from ≥ 2 prompts) is fine. Arm A is untouched, since the 42 shared prompts keep at least 2 draws.
2. **Drop the C1-draw group from PF** (−about 1.96 M prefill tokens, −$0.4-0.7), keeping 8 s2300 + 8 dpo responses per
   reach prompt. I prefer this to Turn 3's cut 2 (thinning those groups to 4). With 4 dpo comparators the no-shift P_low
   rises from 11 % to 20 %, which blunts the gate exactly where it matters.
   - **Always-run after cuts 1-2:** C1 $3.4 + S5R $2.8 + PF $1.1-2.0 (4.5 M prefill at a guessed 4-8 k tok/s, plus
     reload) = **$7.3-8.2**, leaving $1.8-2.7 for loops.
3. **Make PF conditional on C1's actual tokens.** Run it right after C1 only if C1 generated ≤ 1.15 × plan. Otherwise defer
   it to the trigger-A path, where it rides C4's dpo load, and report "PF not run" in a null.
   - PF gates only a power-sampling proposal, and a proposal requires trigger A to fire positive (Leon's rule (i)).
   - In the null path PF is interpretive only, so this is the right thing to defer before thinning S5R.
4. **S5R 8 → 4** (−$1.0). From the stored s0500 reach draws (rate 4.5 %, 21 of 48 prompts with any pass, mean
   within-prompt variance 0.040), the paired SE against the 32 stored draws is about 1.1 points at 8 draws (MDE about 3.2)
   and about 1.5 at 4 draws (MDE about 4.3). That is still a usable test of a doubling or halving.
5. **Drop S5R** (last resort; Turn 3 agrees).

**Worst case.** If both C1 and S5R run to the 1.5× abort threshold, C1 + S5R ≈ $8.7 (C1 ≈ $4.8, S5R ≈ $3.9); with PF
deferred by rule 3 the path stays under about $9.

**Trigger A.** C3 + C4 = $6.3 cannot fit under $10 after any cut. I disagree with silently capping to "C3 only". A fired
trigger A is a finding, and CONTEXT allows about $40 across the three measurements, uncertain by a factor of 2, with a
hard stop at $80. **Required:** if trigger A fires, the agent stops and asks the orchestrator for the $6.3 with the C1
result in hand. Only if refused does it run C3 alone ($3.4), with the "card decoding" label and no power-sampling
proposal (R3).

#### 2f. Points from my Turn 2 that I drop

- Unconditional s2300 at the matched setting: dropped; trigger A covers it.
- Pure-T as the Tier 1 arm: dropped while R3 holds.
- s0500 arm A × 2: left out, as Turn 3 proposes.

#### Required changes (still required after Turn 3)

1. **R1. Erratum for the mangled dollar amounts** in Turn 3 (lines 377, 394, 431; §0), as a dated note in Turn 5. $0.
2. **R2. PF engine and tests.**
   - Its own `LLM(...)` with `enable_prefix_caching=False`, T 1.0 / top_p 1.0, `prompt_logprobs=1` (0 only if a probe
     shows it works), `max_num_batched_tokens` about 4,096, `gpu_memory_utilization` ≤ 0.85.
   - Replace the impossible sanity test with the three-step test in §2b.
   - Budget the reload (+$0.2-0.4). $0 otherwise.
3. **R3. Power-sampling conditions.**
   - Add "(vi) C4 exists and shows the arm-A move at pure T 0.6".
   - Change (iv) to "C2 (pure T 0.8) lies between stored T 1.0 and C4 (pure T 0.6)".
   - No proposal on a C3-only path. $0.
4. **R4. PF gate direction and length.**
   - "Not in the tail" requires S ≥ 0 (point estimate; CI upper bound ≥ 0) and P_low ≤ 25 %, on per-token *and*
     length-adjusted total LL.
   - Answer-segment S and P_low as a pre-stated secondary.
   - The C1 shift is descriptive only. $0.
5. **R5. Watchdog built from what is observable.**
   - `--chunk 1` (arm A may use 2) for every new cell.
   - Abort (a) from integrated vLLM generation throughput against plan.
   - Abort (b) via the proxy (waiting 0, running ≥ 10 % of draws, mean in-flight length ≥ 16,384), plus STALL (no row
     for 20 min while running > 0).
   - Verify the vLLM stats-log format in the first minute of the first job. $0.
6. **R6. Trigger B defined against a baseline.**
   - Freeze the repetition detector.
   - Measure its T 1.0 rate on stored draws of each checkpoint.
   - Exclude the repeat family, or apply the flag to the thinking segment only.
   - Trigger at C1 share ≥ T 1.0 share + 1.5 points. $0.
7. **R7. Cut order under the cap.**
   - (1) C1 reach 8 → 4; (2) drop PF's C1 group; (3) PF conditional on C1 tokens ≤ 1.15 × plan, else deferred to the
     trigger-A path; (4) S5R 8 → 4; (5) drop S5R.
   - Always-run after cuts 1-2: about $7.3-8.2.
8. **R8. Trigger A escalates.** If it fires, ask the orchestrator for C3 + C4 ($6.3) before spending. Fall back to C3 only
   if refused.

#### Points I would accept being left OPEN

1. PF thresholds (P_low ≥ 50 % / ≤ 25 %) and the exact length adjustment (within-prompt regression or length bins).
2. Whether vLLM 0.11's prompt-logprob path is exactly raw fp32 `log_softmax` and refuses prefix-cache hits. `sampler.py`
   and `kv_cache_manager.py` are not in the copy. R2's settings and probe make it harmless either way.
3. The 1.5× plan and 16,384-token abort thresholds, and the trigger-B margin (+1.5 points). No low-T data exist on these
   models.
4. Card setting versus pure T as the Tier 1 arm (Turn 3 OPEN 1), provided R3 holds.
5. The dev / held-out split for any later choice (88 / 88 non-reach arm A is agreed in Turn 3; fine).
6. α, proposal and block size for a power-sampling follow-up, and whether a pilot is warranted at all.
7. The card's strict or loose metric, and Karan's GRPO provenance (prior only).

## Turn 5 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 02:13 PT

Written with a quoted heredoc, so every dollar sign below is literal. Part 1 replies to Turn 4; Part 2 is the erratum; Part 3 is the FINAL design, self-contained (a build agent and a pre-registration can use Part 3 without any earlier turn). I read `pod/README.md`, `pod/m2_watch.sh`, the head of `sample_wrap.py`, and `steer.md` (E0') for this turn; I did not re-read Dan's `sample.py`, so facts about it come from the auditor's Turn 4 and are marked.

### Part 1. Replies to R1-R8

- **R1 (erratum).** Accept; Part 2.
- **R2 (PF engine and tests).** Superseded by the orchestrator's constraint: PF runs in HF transformers inside M3's dpo session, not in vLLM. That removes the vLLM `prompt_logprobs`, prefix-cache and memory questions entirely, and the "sanity test" is replaced by an HF-native one (Part 3, section 7). I agree with the auditor's point that it needs no generation log-probabilities, so the test compares HF against itself (determinism, finiteness, a more-than-4,096-token sequence). Accept the intent.
- **R3 (power-sampling conditions).** Accept in full: (vi) C4 exists and shows the arm-A move at pure T 0.6; (iv) is C2 (pure T 0.8) between stored T 1.0 and C4; no proposal on a C3-only path.
- **R4 (PF gate direction and length).** Accept. "Not in the tail" requires the S interval's upper bound at or above 0 AND P_low at most 25 %, on per-token and on length-adjusted total likelihood; answer segment is a pre-stated secondary; the card-decoded shift is descriptive only. One argument back: the orchestrator asks for 4 card-decoded draws per prompt in the scored set. I keep them (descriptive scale; about 25 % of PF cost) but they enter no gate.
- **R5 (watchdog from observable state).** Accept `--chunk 1` and the proxy, with two changes: the proxy is a small in-job guard (tails the job's own log) and the serrano watcher stays log-only as it is written. What each reads is in Part 3 section 8. Facts I rely on from the auditor (not re-verified by me): FINAL_ONLY output, rows written when a request finishes, vLLM stats line every about 10 s.
- **R6 (trigger B against a baseline).** Accept: frozen detector, thinking segment only (this also removes the IFBench repeat-family problem), threshold at the same checkpoint's stored T 1.0 flag rate plus 1.5 points; the detector is frozen in a committed file whose sha256 goes into every job sidecar.
- **R7 (cut order).** Accept the order; I keep PF's card-decoded group only while PF is at its full size, and drop it first when PF is cut (it is the cheapest cut with no gate loss). Costed in Part 3 section 10.
- **R8 (trigger A escalates).** Accept; the orchestrator decides, and the trigger script only prints its verdict and the cost.

New points from the orchestrator's constraints and one disagreement:
- **E0' as the fresh reach baseline.** Adopted, with the comparison defined in Part 3 section 6. Residual confound: E0' runs through M3's patched engine with a zero vector, C1 through stock vLLM plus the wrapper; M3's identity check is the evidence they agree. If E0' is missing or fails its identity check at decision time, fall back to the proxy (0 of 84 dpo; 2 of 1,536 at s0100, 0.13 %) and say so.
- **PF set size.** Orchestrator: 4 s2300 strict-pass, 4 dpo, 4 card-decoded per prompt. I argue for 4 s2300, **8** dpo, 4 card-decoded, because the no-shift value of P_low is 1/(n+1) for n dpo comparators (11 % at 8, 20 % at 4), so 4 comparators blunt the gate. Cost of the extra 4 dpo sequences per prompt: 192 sequences, about 1.1 M tokens, about 4-8 min of M3's session (about $0.4-0.7). If the orchestrator keeps 4, the 4-comparator thresholds are fixed below (Part 3, section 7) so the choice cannot be made after seeing data.

### Part 2. Erratum for Turn 3 (dated 2026-11-27, written by the reader-implementer)

Turn 3 was written with an unquoted heredoc and some dollar amounts lost their leading digits. Turn 3 is not edited; the intended text is:
- In reply A5 (Ji cost): "...about 143 M tokens, **about $140**; the auditor's 'at least **$45**' assumes a guessed 3x throughput uplift ... my '**$6.5**, pilot Ji first' was wrong ... at B = 1,024 Karan's MCMC (11x, **about $24**) is cheaper than Ji-style (at least 65x)."
- In table row B4: "It is the second cut candidate under a **$10** cap, see section F."
- In the PF cost bullet: "14-27 min, about **$1.3-2.4** on the C1 load (no extra load)." (The auditor corrected "no extra load"; under the new constraint PF runs in M3's session, Part 3.)
No other figure in Turn 3 was knowingly altered; the cost table in Turn 3 section D used escaped dollar signs and is intact. The Turn 3 figure "about $140 per Ji-style cell" is itself a lower bound, not an estimate.

### Part 3. FINAL DESIGN (M1, key `temp`)

**1. Question and rule.** Does sharpening dpo's own distribution recover part of RL's gain, with no training? Dan's stored draws are T 1.0, top_p 1.0 (the RL rollout setting; Olmo 3 report p. 47). The model card and the report's evaluations use T 0.6, top_p 0.95, 32,768 max tokens (cards; report pp. 37, 114). Footnote 71 (p. 114): thinking models "degenerate quickly" at low temperature, so nothing below 0.6 is run. Power sampling is not approved; only its conditions are given (section 9). Strict checker: write "checker passes".

**2. Prompt sets and baselines (all stored, free).** Arm A = 218 IFBench prompts (`pids_armA.txt`); reach = 48 prompts (`pids_reach.txt`; 42 also in arm A; selected as 0 of 16 dpo draws each). Stored T 1.0 baselines (Dan; first-N rows in source order via `dan_curve.Rows.draws`): dpo arm A 16 draws/prompt (29.0 %), s0500 arm A 8 (41.2 %), s2300 arm A 8 (68.3 %), s0500 reach 32 (4.5 %), s2300 reach 32 (69.4 %). Reach baseline for dpo: M3's E0' (section 6) when available, else the proxy. Pipeline reproduction at T 1.0: Stage 0c, dpo -0.8 [-2.9, +1.3], s2300 +2.1 (2026-11-25 log).

**3. Cells.** Every cell is one hand-started pod job (`pod/m2_pod.sh job`), sampling through `sample_wrap.py` (Dan's sampler unedited), grading inside the job via `tools/grade_job.sh`. Draw counts, T and top_p are parameters. No LoRA anywhere: no `--lora` is passed, and the wrapper sidecar's `llm_kwargs` must show no `enable_lora` (checked by the trigger script). `--chunk 1` for every cell (rows are written as each draw finishes; arm A may use `--chunk 2` with `--n 2`).

| Id | Checkpoint | T / top_p | Prompts x draws | Draws (default) | Status |
|---|---|---|---|---|---|
| C1a | dpo | 0.6 / 0.95 (card) | 176 arm-A non-reach x 2 | 352 | always (job `temp_dpo_C1`, invocation a) |
| C1b | dpo | 0.6 / 0.95 | 48 reach x 8 | 384 | always (same job, invocation b) |
| S5R | s0500 | 0.6 / 0.95 | 48 reach x 8 | 384 | always (job `temp_s0500_C1`) |
| PF | scoring only, M3's HF dpo session | n/a | per reach prompt: 4 s2300 strict-pass + 8 dpo stored + 4 C1b draws | 768 sequences | always if M3's session can host it; see section 7 |
| C3 | s2300 | 0.6 / 0.95 | 218 arm A x 2 | 436 | conditional on trigger A (orchestrator decides) |
| C4 | dpo | 0.6 / 1.0 (pure T) | 218 arm A x 2 | 436 | conditional on trigger A |
| C2 | dpo | 0.8 / 1.0 | 218 arm A x 2 | 436 | conditional on trigger B |
Arm-A scoring for the 42 shared prompts uses their first 2 draws (chunks 0 and 1 of C1b). Seeds: one fixed `--seed` per job, recorded in the sidecar; draw identity = (pid, chunk, i). A pure-T 0.6 cell is mandatory before any headline says "temperature" or "sharpening"; until C4 exists every headline says "card decoding".

**4. Analysis definitions (fixed before data).** Unit = prompt. B_p = mean strict pass over all stored T 1.0 draws; N_p = mean over the new cell's first 2 draws. Delta(c) = mean over the 218 arm-A prompts of N_p - B_p. Prompt bootstrap, 10,000 resamples, seed 0, the same resamples for every statistic, paired. Loops, truncations and missing `</think>` count as failures in the headline; pass@1 among finishing draws is secondary.
- share_RL = Delta(dpo, C1) / D, D = mean_p(B_p^s2300 - B_p^dpo) = 39.3 points, numerator and denominator in one joint bootstrap; interval reported.
- Matched gap (only if C3 runs): G(C1) = mean_p(N_p^{s2300,C1} - N_p^{dpo,C1}) against 39.3; reported beside share_RL, never averaged.
- Reach (section 6) and s0500 reach: r(S5R) against its stored 4.5 % (32 draws per prompt) and "any pass in 8" against the first 8 stored draws.
- Secondary split, defined on draws 1-8 of the dpo stored T 1.0 draws (100 prompts with at least one pass, 118 with none) with the baseline from draws 9-16 only (61.5 %, 1.4 %): separates "sharpening where mass exists" from "creating mass". Resolution: MDE about 6.8 points in the ever-passed group, about 0.8 in the never-passed group (auditor).
- Also: pass@2, baseline-rate bins, constraint-family breakdown, token mean / p50 / p90 / p99, share finish=length, share without `</think>`, repetition flag (section 5).
- MDEs reported with every null (80 % power, paired; auditor's computations): about 3.3 at dpo arm A, 4.4 at s0500, 4.7 at s2300; s0500 reach paired SE about 1.1 at 8 draws (MDE about 3.2), about 1.5 at 4 (MDE about 4.3).
- Labels: arm-A effects below +4 points are "detected, small" (under 10 % of the gap). No setting is chosen from data. If one must be (T, alpha), choose on a fixed random half of the 176 non-reach arm-A prompts (88, seed fixed in advance) and report on the other 88 plus reach.

**5. Triggers (computed by a script after a cell is graded; the orchestrator decides).** Script `tools/temp_triggers.py` (to be built) reads graded files and the sidecars, prints each trigger's inputs, verdict and the cost of the cells it would release, and changes nothing.
- **Trigger A** (releases C3 and C4, about $6.3): (a) the paired arm-A pass@1 difference Delta(dpo, C1) has a two-sided 95 % bootstrap interval excluding 0, either sign, no floor; or (b) the reach rule in section 6.
- **Trigger B** (releases C2, about $2.9): trigger A did not fire AND (the share of C1 draws flagged repetitive in the thinking segment is at least 1.5 points above the same detector's rate on the stored dpo T 1.0 draws, or the share cut at the cap is at least 1.5 points above the stored 0.06 %); or a power-sampling proposal is to be written (monotonicity).
- **Repetition detector** (frozen in `tools/temp_repeat_flag.py` before any data; sha256 in every sidecar): thinking segment only (text before `</think>`; whole text if absent); flag if, among sentence or paragraph units of at least 20 characters, any unit repeats 10 or more times, or any word 10-gram repeats 50 or more times. Its stored T 1.0 rates are measured first on dpo, s0500 and s2300 draws (the auditor's crude version gave 0.7 % at dpo, 2.8 % at s2300 on the Stage 0c draws).
- Stop-and-report: if neither trigger fires, report the null at card decoding with the MDEs, and spend nothing further (apart from PF if it is already funded).
- If trigger A fires, the agent stops and asks the orchestrator for C3 + C4 with the C1 result in hand. Only if refused: C3 alone ($3.4), headline "card decoding", no power-sampling proposal.

**6. Reach comparison with E0'.** E0' = M3's dpo, patched engine, zero vector (after its exact identity check), T 1.0, top_p 1.0, 48 reach prompts x 8 draws (384), seed 1000, 32,768 cap. M1's dpo reach comparison: per prompt, rate under C1b (card decoding, 8 draws) against rate under E0' (8 draws); d_p = rate(C1b) - rate(E0'); statistic = mean_p d_p with a prompt bootstrap over the 48 prompts, and also the raw counts (passes, prompts with a pass) for both cells. Reach rule (part of trigger A): C1b has at least 3 passes from at least 2 prompts AND (passes in C1b) - (passes in E0') is at least 3. share_reach = (r(C1b) - r(E0')) / (69.4 % - r(E0')). If C1b is run with 4 draws, rates (not counts) are compared and the count rule becomes at least 2 passes from at least 2 prompts and a difference of at least 2. If E0' is not available at decision time the orchestrator may wait or use the proxy; either way the report names which. s0500 reach has its own stored 32-draw baseline (stock engine), so S5R needs no fresh baseline.

**7. Prefill likelihood check (PF), in HF, in M3's dpo session after M1's sampling.**
- *Model and scoring:* `allenai/Olmo-3-32B-Think-DPO` in bf16, batch 1, chunked log-softmax over raw logits at temperature 1 (no top_p), M3's NLL code path. Score response tokens conditioned on the prompt (prompt ids rebuilt from the `olmo_thinker` template, sha256 checked against Dan's meta; response ids from stored `ids`, which the auditor verified are response-only, ending in EOS 100257 when `finish == stop`). Include the EOS when present; per-token means exclude draws cut at the cap.
- *Sequences per reach prompt:* 4 stored s2300 draws that pass the strict checker (first 4 passing draws in draw order of the 32 stored; if fewer than 4, all of them; prompts with none are dropped), 8 stored dpo T 1.0 draws (first 8 of 16) (argued up from 4), 4 card-decoded C1b draws (first 4 in draw order). 48 x 16 = 768 sequences. Fallback 4 / 4 / 4 (576 sequences) is allowed; its thresholds are fixed here: the no-shift P_low is then 20 %, so "not in the tail" needs P_low at most 35 % instead of 25 %.
- *Statistics (for each response, over the full response and over the answer segment after `</think>`):* per-token mean log-probability l; total log-probability L with length n.
  - Per-token: S_pt = median over prompts of [mean l of that prompt's s2300 passing responses minus mean l of its dpo responses]; P_low,pt = share of s2300 passing responses whose l is below the minimum l of that prompt's dpo responses.
  - Length-adjusted total: pooled within-prompt slope beta from regressing L on n over all dpo responses with prompt fixed effects; residual rho = L - [mean L_dpo,p + beta (n - mean n_dpo,p)]; S_adj = median over prompts of the mean rho of s2300 passing responses; P_low,adj = share of s2300 passing responses whose rho is below the minimum rho of that prompt's dpo responses.
  - Intervals: prompt bootstrap, 10,000 resamples, seed 0.
- *Gate (verdict recorded before looking at arm-A results of other cells):* TAIL if for the full response either S_pt or S_adj has an upper CI bound below 0, or either P_low is at least 50 %. PASS ("not in the tail") only if both S_pt and S_adj have CI upper bounds at or above 0 and both P_low are at most 25 % (35 % in the 4-comparator fallback). Else UNRESOLVED. Answer-segment S and P_low are reported as pre-stated secondaries and do not change the verdict. The card-decoded group (l and L of C1b draws) is a descriptive scale only. The gate feeds only the power-sampling proposal (section 9), not the trigger A/B decisions.
- *HF sanity checks (about 5 min):* score 3 stored dpo responses twice (identical to bf16 batch-1 determinism, every position finite); score one response longer than 4,096 tokens (sliding window); check the template sha. Thresholds on P_low (25 / 50 %) and the length adjustment remain guesses (OPEN 1).
- *Inputs and outputs:* `tools/temp_pf_select.py` writes a manifest (prompt id, source job, chunk, i, group, pass flag); `tools/temp_pf_stats.py` reads M3's per-sequence log-probability dump and prints S, P_low, verdict. Both CPU-testable. Cost in section 10.

**8. Watchdog: exactly what each piece reads.**
- *Serrano side, `pod/m2_watch.sh <alias> <job> <expected-minutes>` (unchanged, log-only, kills nothing).* It polls `m2_pod.sh status` and reads: `state` (RUNNING / DONE / FAILED / STALE / ABSENT), `hb` (age of `state/heartbeat`, written every 60 s by the pod runner), `started`, `progress` (complete lines of the newest output file) and the signature newest file name plus size. It logs STALL when hb is older than 10 min or when the newest output file's name:size has not changed for `GROW_STALL`; OVERRUN when the job runs over 1.6 x expected minutes; EXIT on end of job. Because output is FINAL_ONLY, nothing is written to the output file until a request finishes, and with 5k-token draws at about 1,500 tok/s shared across hundreds of sequences the first finish may take on the order of 14 min (my estimate from the throughput figures), so run it with `GROW_STALL=2400` for the first 30 min rather than the default 900 (a guess; the auditor's 20-min STALL rule maps to this variable). The watcher cannot see tokens in flight.
- *In-job guard, `tools/temp_inflight_guard.py` (to be built; started in the background inside the job command, with the plan tokens as arguments).* Reads: (1) the job's own `logs/job.log` for vLLM's periodic stats line (average generation throughput, running and waiting request counts; format unverified, so the first minute of the first job must show it, otherwise the guard writes GUARD_BLIND to its own log and the orchestrator is told); (2) the count of rows written so far in `samples.jsonl.gz` (zcat tolerating a truncated tail). It integrates throughput into G(t), generated tokens finished plus in flight (about 10-s granularity), and writes `GUARD` lines every 5 min (G, running, waiting, rows written). It aborts the sampler (SIGTERM to its own child process group) when (a) G(t) exceeds 1.5 x plan before 90 % of the cell's draws are written, or (b) waiting = 0, running at least 10 % of the cell's draws, and mean in-flight length (G minus finished tokens) / running is at least 16,384, or (c) no row has been written for 20 min while running > 0. On abort the cell is reported "aborted", finished draws are graded, unfinished draws count as failures and are not dropped. With `--chunk 1` an abort loses only in-flight draws. The watcher then logs EXIT with FAILED state and rc. Plans: dpo 5,100 tokens per draw, s0500 5,640, s2300 6,900 (1.13x stored means, a guess).
- *What neither reads:* per-request lengths (not logged). Thresholds 1.5x and 16,384 are guesses (OPEN 3).

**9. Power-sampling follow-up: conditions only (not approved).** A proposal may be written only if all hold: (i) Leon's budget rule: trigger A fires with a positive arm-A move at dpo C1 (a budget rule, not implied by the papers: Ji Table 1, DeepSeek-Math-7B, low T closes 3 % of the MATH500 gap and power sampling 78 %); (ii) share_RL between 10 % and 70 % and dpo C1 still at least 10 points below s2300; (iii) cut-at-cap below 5 % at C1 and the lift persists among finishing draws; (iv) C2 (pure T 0.8) lies between stored T 1.0 and C4 (pure T 0.6); (v) PF verdict PASS; (vi) C4 exists and shows the arm-A move at pure T 0.6. No proposal on a C3-only path. If proposed: Karan's MCMC, alpha = 1.67, proposal at T 0.6, block size B of 512-1,024; cost N_MCMC T / (4B) times plain decoding (11x at B = 1,024, 22x at 512, about 59x at B = 192, T = 4,500): about $24 per arm-A x 2 cell at B = 1,024 (2.2 M plain tokens x 11 at $0.98 per M); Ji-style at least 65x tokens (at least $45, likely about $140, per cell). Any pilot needs vLLM's sequence log-probabilities (unverified) and a custom MH loop.

**10. Cost under the constraints** ($5.29/h, 1,500 tok/s so about $0.98 per M generated tokens; load or reload from the pod's disk about 8 min = $0.7 first, about 4 min = $0.35 for a second invocation in the same job; 1.13x length guess; PF at a guessed 3-6k tokens/s in HF).

| Item | Detail | Tokens (M) | $ |
|---|---|---|---|
| C1a + C1b, one job, two invocations | 736 x 5,100; load 8 + reload 4 min | 3.75 | about 4.7 (4.4-5.1) |
| S5R | 384 x 5,640; load 8 min | 2.17 | about 2.8 |
| PF, in M3's session | 768 sequences x about 6.1k = 4.7 M forward tokens, 13-26 min | n/a | about 1.2-2.3 |
| **Always-run total** | | | **about 8.7-9.8** |
| C3 (conditional A) | 436 x 6,900; load | 3.0 | about 3.4 |
| C4 (conditional A) | 436 x 5,100; reload | 2.2 | about 2.9 |
| C2 (conditional B) | 436 x 5,100; reload | 2.2 | about 2.9 |
| Trigger A total | C3 + C4 | | about 6.3 |
Loop sensitivity: 5 % loops (mean 5,900) adds about $0.6 to C1 and about $0.3 to S5R; at the 1.5x abort, C1 is about $6.6 and S5R about $3.9. If E0' is not run by M3, nothing in M1 changes except the reach baseline falls back to the proxy.

**Cut order under a $10 cap (always-run set; trigger-A cells are conditional and not counted).**
1. C1b reach draws 8 to 4 per prompt (-192 draws, about -$1.0). Costs: dpo reach bound goes from about 0.8 % to 1.6 %; reach rule becomes at least 2 passes from at least 2 prompts with a difference of at least 2. Arm A untouched.
2. PF: drop the card-decoded group (768 to 576 sequences; about -$0.3-0.6). Costs: only the descriptive scale; gate untouched. If PF is cut further, go to the 4 / 4 / 4 fallback only with its fixed thresholds.
3. S5R 8 to 4 per prompt (about -$1.0). Costs: the test detects only about a doubling or halving of the 4.5 % rate.
4. Drop S5R (about -$2.8). Costs: the experiment can no longer ask whether sharpening multiplies a partly learned behaviour; only dpo remains, where reach is near zero by construction.
5. PF last (it only gates a proposal that needs a positive trigger A first).
After cuts 1-2 the always-run set is about $7.4-8.3, leaving about $1.7-2.6 for loops. At the 1.5x abort both sampling cells together are about $9.1 before PF.
Trigger-A cells (about $6.3) do not fit under $10 after any cut; the orchestrator is asked when trigger A fires.

**11. Build list for the build agent (none written).** `tools/temp_repeat_flag.py` (frozen first), `tools/temp_triggers.py`, `tools/temp_inflight_guard.py`, `tools/temp_pf_select.py`, `tools/temp_pf_stats.py`, input files `inputs/pids_armA_nonreach.txt` (176) and `inputs/pids_reach.txt` (48), the job command lines in the README pattern (cwd `/workspace/m2`), a test for the guard against a recorded vLLM stats line, and a CPU test that reproduces the section 4 baselines (29.0 / 41.2 / 68.3 and the 100 / 118 split with 61.5 % / 1.4 %).

### OPEN (still disputed or undecided)

1. PF thresholds (25 / 35 / 50 %) and the length adjustment (pooled within-prompt slope versus per-prompt); and the dpo comparator count (8 argued, 4 allowed).
2. Card decoding versus pure T as the Tier 1 arm (accepted only with R3).
3. Guard thresholds (1.5x plan, 16,384, 20-min stall, `GROW_STALL=2400`), the +1.5-point trigger-B margin, and whether the vLLM stats-line format is as assumed (unverified).
4. Whether E0' (patched engine, zero vector) is interchangeable with a stock-engine fresh cell; M3's identity check is the only evidence.
5. Dev / held-out split for any later choice (88 / 88 non-reach arm A).
6. alpha, proposal temperature and block size of a power-sampling follow-up, and whether any pilot is warranted.
7. Card IFBench strict versus loose (prior only); provenance of Karan's GRPO checkpoint (no effect on the design).

## Turn 6 — auditor (Claude Opus 5.5), 2026-11-27 02:15 PT

Closing turn (round 3). I reopen no conceded point. For this turn I re-read Part 3 against Turn 4's R1-R8 and checked
Dan's `parse_spec` (sample.py lines 136-143).

**R1-R8 as implemented**
- **R1: done.** "In table row B4: 'It is the second cut candidate under a **$10** cap'" (Part 2). The other five amounts
  are restated there too.
- **R2: superseded, accepted.** PF moves to HF in M3's session ("chunked log-softmax over raw logits at temperature 1").
  That removes the vLLM prompt_logprobs, prefix-cache and memory questions.
- **R3: done.** "(iv) C2 (pure T 0.8) lies between stored T 1.0 and C4 (pure T 0.6); … (vi) C4 exists …. No proposal on
  a C3-only path."
- **R4: done.** "PASS ('not in the tail') only if both S_pt and S_adj have CI upper bounds at or above 0 and both P_low
  are at most 25 %".
- **R5: done, with one gap (fix 3).** "`--chunk 1` for every cell"; guard rules (a), (b), (c) as in Part 3 §8.
- **R6: done.** "thinking segment only …; threshold at the same checkpoint's stored T 1.0 flag rate plus 1.5 points;
  detector frozen … sha256 in every sidecar". The draw set it applies to is defined in fix 1.
- **R7: order changed, rationale wrong.** Part 1 says "Accept the order", but Part 3 puts "5. PF last (it only gates a
  proposal that needs a positive trigger A first)". That reason argues for cutting PF *earlier*, not last. I accept PF
  last for the different reason the new constraint creates: PF can only run inside M3's dpo HF session, so cutting it is
  permanent, while S5R cuts only lose resolution. State that reason (fix 6).
- **R8: done.** "If trigger A fires, the agent stops and asks the orchestrator for C3 + C4 with the C1 result in hand."

**Defects in Part 3**, each with its fix:
1. **Trigger B's draw set is undefined.** It compares "the share of C1 draws" (arm-A C1a plus 384 reach-heavy C1b draws)
   against stored *arm-A* dpo rates ("stored 0.06 %"), which are different prompt mixes. Fix: compute both shares on C1's
   436 arm-A draws (C1a plus the first 2 C1b draws of the 42 shared prompts) against the stored dpo arm-A draws, using the
   same frozen detector.
2. **Reach rule at 4 draws, and the proxy branch.**
   - "a difference of at least 2" compares 192 C1b draws with 384 E0' draws. Fix: when C1b has 4 draws, use E0''s first
     4 draws per prompt.
   - Under the proxy, the count clause has no E0' term. Fix: there the rule is C1b ≥ 3 passes from ≥ 2 prompts (≥ 2 at
     4 draws).
   - "the orchestrator may wait or use the proxy" is discretion exercised after data. Fix: E0' is used if and only if it
     is graded and identity-checked when C1b is graded; otherwise use the proxy. This is decided by availability, not by
     results.
3. **The guard cannot compute rule (b) as written, and goes blind if the stats line is unparseable.**
   - (b) needs "finished tokens", but the guard is listed as reading only the *count* of rows. Fix: read Σ `n_tok` of the
     written rows; rows carry `n_tok`.
   - The guard relies on an unverified stats-line format and only *reports* GUARD_BLIND. Fix: when blind, fall back to a
     mechanical wall-clock cap. Abort if elapsed generation time exceeds 1.5 × (plan tokens / 1,500 tok/s) before 90 % of
     rows are written, keeping rule (c). The guard then never depends on the format.
4. **Cost slip: the C1 reload is avoidable.** Dan's `parse_spec` takes `@file` with `pid:n` per prompt (lines 136-143).
   So C1a (176 pids at 2 draws) and C1b (48 pids at 8 draws) can be one invocation with `--chunk 1`. Saves the second
   load, about $0.35: C1 becomes about $4.4, and the always-run set about $8.4-9.5. Add this to the CPU test.
5. **The PF ">4,096-token (sliding window)" check has no pass criterion.** Fix: assert that the loaded config has
   `sliding_window` 4,096 on the 48 sliding layers and that the attention implementation in use applies it; then
   finite and deterministic. Otherwise label the check descriptive.
6. **State the true reason PF is cut last** (see R7).
7. **§9 (iii) "the lift persists among finishing draws" is undefined.** Fix: Delta computed over finishing draws only has
   the same sign and a 95 % CI that excludes 0.
8. **The PF gate's "verdict recorded before looking at arm-A results of other cells" cannot be enforced**, because C1 is
   graded inside its own job before PF runs. Fix: replace it with "thresholds frozen in Part 3 §7; verdict printed by
   `temp_pf_stats.py`, no manual step".

**E0' (M3's patched engine, zero vector) as the dpo reach baseline.** I accept it. It is better than the proxy: fresh
dpo T 1.0 draws on exactly the 48 reach prompts, at C1b's draw count, from a set not selected on those draws.

The confound can only bite if the zero-vector identity check is not *exact* or was run under different engine settings.
So `temp_triggers.py` should read E0''s sidecar and require all of the following, else fall back to the proxy:
- identity check passed, run on the same engine instance and config (dtype, CUDA-graph/eager setting);
- the same template sha;
- T 1.0, top_p 1.0;
- 32,768 max tokens;
- the same strict grader.

At a true rate of about 0.1 %, E0' expects about 0.5 passes in 384 draws, so kernel-level bf16 differences are
irrelevant at this resolution.

**Remaining objection:** none. With fixes 1-8 applied (all code or text, $0, saving about $0.35), I have no objection to
running Part 3 as the M1 design.

**Final OPEN**
1. PF thresholds (25 / 35 / 50 %), the length-adjustment form, and dpo comparator count (I back 8; P_low no-shift 11 % vs 20 %).
2. Card decoding versus pure T as Tier 1 (accepted with R3).
3. Guard thresholds (1.5× plan, 16,384, 20-min stall, `GROW_STALL=2400`), the +1.5-point trigger-B margin, the stats-line format.
4. E0' interchangeability beyond its sidecar check (accepted as above).
5. Dev / held-out split for any later choice (88 / 88).
6. α, proposal and block size of any power-sampling follow-up, and whether a pilot is warranted.
7. Card IFBench strict or loose; Karan's GRPO provenance (prior only).
