---
written_on: 2026-10-09
written_by: agent:claude
---

# Does training on synthetic thinking traces buy test-time-scaling uplift at matched compute?

Branch `leon/thinking` at `be062f3` (uncommitted: this note, `matched_eval/tts_figs.py`, its tables and the two
figures). Figures: [`fig_tts_uplift`](../../../../experiment-summaries/2026-10-05-thinking-traces/figures/fig_tts_uplift.png)
(13–20) and [`fig_tts_uplift_other`](../../../../experiment-summaries/2026-10-05-thinking-traces/figures/fig_tts_uplift_other.png)
(textbook72, hard712 dev). Every number below is computed from the eval rows under
`/scratch/cowifornia/nd-rl/think/runs/eval/`. The tables in §§ 3.1, 3.2 and 3.4 come from
[`matched_eval/tts_figs.py`](matched_eval/tts_figs.py) (full output:
[`matched_eval/tts_uplift_tables.md`](matched_eval/tts_uplift_tables.md)). § 3.3 uses `stats.ci` on the k = 8 reads in
`eval/matched/`. The prefill counts are means over the same rows, and the wall-clock times are file timestamps.

## 1. The question

Leon's question (scope reset, 2026-10-09): does the small natural-deduction prover trained with **synthetic thinking
traces** get test-time-scaling uplift over a **no-think baseline** when both are given the **same inference compute**?
This is test-time scaling in the LLM tradition: the thinking model spends tokens on think blocks, and the question is
whether those tokens buy more than the same tokens spent on more attempts without thinking. Expert iteration (EI) is
not under test; it is how both models are trained after Stage 1, identically.

## 2. The design

**The pair.** The two models differ only in the thinking traces:

| | Stage 1 (SFT) | then |
|---|---|---|
| thinking | S1g3: honest-search thinking traces with effort tokens (budgets 0/64/256 at 20/40/40%) | the same EI rounds |
| no-think baseline | T0S: the same proofs, no think blocks | the same EI rounds |

The EI rounds are r1 on rl_hard12, r2 on a 5k sample (a) of hard712, and r3 on a disjoint 5k sample (b) of hard712.
Seed 0 has r1, r2 and r3. Seed 1 has r2. Its r3 pair is not in this note: as of 9:25 am PDT the baseline had finished
EI and its 13–20 read was running, and the thinking model was still in EI.

**The sets.**
- **13–20-line proofs** (986 valid items of `t1320_n1000`, filtered by `figs.valid_1320`): out of distribution for
  length. This is the main test.
- **textbook72** (72 problems: textbook_dev 58 + textbook_train 14).
- **hard712 dev** (500 problems): in distribution for r2 and r3.

**The reads.** The thinking model is read at a think budget of 64 tokens per step (B64), its best per-attempt budget
(§ 3.3). It is read at k = 32 attempts per problem on 13–20 and hard712 dev, and at k = 64 on textbook72. The baseline
is read at B0 with k = 64. Temperature is 0.8 and the step limit is 96. Where it exists, the thinking model is also read
at B0 (thinking off) with k = 64.

**The metric.** The metric is pass@k. Lean verifies every proof, so pass@k means "solved within k attempts", and the
unbiased estimator is used. It is compared at matched **sampled tokens per problem**: think + `</think>` + action +
rejected redraws, with prompt prefill excluded.
- `stats.at_tokens` reads each arm at k* = X / (mean sampled tokens per attempt), interpolated linearly in log k.
- `stats.paired_diff_at_tokens` gives thinking − baseline on a shared problem resample.
- Intervals are 95% percentile intervals of 4,000 bootstrap draws over problems. Each arm is one training seed, so seed
  variance is not inside any interval.
- A target outside an arm's sampled range (below 1 attempt or above k attempts) is not drawn, and is flagged ‡ in the
  tables.

**Why matched compute.** A thinking attempt costs 2.45–3.05× the sampled tokens of a baseline attempt (table below), so
comparing pass@1, or pass@k at equal k, gives the thinking model about 3× the compute. The fair question is whether,
for a fixed token budget per problem, X tokens of "think then act" beat X tokens of "more attempts".

## 3. Results

### 3.1 Thinking (B64) − baseline: per attempt and at matched sampled tokens per problem

Percentage points, with 95% intervals. ‡ marks a target outside a read's sampled range; see the footnote.

| set | pair | tokens / attempt (think / base) | per attempt (pass@1) | 1k | 2k | 4k | 8k | 16k |
|---|---|---|---|---|---|---|---|---|
| 13–20 | r1 s0 | 810 / 305 (2.65×) | +1.3 [−0.5, +3.2] | −11.1 [−13.3, −8.8] | −9.6 [−12.0, −7.0] | −7.2 [−9.9, −4.4] | −4.7 [−7.6, −1.7] | −2.3 [−5.4, +1.0] |
| 13–20 | r2 s0 | 898 / 302 (2.98×) | +5.9 [+4.0, +7.9] | −8.5 [−10.8, −6.1] | −6.1 [−8.6, −3.4] | −3.9 [−6.6, −1.0] | −2.5 [−5.3, +0.5] | −1.6 [−4.6, +1.5] |
| 13–20 | r2 s1 | 908 / 313 (2.90×) | +6.8 [+4.6, +8.9] | −6.5 [−9.1, −4.0] | −3.4 [−6.3, −0.4] | +0.7 [−2.5, +3.9] | **+4.6 [+1.4, +8.0]** | **+7.6 [+4.1, +11.2]** |
| 13–20 | r3 s0 | 934 / 306 (3.05×) | +2.9 [+0.8, +5.0] | −13.9 [−16.3, −11.4] | −12.6 [−15.2, −9.8] | −10.7 [−13.5, −7.8] | −8.8 [−11.8, −5.8] | −7.1 [−10.2, −3.9] |
| textbook72 | r2 s0 | 488 / 179 (2.73×) | +4.2 [−3.2, +11.2] | −0.9 [−7.6, +5.3] | +1.3 [−4.0, +6.6] | +3.0 [−0.9, +7.4] | **+3.5 [+0.4, +7.5]** | +4.2 [+0.7, +8.8] ‡ |
| textbook72 | r2 s1 | 451 / 156 (2.90×) | +11.7 [+4.8, +18.7] | **+5.4 [+0.1, +10.6]** | **+6.0 [+1.5, +10.8]** | **+6.1 [+1.7, +11.2]** | **+6.2 [+1.6, +11.9]** | +7.5 [+2.2, +13.8] ‡ |
| hard712 dev | r2 s0 | 465 / 189 (2.45×) | +2.4 [−0.5, +5.3] | −3.0 [−5.5, −0.4] | −0.6 [−2.8, +1.7] | +1.0 [−1.0, +3.1] | +2.0 [+0.0, +4.1] | +2.6 [+0.6, +4.8] ‡ |
| hard712 dev | r3 s0 | 467 / 182 (2.57×) | +0.0 [−2.4, +2.4] | −2.5 [−4.7, −0.4] | −0.5 [−2.5, +1.4] | +0.3 [−1.5, +2.1] | +0.7 [−1.0, +2.5] | +1.0 [−0.6, +2.8] ‡ |

Bold: matched-token differences whose interval excludes 0 in thinking's favour.

‡ At 16k the baseline's 64 attempts cost less than 16k tokens on textbook72 (at most 11.4k and 10.0k tokens) and on
hard712 dev (12.1k and 11.6k). Its pass@64 stands in, an underestimate, so on textbook72 the difference is an upper
bound. On hard712 dev the thinking read's 32 attempts also end below 16k (14.9k and 15.0k), so both arms are clamped and
the difference has no clear direction. Every 13–20 cell is in range.

Absolute values for context, all from the same reads:
- **13–20 pass@1, thinking / baseline:** r1 22.6 / 21.3%; r2 s0 30.2 / 24.3%; r2 s1 31.0 / 24.2%; r3 30.9 / 28.0%.
- **Pass@32 thinking against pass@64 baseline:** r1 67.0 / 66.4%; r2 s0 73.0 / 71.1%; r2 s1 76.8 / 65.4%; r3
  71.8 / 75.3%.

### 3.2 The thinking model with thinking off (B0) and its best budget per compute level

The thinking model was also read with thinking off (B0, k = 64) for both round-2 seeds on 13–20, and for seed 0 on
textbook72. Its best budget per compute level is the upper envelope of its B0 and B64 curves. Below the cost of one B64
attempt, only B0 is possible.

| set | pair | thinking at B0: tokens / attempt, pass@1 | B0 − baseline, per attempt | best budget − baseline: 1k | 2k | 4k | 8k | 16k | best budget |
|---|---|---|---|---|---|---|---|---|---|
| 13–20 | r2 s0 | 392 (1.30× baseline), 24.0% | −0.3 [−2.0, +1.4] | −3.1 [−5.4, −0.8] | −2.2 [−4.8, +0.5] | −1.7 [−4.4, +1.2] | −1.5 [−4.0, +1.6] | −1.1 [−3.6, +2.2] | B0 from one attempt (392 tokens) to 20k |
| 13–20 | r2 s1 | 359 (1.15×), 15.9% | −8.3 [−10.1, −6.5] | −6.5 [−9.1, −4.0] | −3.4 [−6.3, −0.4] | +0.7 [−2.5, +3.9] | **+4.6 [+1.4, +8.0]** | **+7.6 [+4.1, +11.2]** | B0 below one B64 attempt (908 tokens), B64 above |
| textbook72 | r2 s0 | 178 (1.00×), 47.9% | −14.9 [−21.5, −8.5] | −0.9 [−7.6, +5.3] | +1.3 [−4.0, +6.6] | +3.0 [−0.9, +7.4] | **+3.5 [+0.4, +7.5]** | +4.2 [+0.7, +8.8] ‡ | B0 below 488 tokens, B64 above |

- **Seed 0.** With thinking off, the thinking model matches the baseline per attempt (24.0 against 24.3% pass@1) but
  spends 1.30× its tokens. At matched tokens B0 beats B64 at every budget read, so for this seed thinking never pays.
  Its best is −3.1 at 1k and a tie from 2k.
- **Seed 1.** Thinking off is far worse (15.9% pass@1), so the best budget is B64 as soon as one B64 attempt fits. The
  envelope is the B64 curve, ahead from 8k.
- **textbook72 seed 0.** Thinking off is 14.9 points behind per attempt, so the envelope is the B64 curve.

The adaptive-budget hope was "thinking off at low compute to tie the baseline, thinking on at high compute to beat it".
Neither seed shows it: seed 0 can turn thinking off and roughly tie, but thinking on never wins; seed 1 wins with
thinking on but loses badly with it off.

### 3.3 Why B64 (per-attempt context, k = 8 reads of the same checkpoints)

| 13–20, seed 0 | thinking B0 | thinking B64 | thinking B256 | baseline |
|---|---|---|---|---|
| r1: pass@1 (tokens / attempt) | 16.1% (361) | **23.0%** (800) | 22.0% (1,327) | 20.8% (307) |
| r2: pass@1 (tokens / attempt) | 24.0% (397) | **30.7%** (890) | 27.6% (1,530) | 24.3% (300) |

B64 is the thinking model's best budget per attempt. B256 costs 1.7× more tokens for a lower pass@1, so it is unlikely
to help at matched tokens; it was not read at high k (B256 curves were dropped from the queue). Per attempt is not the
same as at matched tokens, though: B0 is cheaper, and for seed 0 it beats B64 at matched tokens (§ 3.2).

### 3.4 Sensitivity: counting prompt prefill as compute

Each environment step re-prefills the state prompt, so prefill is 70–90% of the forward tokens of an attempt
(13–20 r2 s0: 3,993 prefill against 898 sampled tokens for thinking, and 2,660 against 302 for the baseline). Counting
**forward tokens** (sampled + prefill) shrinks thinking's cost ratio from 2.45–3.05× to 1.36–1.67×. Thinking − baseline
at matched forward tokens per problem:

| set | pair | ratio | 6k | 12k | 24k | 48k | 96k |
|---|---|---|---|---|---|---|---|
| 13–20 | r1 s0 | 1.42× | −2.8 [−4.8, −0.6] | −1.6 [−4.1, +0.9] | −0.0 [−2.7, +2.7] | +1.5 [−1.3, +4.5] | +2.7 [−0.4, +5.8] |
| 13–20 | r2 s0 | 1.65× | −0.4 [−2.6, +1.8] | +1.3 [−1.1, +3.9] | +2.8 [+0.2, +5.6] | +3.4 [+0.7, +6.3] | +3.4 [+0.6, +6.4] |
| 13–20 | r2 s1 | 1.45× | +2.8 [+0.3, +5.2] | +5.2 [+2.4, +8.0] | +8.1 [+5.1, +11.3] | +10.9 [+7.6, +14.2] | +12.8 [+9.4, +16.3] |
| 13–20 | r3 s0 | 1.67× | −4.9 [−7.2, −2.6] | −4.6 [−7.1, −2.0] | −4.0 [−6.7, −1.3] | −3.5 [−6.3, −0.6] | −2.7 [−5.7, +0.3] |
| textbook72 | r2 s0 | 1.50× | +2.1 [−3.6, +8.0] | +3.7 [−0.9, +8.8] | +4.7 [+1.0, +9.2] | +4.7 [+1.1, +9.1] | +5.3 [+1.2, +11.0] ‡ |
| textbook72 | r2 s1 | 1.64× | +8.8 [+3.5, +14.3] | +8.5 [+3.5, +14.1] | +8.1 [+3.1, +13.8] | +7.7 [+2.4, +13.9] | +8.3 [+2.8, +15.2] ‡ |
| hard712 dev | r2 s0 | 1.36× | +1.0 [−1.4, +3.4] | +1.9 [−0.3, +4.0] | +2.6 [+0.5, +4.7] | +3.0 [+0.9, +5.1] | +2.6 [+0.6, +4.8] ‡ |
| hard712 dev | r3 s0 | 1.45× | +0.2 [−1.8, +2.2] | +0.9 [−0.9, +2.8] | +1.2 [−0.5, +3.0] | +1.2 [−0.4, +3.0] | +1.0 [−0.6, +2.8] ‡ |

## 4. Verdict

**Observed.**
- Per attempt, thinking is ahead or level in all eight pairs (+0.0 to +11.7 points pass@1). The interval excludes 0
  in four of them: 13–20 r2 s0, r2 s1 and r3, and textbook72 seed 1.
- At matched sampled tokens on 13–20, thinking is behind at 1k and 2k tokens per problem in all four pairs (−3.4 to
  −13.9 points).
  - At 8k–16k it is still behind for r3 (−8.8 at 8k, −7.1 at 16k) and for r1 at 8k (−4.7; tied at 16k). It is tied for
    r2 seed 0 (−2.5 and −1.6, intervals include 0).
  - It is ahead only for r2 seed 1 (+4.6 at 8k, +7.6 at 16k).
- The difference rises with compute in every 13–20 pair, from 1k to 16k: r1 −11.1 → −2.3, r2 s0 −8.5 → −1.6, r2 s1
  −6.5 → +7.6, r3 −13.9 → −7.1.
- Round 3 widened the gap against thinking: the baseline gained (pass@1 24.3 → 28.0%), the thinking model did not
  (30.2 → 30.9%).
- On textbook72 thinking is ahead at matched tokens in both seeds: from 8k in seed 0 (+3.5 [+0.4, +7.5]) and at every
  budget from 1k in seed 1 (+5.4 to +6.2 up to 8k).
- On hard712 dev it is behind at 1k (−3.0 and −2.5), then tied from 2k. The largest in-range gap is r2's
  +2.0 [+0.0, +4.1] at 8k.
- Letting the thinking model choose its budget per compute level (round 2, both seeds) changes nothing.
  - Seed 0's best budget is thinking off at every budget read. That ties the baseline at best: −3.1 [−5.4, −0.8] at 1k,
    −1.1 [−3.6, +2.2] at 16k.
  - Seed 1's best budget is B64 from one B64 attempt (908 tokens), so its envelope is the B64 curve.

**Interpretation.** The thinking traces raise per-attempt accuracy, but on the main out-of-distribution test (13–20) not
by enough to pay for about 3× the sampled tokens. At equal sampled tokens, more no-think attempts are as good or better
in three of four pairs, and turning thinking off does not close the gap. So there is **no robust test-time-scaling
uplift at matched compute**.

The one positive 13–20 pair (r2 seed 1) is as much a weak baseline seed as a strong thinking seed: its baseline reaches
65.4% at k = 64, against 71.1% for seed 0. The seed-to-seed spread at 16k (−1.6 vs +7.6) is as large as the effects
being measured. The rising difference is consistent with thinking's pass@k growing faster per added attempt (more
diverse attempts), so thinking could win beyond the budgets read here. That is an open hypothesis, not an
observation. On textbook72 the uplift is real in both seeds
but small-sample and design-contaminated (§ 5).

## 5. Caveats

- **Seeds.** One training seed per arm per cell, and two seeds only for r2. The intervals resample problems only. The
  two r2 seeds disagree in sign at 8k–16k on 13–20, so seed variance is at least as large as the differences.
- **The B64 choice.** B64 is the thinking model's best per-attempt budget on 13–20 (§ 3.3), chosen on reads of the same
  sets, which slightly favours thinking. B256 is worse per attempt and costs more, so it is unlikely to help at
  matched tokens. Letting the model pick B0 or B64 at each compute level (§ 3.2) does not rescue it either. That
  envelope picks the better budget after seeing the results, a mild optimistic bias, and is still not ahead of the
  baseline for seed 0.
- **k limits.** The thinking reads stop at 32 attempts on 13–20 and hard712 dev (26–30k tokens on 13–20, 15k on hard712
  dev). The baseline reads stop at 64 (19.3–20.0k on 13–20, 10–12k on textbook72 and hard712 dev). Nothing here says
  what happens beyond about 16–20k tokens per problem on 13–20, which is where a crossover would have to be for r1, r2
  s0 and r3.
- **The free Lean verifier.** pass@k is the right metric only because Lean checks every proof at no token cost, so
  "best of k" can be selected exactly. Without a verifier, or if verification were charged, the comparison would need
  majority vote or a learned verifier, and more attempts would be worth less.
- **What counts as compute.** Sampled tokens is the LLM-tradition convention used here. Counting prompt prefill too
  (§ 3.4) shrinks thinking's cost from about 3× to about 1.5× and turns 13–20 r2 seed 0 positive at 24k forward tokens
  and above (+2.8 to +3.4). r3 stays behind (−4.9 to −2.7; the 96k interval includes 0).
  - As a rough wall-clock check, the k = 32 thinking reads took about as long as the k = 64 baseline reads (13–20:
    37–52 against 44–61 min, on shared GPUs and including Lean checking). That is about 2× per attempt.
- **dev58 informed hard712's design.** hard712 was built from a gap analysis of textbook_dev (dev58), which is 58 of
  textbook72's 72 problems. The r2 and r3 models are trained on hard712, so textbook72 is a design set for them, not a
  clean test. The textbook72 uplift should be read with that in mind; 13–20 is the clean out-of-distribution test.
- **13–20 filtering.** 14 of the 1,000 13–20 items fail the prompt and reference check of `valid_1320` and are dropped
  from every 13–20 read.

## 6. What would change the verdict

- **Seed-1 round 3 (in progress).** If its 13–20 pair is positive at 8k–16k like r2 seed 1, the count becomes two
  positive of five, and the r2 → r3 decline in seed 0 would look like seed noise. If it is negative like r3 seed 0, the
  verdict hardens.
- **More seeds of the r2 pair.** A third and fourth seed of both arms would show whether the r2 seed-1 uplift is
  typical. Today seed variance dominates.
- **Larger k.** Thinking k = 64 and baseline k = 128 on 13–20 would extend both curves to about 40–60k tokens, and would
  test whether the rising difference crosses zero for r1, r2 s0 and r3.
- **A different cost model.** If inference cost is closer to forward tokens (prefill-heavy) or wall-clock than to
  sampled tokens, the 13–20 picture moves toward a tie or a small thinking win in three of four pairs; r3 stays behind
  (§ 3.4). Agreeing on the cost model matters as much as any new run.
- **A clean textbook-like test set.** If thinking's matched-token advantage on textbook72 held on a set that did not
  inform hard712 (for example `transfer_long`), the answer for textbook-like problems would be a qualified yes.
- **Cheaper thinking.** Training or forcing a lower think cost at the same per-attempt accuracy would move every curve
  left. For example, effort levels that the model actually adapts, or a token cost in RL. § 3.4 gives a rough preview of
  a 1.4–1.7× cost ratio on these models: on 13–20, two of four pairs are ahead at high compute, one is tied and r3 is
  still behind.

## Reproduce

```
ND_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice code/.venv/bin/python -m \
    code.experiments.current.thinking_traces_20261005.matched_eval.tts_figs --verbose
```

CPU only, about 30 s. The command writes both figures, their JSON and `matched_eval/tts_uplift_tables.md`. A pair whose
rows are not there yet (seed-1 round 3, further B0 reads) is skipped and appears on the next run.

## Maximum budget: coverage ceilings at 512 attempts (added 2026-10-09)

**Setup.**
- Leon asked whether both methods should get their maximum budget, since the baseline can always resample.
- Pair: round 2, seed 0 (thinking at B 64 vs the no-think baseline at B 0).
- Problems: a random 250 of the 986 valid 13–20 problems (`t1320_ceil250.jsonl`, seed 512).
- Budget: 512 attempts per problem for each method.
- Data: `runs/eval/ceil/`. Tokens per attempt: thinking 884, baseline 296.

**Equal attempts:**

| k | thinking | baseline | thinking − baseline |
|---|---|---|---|
| 1 | 32.6 [28.6, 36.8] | 26.1 [22.4, 29.7] | +6.5 [+2.8, +10.5] |
| 8 | 63.6 | 53.1 | +10.5 [+4.9, +16.3] |
| 32 | 76.8 | 65.7 | +11.0 [+5.5, +16.7] |
| 64 | 81.2 | 71.2 | +10.0 [+4.7, +15.5] |
| 128 | 84.5 | 76.1 | +8.4 [+3.2, +13.8] |
| 256 | 87.2 | 80.4 | +6.8 [+1.6, +12.1] |
| 512 | 89.2 [85.2, 92.8] | 83.2 [78.4, 88.0] | +6.0 [+0.4, +11.6] |

- At 512 each, 31 problems are solved only by thinking, 16 only by the baseline, 192 by both and 11 by neither.
- Neither curve has flattened. From 128 to 512 attempts thinking gains about 2.0 points per doubling and the
  baseline about 2.8.

**Equal sampled tokens per problem (thinking − baseline):**

| tokens | 1k | 4k | 16k | 32k | 64k | 128k | 150k |
|---|---|---|---|---|---|---|---|
| diff | −8.5 [−13.0, −3.7] | −1.8 [−7.0, +4.2] | +2.2 [−3.4, +8.0] | +2.7 | +2.4 | +2.3 | +2.5 [−2.9, +8.0] |

**Reading.**
- *Observed.* With every method at its maximum attempt budget, the thinking model reaches a higher ceiling (+6.0
  points at 512) and solves about twice as many problems the baseline never solves.
- *Observed.* Per sampled token the advantage is +2 to +3 points from 16k tokens per problem upward, but it is not
  significant with 250 problems. At low compute the baseline wins.
- *Interpretation.* The traces raise coverage per attempt. Their per-token edge at high compute is small. If the
  baseline's growth continues, it would match thinking's 512-attempt coverage at about 2,000 attempts (about 600k
  tokens, against 450k for thinking).
- *Caveat.* One seed, and 250 problems.
