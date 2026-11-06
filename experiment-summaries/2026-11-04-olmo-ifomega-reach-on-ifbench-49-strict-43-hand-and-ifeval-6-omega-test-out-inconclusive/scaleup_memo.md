---
written_on: 2026-11-06
written_by: agent:claude (executor, run olmo-ifomega; reviewed 2026-11-06, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-06).** This is the scale-up memo of run `olmo-ifomega`, written by the executor.
> The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-ifomega`, file `olmo_ifomega/scaleup_memo.md` at
> [`970e8b42`](https://github.com/dan-pandori/nd-takehome/blob/970e8b42/olmo_ifomega/scaleup_memo.md). The
> [review](https://github.com/dan-pandori/nd-takehome/blob/671bf753/review_olmo-ifomega.md) did not re-derive the memo's prices or token projections. Its next measurement is the memo's
> experiment 1 with three conditions, and for IFEval it calls the brevity control (experiment 2) "the test that
> matters". Its corrections are applied and marked *[review]*:
> - perp_069 is a reach prompt only if RL-verifier credits of the DPO model's answers do not count, and the
>   corrected-label perp_055 holds only under the doubly-credited reading (§Verdict reword item 1; C1);
> - geometry_basic_010: 0–1 sound of 7, not 1 (item 6; C0; C1);
> - gaming-only strict reach: 10 prompts by the review's rule (C1); the IFBench hand count: 43 in the review (C1);
> - ifeval_2447's reach rests on one borderline gamed label (item 8), and the IFEval miss distances are post hoc and
>   not re-derived (C1);
> - experiment 1: the review's three conditions (§Verdict, "Next measurement").

# Scale-up memo — olmo-ifomega (2026-11-06, executor)

Models: `allenai/Olmo-3-32B-Think-DPO` (the RL start) and `allenai/Olmo-3.1-32B-Think` `step_0750` / `main`
(= `step_2300`); Think RL decoding (T 1, top-p 1, 32,768 tokens). Prices are this run's: 1 × H200, $3.59/h community
(≈ 1,000–1,800 generated tokens/s, ≈ $0.6–1.0 per M tokens) and $5.29/h secure when community stock is out.

## What the run established

- **IFBench: reach.** 42 headline prompts (hand; 49 strict; the review's blind re-read gives 43, 40 of them shared, ± 2 reader noise *[review]*) where `step_2300` is confirmed with genuine answers and
  DPO has none in 256 samples. These are the scale-up set for instruction following (`olmo_ifomega/candidates.csv`,
  `verdict.csv` `reach_s2300`).
- **OMEGA `test_out`: reach only at RL's token cap.** perp_068 and perp_069 (DPO 0 doubly-credited / 256 *[review]*, cut at 32,768 tokens on 255
  of 256; `step_2300` 21 / 32), and perp_055 if its label is corrected (17 → 15). On perp_069, RL's own verifier credits 2 of DPO's 256 samples, both read as sound derivations (neither states an answer after `</think>`: draw 69 stops mid-word in its answer section at 30,198 tokens, draw 237 is cut at the cap). So perp_069 is a reach prompt only if `rl`-only DPO credits are not DPO successes; counting them, the hand count is 1 (perp_068), and the corrected-label third holds only under the doubly-credited reading *[review]*. DPO's geometry failures are 99 %
  cut at the cap, so whether this is reach or a length budget is not decided by this run.
- **IFEval** (in-distribution control): **reach too** (strict and hand), 6 prompts, all letter-avoidance or all-caps
  constraints of types RL trained on (ifeval_2447's reach rests on one borderline gamed label: DPO's only strict success there is a poem written entirely in Chinese, which meets "fewer than 2 w's" without attempting the lipogram *[review]*). DPO misses letter avoidance by far (median 44–97 uses of the letter) and slips
  into lowercase on all caps; `step_2300` passes with 5–15× shorter answers (`olmo_ifomega/ifeval_misses.csv`; descriptive, added after the verdict, not pre-registered (deviation 19), and not re-derived by the review *[review]*).

## Next experiments, cheapest first

1. **Token budget on the geometry reach prompts** (decides "reach at the cap" vs reach): DPO on perp_055 / 068 / 069
   with a 65,536-token budget, 64 samples each: ≈ 192 × 50k ≈ 10 M tokens ≈ $8–12 (one H200, ≈ 3 h; KV-bound at long
   context). If DPO then solves them, the `test_out` geometry gap is length, not reach. The review's version: grade with both `omega` and `rl`, with 055's label corrected, and hand-read every credited response, `rl`-only ones included *[review]*.
2. **Brevity control** (is IF reach a length habit?): DPO with a length hint ("answer in at most 60 words") on the
   three letter-avoidance IFEval reach prompts and a seeded 10-prompt IFBench reach subset, 64 samples each: ≈ 13 × 64
   × 4k ≈ 3.3 M tokens ≈ $3 plus hand checks. If DPO then succeeds, those prompts measure brevity, not new ability.
3. **Deeper sampling on the IFBench reach set** (does reach survive 1,024 samples?): DPO on the 42 hand-reach prompts
   to 1,024 (768 more each): 42 × 768 × 5.4k ≈ 174 M tokens ≈ $110–170. A 10-prompt seeded subset ≈ $30–40.
4. **The intermediate checkpoints' reach** (cut here): `step_0750`'s confirmation on the IFBench reach set (fresh
   chunks of 16 until 2 genuine): ≈ 42 × 32 × 5.6k ≈ 7.5 M tokens ≈ $5–7, plus hand checks.
5. **OMEGA `test_in` and `step_0100` beyond IFBench** (cut): `test_in` fixed phase for all four models ≈ $90 (long
   OMEGA responses); `step_0100` on IFEval / `test_out` ≈ $3 / $25 (its IFBench fixed phase ran at the end: $9.64).

## Cautions for any scale-up

- **Checker gaming:** 36 % of `step_2300`'s first-checked strict IFBench successes are gamed (minimal answers that
  meet the constraint and drop the content); on 11 prompts its strict reach is gaming only (10 by the review's rule, 8 of them on both lists *[review]*). Any IFBench reach count
  needs the hand gate (rules in `olmo_ifomega/handcheck_rule_if.md`).
- **Label errors:** one of 14 `test_out` candidate labels is wrong (perp_055); check every deciding label.
- **Rounding-boundary answers:** geometry_basic_010's answer (13.1549) lies 0.00007 below a rounding boundary; 6 of
  DPO's 7 credited answers reach "13.15" through cancelling slips or a guess (7 of 7 in the review's reading, and its 2 `rl`-only credits too: the seventh estimates FE only to ± 0.003, 0.000075 from the boundary, so `dpo` has 0–1 sound of 7 *[review]*). Grade decimal answers by hand there.
- **Cost driver:** long OMEGA responses fill the KV cache (≈ 30 sequences at 32k tokens on an H200), so geometry
  problems cost ≈ $1.9 per 48 samples; budget them per sample, not per prompt.
