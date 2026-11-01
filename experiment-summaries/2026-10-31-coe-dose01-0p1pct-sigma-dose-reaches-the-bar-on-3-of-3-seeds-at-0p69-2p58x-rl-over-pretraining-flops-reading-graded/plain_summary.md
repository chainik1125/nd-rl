---
written_on: 2026-11-01
written_by: agent:claude (executor, run coe-dose01; reviewed 2026-11-01, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-01).** This is the plain-language summary of run `coe-dose01` (fork
> `dan-pandori/nd-takehome`, branch `dan_coe-dose01`, DONE
> [`ec488738`](https://github.com/dan-pandori/nd-takehome/commit/ec488738), file `summary_coe_dose01.md`), written by
> the executor, with the corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/4c932eec/review_coe-dose01.md) applied. Each change is
> marked *[review]*:
> - "Full models start at the bar" holds for their median, which is how the bar is set; 3 of the 6 full models start
>   below it (review, Verdict item 1);
> - "RL found the trace and amplified it" is replaced: seed 1's base wrote no free proof by contradiction, and the
>   training mix also replays about 20 σ proofs per round, so without a replay-only control the rise cannot be
>   attributed to RL alone (item 2);
> - two caveats the review asks the write-ups to add: one σ draw per dose, and attempts cut off by length limits.
>
> The corrections take it from about 600 to about 730 words, against the brief's 600. The executor's text, byte for
> byte, is this file at commit `c53febd`. See the [summary](README.md) for the reviewed numbers.

# How much RL does it take to bring back a reasoning move that pretraining barely showed? (plain-language summary)

**The setting.** We train small (9.56 M-parameter) theorem provers from scratch on 155,000 propositional-logic proofs,
then improve them with reinforcement learning (expert iteration: the model attempts problems, Lean checks each attempt,
and the model trains on its accepted proofs).

**The move.** Some theorems can only be proved by contradiction: assume the goal is false, derive a contradiction,
conclude the goal (σ, Lean's `Classical.byContradiction`). An earlier run (create-or-elicit) varied how much of the
pretraining data used σ:
- **full**: σ in 10.9 % of the proofs;
- **1 %**: σ in 1 % of the proofs;
- **knockout**: σ removed entirely.

The test: 106 held-out theorems that need σ, never trained on. The bar: what full-exposure models solve before any
RL, 14 of 106.

**What was known.** Full models start at the bar *[review]* on the median, which defines the bar; 3 of the 6 start
below it and reach it in round 2, at 0.39–0.43 times their pretraining compute. The 1 % models reach it after RL costing 0.4–0.7 times their
pretraining compute; the knockout models not even after 6–10 times. Two doses and a lower bound cannot tell a cliff
(a threshold) from a slope.

**This run** added **0.1 % σ**: 155 of the 155,000 pretraining proofs. Everything else copied the 1 % arm (three
seeds, sixteen rounds).

**What happened.**
1. **Before RL, the 0.1 % models barely used the move.** In their first 32 attempts per problem they wrote between 0
   and 2 "free" proofs by contradiction across 4,495 problems (knockout: 0; 1 % models: 170–219). They solved 0, 0 and 3
   of the 106 held-out theorems (1 % models: 3, 13, 2). Their accuracy on a 5,000-theorem general test (0.84–0.89) was
   in or below the knockout models' range (0.87–0.90), not near the 1 % models' (0.92–0.95).
2. **During RL the move became common on every seed** *[review]*. The free proofs by contradiction grew to 464–1,116 per
   round by round 16 *[review]* (seed 1's base wrote none, and with about 20 σ proofs replayed per round the rise
   cannot be attributed to RL alone without a replay-only control). The models crossed the bar at round 3, 5 and 8. In compute terms that is **0.69, 1.37 and 2.58 times their
   pretraining compute** (median 1.37×). The 1 % models needed 0.39–0.73×.
3. **By round 16** the models solved 30, 28 and 19 of the 106 (1 % models: 58, 34, 30; knockout: 0–3). By the plain reads, all
   three acquired converse contraposition during RL; two also acquired the negated conditional.
4. **The knockout model given extra RL** (seed 1, rounds 25–32) went from 5 to 6 of 106 by 13 times its pretraining
   compute.

**What it means.** The pre-registered reading is a **graded response**: at a tenth of the dose, the median RL needed
was about three times the 1 % arm's (descriptive: three seeds, and they overlap), but RL still got there. Removing σ entirely
moved the requirement past everything we tried (over 13× on the one knockout seed run that long); 0.01 % is untested. *Interpretation, not tested here:* this fits the
"elicitation" picture — RL amplified a behaviour the base showed only at a trace level (0–2 distinct free proofs by
contradiction among 143,840 first-round attempts), and needed more compute the rarer the trace was.

**Caveats.**
- Three seeds per dose. The 0.1 % vs 1 % differences (final scores and crossing compute) are descriptive at this n;
  only the contrast with the knockout is statistically resolved (3 / 3 vs 0 / 6 seeds crossing, p = 0.003).
- RL also replays pretraining data, which for the 0.1 % arm includes about 20 σ proofs per round; some of the recovery
  may come from that replay.
- *[review]* One draw of σ proofs per dose: all three 0.1 % seeds trained on the same 155, drawn separately from the 1 %
  arm's 1,550 (17 shared), so the 0.1 % vs 1 % contrast mixes dose with draw.
- *[review]* Attempts cut off by length limits count as failures (up to 3.1 % of a late round's test attempts, a lower
  bound; 20.7 % on excluded middle in one final read), so counts on long-proof families can only be too low. The 1 %
  arm had the same limits.
- One small model, one RL method, one family of logic problems.

**Cost:** $39.35 of $45 (four rented A40 GPUs).
