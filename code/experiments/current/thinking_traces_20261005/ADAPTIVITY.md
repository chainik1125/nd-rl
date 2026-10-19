---
written_on: 2026-10-06
written_by: agent:claude
---

# Is the thinking adaptive? Think length and thinking gain against difficulty

Question (Leon, 2026-10-06): under GRPO the training-batch pass@1 shows little gain from thinking
(iteration 13: e0 78%, e16 58%, e64 74%, e256 77%) and think length is shrinking. Does the model think
*more* on harder targets, and does thinking *help more* there? How does that change under GRPO and
under expert iteration (EI)?

## Status (2026-10-07 05:51 UTC): complete

All three checkpoints are measured at all four levels: 672 targets, k = 16. The full tables are in
[`adaptivity/compare_3ckpt.md`](adaptivity/compare_3ckpt.md). The new reads are in
[Results: all three checkpoints](#results-all-three-checkpoints-observed). The status note from 01:05 UTC
follows, kept for the record.

## Status (2026-10-07 01:05 UTC): partial

Measured: checkpoint 1 (`grpo_T1v1e_it0`, the effort-expanded Stage-1 T1v1 init) at **e0 and e16**, all
672 targets, k = 16. **Not measured: e64 and e256 for it0, and GRPO it39 and EI r1 at any level.**
Both checkpoints were on disk (it39 at 23:25, r1 at 23:47 UTC). The measurement stopped under its GPU rule:
it waits at most 60 min for GPU 0 to hold only our `code.thinking.train` jobs with more than 12 GB free.
From 00:03 UTC GPU 0 also held another agent's calibration eval (`test_21_30/calib.sh`: T1v1_s0, 2,775
targets × k 8 at B 256, 15.9 GB), still running at 01:04. Earlier, a textbook72 eval of T1f (22:45–23:27)
and the yields to Leon's GPU-0 training queue (T1f → T0g → T1v1g) had used the slack. To resume: the
runner skips finished levels.

```
C=/scratch/cowifornia/nd-rl/think/runs/restart_20261006; A=/scratch/cowifornia/nd-rl/think/runs/adaptivity
cd $A && nohup python3 runner.py --job grpo_T1v1e_it0=$C/grpo_T1v1e/ckpts/grpo_T1v1e_it0.pt \
  --job ei_T1v1e_r1=$C/ei_T1v1e/ckpts/ei_T1v1e_r1.pt \
  --job grpo_T1v1e_it39=$C/grpo_T1v1e/ckpts/grpo_T1v1e_it39.pt >> runner.log 2>&1 &
# then, from the repo root:
python -m code.thinking.adaptivity compare --out $A/compare.md \
  --runs "it0=$A/grpo_T1v1e_it0,r1=$A/ei_T1v1e_r1,it39=$A/grpo_T1v1e_it39"
```
`runner.py` (in that directory) applies the GPU-0 rules before each level, yields to the training queue,
wraps each level in `timeout` (at most 60 min), and stops if the gate is not met within 60 min.

At full scale, while sharing GPU 0 with a training job, one level took 11–15 min (e0 15.2 min, e16 11.4 min).
From the pilot ratios, e64 and e256 should take about 20–30 min each, so about 75 min per checkpoint.

**What it0 is.** `expand_to_effort` adds `<e0..e256>` with new embedding rows drawn from the old rows'
Gaussian, and it0 is saved before any GRPO update. So **it0 has never been trained to read the effort
token**: at it0, "e16" means Stage-1 T1v1's own thinking cut at the first line end past 16 tokens, plus an
untrained token in the input. It is a forcing baseline, not effort-conditioned behaviour.

## Results: it0 (observed)

pass@1 / pass@5, % of targets, 95% bootstrap CI over targets:

| level | all (672) | RL pool (600) | textbook dev (58) | textbook train (14) |
|---|---|---|---|---|
| e0 | 24.1 [21.9, 26.2] / 49.1 [45.8, 52.3] | 26.1 / 53.4 | 6.0 / 11.5 | 11.6 / 20.2 |
| e16 | 13.2 [11.6, 14.9] / 30.3 [27.3, 33.2] | 14.0 / 32.4 | 4.2 / 8.9 | 17.4 / 27.5 |

e16 by difficulty. Think tokens per block and per attempt at e16; gain = pass@1(e16) − pass@1(e0) in
percentage points, paired bootstrap over targets:

| bin | n | think/block | think/attempt | blocks/attempt | pass@1 e0 | pass@1 e16 | gain e16 − e0 |
|---|---|---|---|---|---|---|---|
| T0 = 0 | 150 | 20.4 [18.9, 22.1] | 101 [94, 109] | 5.5 | 10.0 | 6.8 | −3.2 [−5.9, −0.5] |
| T0 (0, .25] | 150 | 18.4 [17.4, 19.5] | 115 [108, 123] | 6.8 | 19.5 | 11.4 | −8.1 [−11.8, −4.8] |
| T0 (.25, .5] | 150 | 17.4 [16.3, 18.7] | 120 [112, 129] | 7.4 | 30.9 | 18.2 | −12.7 [−16.9, −8.5] |
| T0 (.5, .625] | 150 | 16.3 [15.2, 17.4] | 105 [98, 114] | 6.8 | 44.1 | 19.4 | −24.7 [−29.6, −19.6] |
| length ≤ 12 | 300 | 19.0 [17.9, 20.1] | 106 [102, 111] | 6.3 | 29.1 | 17.5 | −11.6 [−14.6, −8.5] |
| length 13–20 | 300 | 17.3 [16.6, 18.0] | 115 [109, 120] | 6.9 | 23.1 | 10.4 | −12.7 [−15.5, −10.2] |
| textbook dev | 58 | 15.2 [13.6, 16.9] | 68 [57, 80] | 4.6 | 6.0 | 4.2 | −1.8 [−4.4, +0.2] |
| textbook train | 14 | 15.4 [12.6, 19.1] | 50 [40, 60] | 3.5 | 11.6 | 17.4 | +5.8 [+0.0, +15.2] |
| all | 672 | 17.8 [17.2, 18.4] | 106 [102, 110] | 6.4 | 24.1 | 13.2 | −10.9 [−12.9, −9.1] |

Gain by the checkpoint's own e0 pass rate (split halves, see Method): own e0 = 0 (302 targets) +0.2
[−1.0, +1.4]; (0, .25] −8.5 [−12.5, −4.5]; (.25, .5] −19.8 [−25.3, −14.2]; (.5, .75] −35.1 [−42.4, −27.9];
(.75, 1] −36.4 [−46.0, −25.7]. The full 8-cell T0 × length table is in
`/scratch/cowifornia/nd-rl/think/runs/adaptivity/grpo_T1v1e_it0/adaptivity.md`.

Spearman ρ at e16, per-target means, 95% CI:

| y ~ x | ρ |
|---|---|
| think/block ~ T0 pass@1 (600 RL targets) | −0.21 [−0.29, −0.13] |
| think/block ~ proof length (600) | −0.08 [−0.16, +0.00] |
| think/block ~ own e0 pass@1 (672) | −0.14 [−0.21, −0.07] |
| think/attempt ~ T0 pass@1 (600) | +0.06 [−0.02, +0.14] |
| think/attempt ~ proof length (600) | +0.07 [−0.01, +0.15] |
| think/attempt ~ own e0 pass@1 (672) | +0.24 [+0.17, +0.31] |
| gain e16 − e0 ~ T0 pass@1 (600) | −0.33 [−0.40, −0.25] |

Think by attempt outcome at e16: per block, accepted 12.3 vs failed 21.1 (pooled). Within target
(targets with both outcomes), failed minus accepted is +8.9 [+7.8, +10.1] tokens per block, and −15.7
[−22.4, −9.0] per attempt (failed attempts are shorter). 31% of accepted attempts' blocks and 42% of failed
attempts' blocks hit the forced close. Pass rate by quartile of think per block (pooled): 20.7%, 21.8%,
10.0%, 0.5% (the top quartile is 26.7–80 tokens per block, so these are blocks that ran past the budget to
the line end).

**e256 pilot (small; same checkpoint, settings and seed; 40 targets = every 17th of the set, 36 of them
RL).** Pass@1 e0 29.4%, e256 39.1%; gain +9.7 pp [+1.9, +17.7]. By T0 bin (9 targets each): T0 = 0 +0.0
[−2.8, +3.5], (0, .25] +13.9 [+0.7, +27.8], (.25, .5] +22.9 [+10.4, +36.8], (.5, .625] −2.8 [−21.5, +16.0].
Think per block about 50 tokens, with ρ(think/block, T0) −0.48 [−0.72, −0.17] (n = 36). Within target,
failed minus accepted is +14.8 [+6.0, +24.3] tokens per block. Pass rate by think/block quartile: 50, 59,
37, 11%. With n = 9 per bin this only indicates direction.

## Interpretation (it0 only; may be wrong)

- **Think length is weakly adaptive per step, not per attempt.** Per block, the model writes a little more on
  harder targets (20 vs 16 tokens from the hardest to the easiest T0 bin at e16, ρ −0.21; the e256 pilot
  points the same way, more strongly). Per attempt there is no adaptivity: think per attempt tracks how many
  steps an attempt survives, and easier targets survive longer (ρ +0.24 with own e0 pass rate).
  **Confound:** rows store only per-attempt totals, not per-step lengths. Attempts on hard targets fail
  earlier, so their per-block mean weighs early steps more. If early steps think longer, part of the
  per-block difficulty effect and of the failed-vs-accepted gap is position, not adaptivity. A per-step
  read (`--dump_traces` keeps only accepted attempts) would separate the two.
- **Thinking gain at e16 is not adaptive in the useful sense.** Forcing T1v1's thinking to about 16 tokens
  hurts on every RL bin, and hurts least on the hardest only because there is least to lose (T0 = 0: 10.0 →
  6.8%). Targets the checkpoint never solves at e0 stay unsolved at e16 (+0.2 pp). This matches the
  training-batch e16 dip at GRPO iteration 13. At it0 it is a truncation effect, since the effort token is
  untrained.
- **Long thinks go with failure.** Within a target, failed attempts think about 9 more tokens per block at
  e16, and attempts whose blocks run past the budget almost never succeed. This reads as "thinking that
  runs on is a sign of being lost", not "more thinking helps", subject to the position confound above.
- **The e256 pilot hints at the shape a full run should test:** a gain concentrated on middle-difficulty
  targets (T0 in (0, .5]), nothing on T0 = 0 targets, and nothing on the easiest bin. That would mean thinking
  helps on targets the model can already sometimes do, which is not the same as adapting to difficulty.
- **Under GRPO and EI: not measured.** GRPO's training-batch think at e256 fell from 103 tokens per attempt
  (iteration 13) to 30 (iteration 38). EI's `--select shortest` adds a shortness pressure. Both are
  reasons to expect less think per block at it39 and r1. Whether adaptivity survives is the open
  measurement.

## Results: all three checkpoints (observed)

**Checkpoints.** All three come from the first RL restart (`restart_20261006`, pool `rl_hard_filtered`):
- `it0`: the effort-expanded T1v1 init, which has never read an effort token;
- `ei_r1`: expert iteration, round 1;
- `grpo_it39`: GRPO, iteration 39.

**Caveat on `grpo_it39`.** That run used the GRPO loss that scored the full vocabulary while the sampler
masked tokens; see [GRPO_COLLAPSE_AUDIT.md](GRPO_COLLAPSE_AUDIT.md). At it39 the banned mass was only about 2e-4,
but the loss pushed against think tokens throughout. Its near-absent thinking may be partly that bug.

pass@1 / pass@5 (%) over all 672 targets:

| level | it0 | ei_r1 | grpo_it39 |
|---|---|---|---|
| e0 | 24.1 / 49.1 | 62.0 / 78.9 | 77.9 / 83.4 |
| e16 | 13.2 / 30.3 | 36.3 / 61.3 | 78.5 / 85.5 |
| e64 | 18.9 / 42.1 | 51.2 / 77.4 | 79.0 / 86.9 |
| e256 | 38.6 / 69.4 | 69.7 / 88.7 | 79.5 / 88.1 |

At e256, all targets:

| measure | it0 | ei_r1 | grpo_it39 |
|---|---|---|---|
| think tokens per attempt | 420 | 484 | 41 |
| think tokens per block | 45.6 | 41.7 | 4.3 |
| pass@1 gain over e0 (pp) | +14.5 [+12.6, +16.3] | +7.8 [+5.7, +9.8] | +1.6 [+0.7, +2.6] |
| Spearman ρ, think per attempt ~ T0 pass@1 | −0.19 | −0.38 | −0.32 |
| Spearman ρ, think per attempt ~ own e0 pass@1 | −0.23 | −0.43 | −0.42 |

e256 gain over e0 (pp), binned by the checkpoint's own e0 pass@1 (split half):

| own e0 | it0 | ei_r1 | grpo_it39 |
|---|---|---|---|
| 0 | +16.2 | +23.0 | +10.8 |
| (0, .25] | +21.5 | +31.0 | +15.3 |
| (.25, .5] | +15.4 | +15.9 | +12.2 |
| (.5, .75] | +1.4 | +8.1 | −1.2 |
| (.75, 1] | −4.4 | −4.5 | −1.5 |

**Low budgets hurt.** e16 is below e0 at it0 (−10.9 pp) and more so at ei_r1 (−25.7 pp). At grpo_it39 the
levels are level, because the policy hardly thinks.

**Long thinks go with failure.** Within a target, failed attempts write about 16 more think tokens per
block than accepted ones at it0 and ei_r1. Pass rate by think-per-block quartile, Q1 to Q4:
- it0: 53 / 51 / 36 / 14%;
- ei_r1: 83 / 79 / 70 / 47%.

### Interpretation (may be wrong)

- **Think allocation is adaptive at e256, and RL strengthens it.** Every checkpoint thinks more per attempt
  on targets it finds harder. The correlation roughly doubles after RL (ρ −0.23 to −0.43 against its own e0
  difficulty).
- **The thinking gain is where it should be.** It is largest on targets the checkpoint rarely solves
  without thinking (own e0 in (0, .25]: +15 to +31 pp) and slightly negative on the ones it nearly always
  solves. So a policy that thinks only when unsure would beat always-thinking. The current models do not
  switch off on easy targets. On the easiest T0 bin, (.5, .625], they still write 356–370 think tokens
  per attempt at it0 and ei_r1.
- **EI round 1 widens the gap between low and high budgets.** e256 improves while e16 collapses relative to
  e0. Writing a few think tokens is worse than writing none, plausibly because EI trains mostly on e256
  successes and short thinks are off-distribution.
- **GRPO it39 learned to barely think.** About 4 tokens per block, gain +1.6 pp, yet it still allocates
  more to harder targets. Given the bug above, this is not yet evidence that RL removes thinking. The fixed
  GRPO relaunch is the test.

## Method

**Problem set (fixed, 672 targets).** [`adaptivity/adapt_set.jsonl`](adaptivity/adapt_set.jsonl), with labels in
[`adaptivity/adapt_set.labels.json`](adaptivity/adapt_set.labels.json). Built by
`python -m code.thinking.adaptivity build-set` (seed 0):
- 600 targets from `/scratch/cowifornia/nd-rl/think/runs/pools/rl_hard_filtered.jsonl`: 75 per cell of
  T0 pass@1 bin {0, (0, .25], (.25, .5], (.5, .625]} × generated proof length {≤ 12, 13–20}. T0 pass@1 is
  the pool filter's per-record `t0_pass1` = `t0_n_ok / 16`, measured with the Stage-1 T0 model
  (`stage1/T0_s0`), k = 16, temperature 0.8, plain arm, no think. Length is the record's `n_lines`
  (the generator's proof length, not a minimal length).
- All of textbook72: 58 dev targets (`reference_lines` kept) and 14 train targets.

**These RL-pool targets are training targets.** GRPO and EI both train on `rl_hard_filtered.jsonl`.
GRPO it39 has seen about 39 × 256 prompts from the 5,000-target pool, and EI r1 has sampled and
fine-tuned on it. So this measures adaptivity on targets the policy may have been trained on, not
generalisation. textbook72 dev is the only part outside the RL pool. textbook72 train is in Stage-1's
mixture, not the RL pool.

**Sampling.** `python -m code.thinking.adaptivity --ckpt CKPT --sets adapt_set.jsonl --labels
adapt_set.labels.json --k 16 --batch 512 --out RUNDIR`. That runs one `code.thinking.eval` per effort
level B ∈ {0, 16, 64, 256}: plain arm, temperature 0.8, line forcing (close at the first line end past
B, hard cap B + 64), seed 1, eval defaults otherwise (max_steps 96, max_action 512), and Lean judges
every finished proof. The sampler prefills the effort token `<eB>`. Outputs:
`/scratch/cowifornia/nd-rl/think/runs/adaptivity/<ckpt>/` (`eval/B<B>/rows_plain_B<B>.jsonl`,
`adaptivity.json`, `adaptivity.md`).

**Measures.**
- *Think per attempt*: think tokens summed over every block of an attempt, including the draw that
  ended a failed attempt. A longer proof has more blocks, so this rises with proof length even at a
  fixed per-step budget. *Think per block* = think / blocks, a ratio of sums per target, separates
  "thinks longer per step" from "takes more steps".
- *Gain*: per target, pass@1(eB) − pass@1(e0) from k = 16 attempts each, averaged over the bin's
  targets. The 95% interval is a percentile bootstrap over targets (2,000 draws), paired: both levels
  share each resample.
- *Own e0 difficulty*: the checkpoint's own e0 pass@1. For the gain by own-e0 bin, the bins come from
  e0 attempts 0–7 and the e0 side of the gain from attempts 8–15. Binning and measuring on the same
  attempts would build in regression to the mean (an e0 = 0 target can only gain).
- *Spearman ρ* between a per-target mean and a difficulty measure, with a 95% interval from a target
  bootstrap.
- *Think by outcome* at each level: pooled over attempts, and within target (targets with both
  accepted and failed attempts: failed minus accepted), so that "hard targets think longer *and*
  fail more" does not pass for "long thinks fail".

Code: [`code/thinking/adaptivity.py`](../../../thinking/adaptivity.py) and its test
[`code/tests/thinking/test_adaptivity.py`](../../../tests/thinking/test_adaptivity.py) (synthetic rows).
Branch `leon/thinking`, HEAD `8644bf4`, with these files uncommitted.
