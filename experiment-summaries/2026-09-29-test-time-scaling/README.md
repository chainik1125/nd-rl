---
written_on: 2026-09-29
written_by: agent:claude
---

# Test-time scaling: simple methods on c005

Status: done. The holdout was scored once per configuration, on 3 seeds.

- **Working folder:** code, plan and job lists are in
  [test_time_scaling](../../code/experiments/current/test_time_scaling/test_time_scaling_initial_plan.md),
  branch `robbie-experiments` (developed in checkout `nd-rl-8`).
- **Raw results:** `artifacts/test_time_scaling/hold_{a,b}_s{0,1,2}/`, which are not committed.
- **Chart data:** [charts/scaling.csv](charts/scaling.csv).

Question: Robbie (2026-09-29), "get some notion of test time scaling curves on our RL problem",
starting with the simple methods before anything fancy:

- 0: plain sampling;
- 1: resample invalid lines;
- 2: beam search and sequential Monte Carlo (SMC);
- 3: line-level search.

No expert iteration and no learned value head. For each method report pass@k up to 64, inference
time, and the long-proof (> 10 lines) numbers separately.

![Solve rate vs sampled tokens per theorem, all theorems and the 114 longer than 10 lines](charts/scaling.png)

## Configuration

**Model.** Combined-pipeline run c005: 186-claude pretraining, Leon's GEAR expert iteration and
guided RL sampling, `lean_seq`. Checkpoints `runs/combo-c005_s{0,1,2}/final.pt`. Every number is
a mean over the 3 seeds.

**Pool.** The holdout half of the fork's `data/ladder/transfer.jsonl` (fork `51604b3`):

- 1,177 theorems of reference length 7-14;
- **114 of them longer than 10 lines**.

The dev half was used only for a 300-theorem calibration on seed 0.

**Sampling constants.**

- T = 0.8 and 512 kept tokens per proof, as in the harness eval.
- A proof counts only if Lean 4 and nd_verify both accept it.
- A line is one `;`-terminated Lean tactic segment, checked with the fork's `inverse` parser and
  Leon's `LineChecker` (as `guided.py` does).

**Compute measure.** Sampled tokens per theorem: every token drawn from the model, including
rejected segments. Re-encoding a prefix (prefill) is not counted there; it shows up in wall-clock.
Wall-clock is measured on one A40 and includes the Lean gate.

**Methods** (full definitions in the plan):

| method | what it does | compute knob |
| --- | --- | --- |
| plain | i.i.d. samples: the current eval | k = 1..64 |
| resample | method 1: an invalid segment is redrawn from the same prefix; the sample is abandoned at its 11th rejection (cap 10) | k = 1..64 |
| beam | method 2a: E = 4 segments per beam node; keep the top B valid prefixes by log-prob | B = 1..16 |
| smc | method 2b: N particles; invalid segments get weight 0; survivors resampled back to N | N = 2..64 |
| best-first | method 3: a frontier of every valid prefix, scored by log p / segments^0.5; E = 8 per pop; anytime | budget up to 16k tokens |

**Implementation checks.**

- Plain sampling on the full dev half reproduces c005's recorded metric: 914 vs 915 (seed 0).
- Resample read at cap 0 matches plain sampling, as it must, since every rejected line fails the
  verifier anyway. pass@1 is 31.0 vs 31.2 and pass@64 is 80.5 vs 80.8.
- The combined pipeline's own KV-cache sampler, `guided_kv` (cap 10), reproduces my engine's cap-10
  pass@64 within 0.2 points on every seed.

**Cost.** About $5.70 of A40 time: 6 pods × ~107 min, plus a 50-min calibration pod.

## Results (holdout, 3-seed means)

### pass@k

| k | plain % | resample % | plain, > 10 lines (of 114) | resample, > 10 lines |
| --- | --- | --- | --- | --- |
| 1 | 31.2 | 57.4 | 6.6 | 19.5 |
| 4 | 60.5 | 76.5 | 18.0 | 35.6 |
| 16 | 75.2 | 82.6 | 31.9 | 47.1 |
| 64 | 80.8 | 85.6 | 40.7 | 54.3 |

### Compute-matched comparison

Budget is about 14-16k sampled tokens per theorem, i.e. 64 plain samples.

| method | setting | tokens / theorem | solved % | > 10 lines solved | wall s (all 1,177) |
| --- | --- | --- | --- | --- | --- |
| plain | k = 64 | 13,971 | 80.8 | 40.7 | 407 |
| resample, engine | k = 32 | 10,759 | 84.3 | 51.1 | ~1,220 (half the k = 64 run) |
| resample, engine | k = 64 | 21,518 | 85.6 | 54.3 | 2,445 |
| resample, `guided_kv` | k = 64 | not counted | 85.5 | 53.0 | 610 |
| smc | N = 64 | 16,147 | 84.4 | 49.0 | 1,070 |
| smc | N = 32 | 7,968 | 82.9 | 46.0 | 532 |
| beam | B = 16 | 6,784 | 76.6 | 40.0 | 537 |
| best-first | 16k budget | 6,800 | 75.8 | 34.3 | 450 |

Wall-clock notes:

- Plain's time varies by pod (290-480 s across seeds).
- The engine re-encodes every prefix at every step. For resample that is 122M prefill tokens against
  9.8M sampled on the 300-theorem calibration. That overhead is why the engine runs cost 4× the
  KV-cache implementation for the same method.
- `guided_kv` is the real cost of resampling: **1.5× plain's wall-clock for +4.7 points overall
  and +30% on long proofs.**

### Where resampling's gain comes from

Numbers are with cap 32, which records every rejection, so any smaller cap can be read off the
same run.

- **Rejection rate.** 69% of samples draw at least one invalid segment. 58-61% of all checked
  segments are rejected, about 14 rejections per sample.
- **Causes:**
  - 65% a rule misapplied (the cited lines don't license the formula);
  - 31% outside the Lean grammar;
  - 3% a premature `exact`/end;
  - 1% a segment over 400 tokens.
- **Repeats.** **50-55% of rejections redraw a segment already rejected at the same prefix.** The
  policy is confidently wrong, and resampling from the same distribution returns the same wrong
  line.
- **Salvage.** At cap 10, 45-47% of all successful samples had at least one line rejected and
  redrawn, about 19,900 salvaged samples per seed.
- **Theorem level at k = 64.** Resample solves 1,006-1,009 theorems vs plain's 947-954. It solves
  55-61 that plain misses and misses only 1-2 that plain solves. About 168 theorems per seed are
  solved by neither.
- **Choice of cap** (pass@64): 0 → 80.5, 1 → 82.3, 2 → 83.6, 4 → 84.5, 10 → 85.6, 32 → 86.5.
  Most of the gain arrives by cap 4. At matched tokens, caps 2-32 sit on roughly one curve.

## Observations

1. **Resampling invalid lines is the best method at every budget measured, and the gain is
   largest on long proofs and at small k.**
   - pass@1 goes from 31.2% to 57.4%.
   - At about 11k tokens per theorem it solves 84.3% vs plain's 80.8% at 14k.
   - On proofs longer than 10 lines: 51.1 vs 40.7 theorems.
   - It is also cheaper per sample than plain at cap 0, because failing samples stop at the first
     bad line (158 vs 218 tokens per sample).
2. **SMC is second, just behind resampling.** Both target the same distribution (the policy
   conditioned on valid lines). SMC's correction for resampling's local bias buys nothing measurable
   here, and particle collapse costs it on long proofs (49.0 vs 54.3 solved beyond 10 lines).
3. **Beam and best-first search are no better than plain sampling at matched tokens.**
   - Beam flattens from B = 8: 76.0% → 76.6%.
   - Best-first solves under 5% of theorems within 1k tokens, because it expands short
     high-probability prefixes breadth-first before finishing any proof. It only catches plain near
     its 16k budget.
   - Log-prob is a bad ranking signal for this policy, consistent with the literature on
     log-prob-guided search without a value function.
4. **All curves flatten.** Plain gains 2.3 points from k = 32 to 64, resample 1.3. 137-147
   theorems per seed (about 12%) stay unsolved by every method. These curves are not steep enough to predict
   the unsolved set will fall to more compute.

Interpretation, not tested: most plain-sampling failures are *local* errors, a wrong citation or
rule on one line, not dead-end proof plans. A cheap line check fixes those. What remains looks like
theorems where the policy has no working plan, which is where a value function or training (expert
iteration on the failures) would be needed.

## Follow-up A: resample without repeats

Robbie (2026-09-29) asked whether the model could be stopped from redrawing a line it was just
rejected for. This is sampling without replacement over segments (UniqueRandomizer, Shi et al.
2020), in `engine._swor_shift`:

- Every rejected segment is stored with its per-token probabilities.
- On a redraw at the same prefix, each next token's probability is multiplied by the share of its
  mass not already spent on rejected segments.
- A token that would complete a rejected segment gets probability 0.
- Each redraw is therefore an exact sample from the model with the rejected lines removed.

It runs as `ttscale.py --method guided --swor`, on the same holdout and seeds, at cap 32 (read at
every smaller cap).

| cap | resample pass@1 / @64 | SWOR pass@1 / @64 | long @64 (resample → SWOR) | tokens / theorem @64 |
| --- | --- | --- | --- | --- |
| 2 | 49.5 / 83.6 | 51.0 / 84.3 | 48.3 → 48.7 | 15.2k → 15.5k |
| 4 | 54.0 / 84.5 | 56.1 / 85.6 | 51.0 → 53.3 | 17.4k → 17.9k |
| 10 | 57.4 / 85.6 | 59.7 / 86.5 | 54.3 → 56.7 | 21.5k → 22.6k |
| 32 | 59.6 / 86.5 | 61.5 / 87.3 | 56.7 → 57.3 | 31.1k → 34.1k |

- **The gain is about +2 points at pass@1 and +1 at pass@64, for about 5% more tokens.**
  - It holds on every seed at cap 10: pass@64 +1.5, +0.3, +1.0.
  - SWOR at cap 10 matches plain resampling at cap 32 while using 73% of the tokens.
- **Repeats fall from 52% of rejections to 2.5%.** All the remaining repeats are segments that hit
  the 400-token cap without ending, which are not forbidden by design.
- **The rejection mix shifts.** Once the confident wrong line is banned, the next draw is more
  often ungrammatical: grammar failures go from 31% to 48% of rejections (seed 2). Length-capped
  runaways rise from about 1% to 2.3%. Those come from samples that have exhausted their plausible
  lines, and they carry most of the extra token cost.

## Follow-up B: k = 1,024 on the theorems nothing solved

The theorems no method solved at k = 64 (any cap, any N or B, best-first) were rerun at 16× the
samples. Both runs use `--names`, on each seed's own set.

| seed | stubborn theorems (> 10 lines) | plain k = 1,024 | resample (cap 10, `guided_kv`) k = 1,024 |
| --- | --- | --- | --- |
| 0 | 137 (47) | 1 | 12 |
| 1 | 143 (57) | 3 | 17 |
| 2 | 147 (60) | 3 | 11 |

- About 90% stay unsolved at 1,024 resampled attempts, so their success probability is below
  roughly 0.3% (the 95% upper bound for 0 of 1,024).
- The ones that fall are rare hits: 1-8 successes of 1,024 (one theorem had 15). Most are 9-10
  lines, not the longest.
- Plain sampling barely touches the set (7 theorems across the three seeds).
- **Interpretation, not tested:** for these ~140 theorems per seed, this policy puts almost no
  probability on any proof. More samples or a better search over the same policy will move them
  slowly at best. Changing the weights would be the lever, but test-time training on evaluation
  theorems is ruled out (Robbie, 2026-09-29), so that belongs in the training pipeline. A value
  function could still make the solvable part cheaper (round 2 below: it did not).

Cost of both follow-ups: about $2.20 (six pods, 33-58 min each). Total for the experiment: about
$7.90.

## Round 2: a learned value head (dev only, seed 0)

Plan: [value_head_plan.md](../../code/experiments/current/test_time_scaling/value_head_plan.md).
**Holdout rule (Robbie, 2026-09-29):** holdout is only scored for final configurations. Nothing
here touched it, and since no configuration beat round 1 on dev, none was scored.

### The head

- **Data:** c005 seed 0 resample rollouts (cap 10, T 0.8), 8 per theorem, on the 2,995
  `rl_targets` theorems c005's RL did not train on (0 overlap with dev or holdout).
  - 23,960 rollouts, 71% successful; 210k labelled prefixes.
- **Model:** an MLP (384 → 256 → 1) on the frozen trunk's final hidden state at the prefix's last
  token; BCE loss; 3,000 steps.
- **Validation** (332 held-out `rl_targets` theorems):
  - AUC 0.857 overall.
  - BCE 0.41 vs 0.58 for a constant prediction.
  - **Only 0.689 within theorem and depth**, i.e. among prefixes of the same theorem at the same
    depth, which is what search uses to choose between them. Most of the overall AUC is telling
    easy theorems from hard ones.
  - Roughly monotone calibration, under-confident at the low end (predicted 0.1-0.2, observed
    0.37).

### Search with it

Solves out of the 300-theorem dev subset, with sampled tokens per theorem. The value adds
forward passes, counted separately from sampled tokens.

| method | round 1 (log-prob / 0-1 weights) | with V |
| --- | --- | --- |
| SMC, N = 4 | 206 @ 897 | 202 @ 881 |
| SMC, N = 16 | 237 @ 4,243 | 234 @ 4,070 |
| SMC, N = 64 | 251 @ 17,646 | 251 @ 16,202 |
| beam, B = 4, E = 4 | 213 @ 3,481 | 221 @ 2,556 |
| beam, B = 16, E = 4 | 226 @ 8,540 | 223 @ 8,259 |
| best-first, E = 8, 16k budget | 217 @ 7,320 (α 0.5) | 220 @ 6,396 |
| *resample, cap 10 (reference)* | *237 @ 2,815 (k 8); 244 @ 5,630 (k 16); 255 @ 22,520 (k 64)* | |

- **No gain from V.**
  - Every with-V row is within ±8 theorems of its round-1 counterpart, the same spread as
    seed-to-seed noise on 300 theorems. Beam B = 4 is the only positive and B = 16 goes the other
    way.
  - V saves 13-27% of sampled tokens in beam and best-first, where log-prob ranking was
    especially poor, but neither catches resampling.
  - V is not free. Scoring every child re-encodes its prefix, which adds 5-6× the sampled tokens
    in prefill: 4-6× the sampled tokens, e.g. 99.6k per theorem for SMC N = 64 against 16.2k
    sampled.
- **Interpretation, not tested.** A within-theorem ranking of 0.69 is too weak to beat simply
  sampling more. The rollouts suggest outcomes depend mostly on the theorem, not on which valid
  prefix the model is in. That matches follow-up B: the stubborn theorems fail at every prefix,
  not at a fork a value could steer around.
- **Not tried:**
  - several rollouts per prefix to cut label noise;
  - fine-tuning the trunk instead of a frozen-feature head;
  - rollouts on dev-difficulty theorems (`rl_targets` succeed at 71% per rollout vs about 57%
    pass@1 on dev);
  - MCTS with the value as a leaf estimate.

Cost: one pod, 40 min, about $0.33.

## Limitations and next step

- **Seeds are 3 checkpoints of one recipe.** The seed-to-seed spread is small (pass@64 plain
  80.5-81.1, resample 85.5-85.7), but these results say nothing about other pretraining choices.
- **Search methods were tuned lightly** (α and E on 300 dev theorems, seed 0).
  - Beam with a length-normalised score and best-first with state deduplication (canonical ND
    states rather than exact line sequences) were not tried.
  - Neither is expected to fix the ranking problem without a value.
- **Engine wall-clock overstates search cost.** No KV cache is kept across tree nodes. Sampled
  tokens is the fair axis there.
- **Next, cheapest first:**
  1. ~~Resample without replacement~~: done (follow-up A), +1 point at pass@64.
  2. Temperature sweep for resample. The RL sampler used T 1.0; the eval uses 0.8.
  3. Use `guided_kv` as the harness eval sampler, a protocol decision for Robbie. It would change
     how every pretraining ablation scores.
  4. ~~Test-time expert iteration on the ~142 unsolved theorems~~: ruled out; no training on
     evaluation theorems.
  5. ~~A learned value for SMC, beam and best-first~~: done (round 2), no gain on dev.
