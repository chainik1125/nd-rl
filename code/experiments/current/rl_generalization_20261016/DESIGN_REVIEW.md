---
written_on: 2026-10-16
written_by: agent:claude
---

# Review: will the mini-RL knob runs teach us about the learning dynamics of generalization?

Read-only review of the planned mini runs: from Stage-1 `S1g3`, 1,000 L2 problems then 1,000 L3 problems,
10 iterations each, one knob per run in both stages, final-checkpoint reads, plus reads of one full-recipe run at
it 0/10/20/30/39. Nothing was launched or edited. Some numbers below are new: I computed them on CPU from existing
eval rows (`runs/eval/v2/*/rows_plain_B64.jsonl`) and training logs (`runs/grpo_v2/*/metrics.jsonl`). The scripts
are in [`analysis/`](analysis/): `traj.py`, `l3_end_stats.py`, `hazard.py`, `slope.py`, `slope_paired.py`,
`power.py` and `set_correlations.py`. I label each claim as **observed** (in the data), **interpretation** or
**hypothesis**.

Provenance: the first copy (written 2026-10-16 in a session scratchpad) was lost when /tmp was wiped. This copy
was rewritten from the session transcript on 2026-10-18 with the same substance. The scripts were reconstructed
the same way and re-run on CPU; they reproduce every number below.

## Verdict

As designed, the runs will mostly measure **one noisy endpoint per knob** in a training regime unlike the full
recipe's. That is not enough to separate generalization from competence, or to see dynamics:

1. **No control for competence.** Every knob is read only at its final checkpoint. With about 12 endpoints there
   is no ID→OOD trend to measure residuals against, and the L3 in-distribution (ID) set is close to its ceiling
   wherever the recipe resembles the full one.
2. **Mostly endpoints, not dynamics.** The one time-resolved element is a single full run at 5 checkpoints.
3. **Underpowered and unvalidated.** One seed, one noise draw, and no check that the mini setup reproduces known
   effects. The existing data suggest that most run-to-run variance in long-proof results enters *before* L3.
   Applying knobs to both stages from the base therefore adds the largest noise source to every comparison.
4. **Two knobs are not interpretable as designed.** `updates` 1/8 is mostly an lr/step-count knob, which Leon
   excluded. Temperature is confounded by the train/eval temperature mismatch.

All of this can be fixed at moderate cost. The cheapest, highest-value step needs **no new training**: twelve
existing full-scale L3 runs already have checkpoints every 5 iterations, but only their it39 has ever been read.

## What the existing data already show (relevant to the design)

**E1. "Generalization" is at least two separate axes (observed; `analysis/set_correlations.py`).** I correlated
pass@1 across the 31 (arm, seed) models in `thinking_traces_20261005/matched_eval/v2_results_ynez.md` § 2 (plain
reads, GRPO and EI arms).

| pair of sets | r (31 models) | r (22 GRPO models) |
|---|---|---|
| 13–20 vs 21–30 | 0.92 | 0.94 |
| 13–20 vs 16–20 | 0.98 | 0.96 |
| textbook72 vs Charles149 | **0.82** | **0.81** |
| 13–20 vs Charles149 | 0.20 | 0.46 |
| 21–30 vs Charles149 | −0.03 | 0.24 |
| 21–30 vs textbook72 | −0.06 | 0.02 |

- The long generated sets move together, and the two human-written sets move together.
- Across models the two families are nearly independent. The main algorithmic contrast even points in opposite
  directions on them (`length_split_ynez.md` § 1b, § 2b): GRPO − EI is +6.6 on the human-written problems and −3.1
  on the generated ones, at matched length.
- Seeds show the same split. The merged recipe's seeds 0/1/2 read 34.4/30.1/26.7 on 13–20 but 70.0/72.0/69.3 on
  Charles149 (§ 2).

**E2. On the long sets, knobs split into "level" and "slope" effects (observed; exploratory; `analysis/slope.py`,
`analysis/slope_paired.py`).** I fitted the logit of the mean per-line pass rate against the reference length
(13–30 lines, t1320 and t2130 pooled). The level is the logit at 13 lines; the slope is the change per extra line.
Intervals are paired bootstraps over problems (400 draws).

| comparison (same L2 start unless noted) | Δ level (logit) | Δ slope (per line) |
|---|---|---|
| g64 − merged s0 | **−0.27 [−0.38, −0.17]** | −0.005 [−0.024, +0.012] |
| KL 0.1 − merged s0 | **−0.15 [−0.22, −0.06]** | +0.002 [−0.016, +0.016] |
| T 1.2 − merged s0 | −0.06 [−0.16, +0.03] | −0.013 [−0.033, +0.005] |
| KL 0 − merged, seeds 0 / 1 / 2 | +0.00 / +0.07 / +0.05 | **+0.017 / +0.014 / +0.018** (each lower bound −0.001 to −0.003) |
| rerun (krr) − merged s0 | −0.02 | −0.009 [−0.022, +0.005] |
| merged s1 − s0 (different Stage-1 and L2) | −0.07 [−0.18, +0.05] | **−0.049 [−0.072, −0.031]** |
| merged s2 − s0 | −0.28 [−0.41, −0.15] | **−0.045 [−0.067, −0.023]** |
| EI + verifier s0 − merged s0 | −0.10 [−0.22, +0.03] | −0.021 [−0.045, −0.002] |

- The baseline slope is about −0.23 per line: the odds of a solve fall about 21% per extra line.
- Interpretation:
  - g64 and KL 0.1 lower the level and leave the slope alone: competence-like effects.
  - KL 0 leaves the level and flattens the slope, in all three seeds: a length-specific effect.
- Hypothesis: the KL anchor acts on teacher-forced *long-state* steps (`canary.anchor_kl`, states with ≥ 10
  hypothesis lines), so removing it frees behaviour on long contexts specifically.
- The seed-to-seed slope differences are about 3× the KL 0 effect.

**E3. Most of the variance enters before L3 (observed).**
- An L3 rerun from the same L2 checkpoint (krr) moves 13–20 pass@1 by −1.2. The seed-2 control rerun moved it
  by +1.3 (25.8 vs 24.5).
- Seeds, which differ in Stage-1 and L2, span 26.7–34.4 on 13–20 and 4.0–8.9 on 21–30.
- Paired arm differences change sign across seeds even when each seed's interval excludes 0. Merged − stop on
  13–20 is +3.9, −2.6 and +2.2 (§ 3).
- Zhang et al. 2018 report the same pattern in deep RL: identical training gives very different test performance.

**E4. ID L3 is near its ceiling in the full recipe (observed, `metrics.jsonl`; `analysis/traj.py`).**
- Training pass@1 at L3 iteration 1 is already 0.81 at T 1.0 (13-line groups 0.80). It ends at 0.88–0.93.
- 45–50% of groups are all-solved late in L3.
- The 13-line problems of t1320 read about 47–51% at it39 (`analysis/hazard.py`), so the gap between the pool
  family and the composite family *at the same length* (~30–40 points) is larger than the whole decline from 13 to
  20 lines within t1320 (47 → 21%).
- Interpretation, to confirm with the planned L3 test_1k read: what the long sets measure is mostly **structural
  composition** (3+ premises, 4 atoms, more than 8 connectives, padding; 0% of the pools, 79–100% of the sets,
  `source_split_ynez.md` feature table), with length on top of it.
- L2 from the base runs 0.23 → 0.63 (it10) → 0.86 (it39). So a 10-iteration mini L2 ends in the steep part of
  learning.

**E5. Think-token entropy rises through L3 in every run (observed; mechanism is a hypothesis;
`analysis/l3_end_stats.py`, `analysis/traj.py --late`).** Mean of the last 5 iterations of `entropy_think`, at the
training temperature:

| run | think entropy | note |
|---|---|---|
| merged s0 / s1 / s2 | 0.167 / 0.208 / 0.288 | |
| KL 0 s0 / s1 / s2 | 0.136 / 0.196 / 0.194 | below merged at every seed |
| krr | 0.166 | |
| KL 0.1 | 0.188 | |
| x66 | 0.198 | |
| ch4 | 0.221 | |
| g64 | 0.327 | |
| T 1.2 | 0.239 | measured at T 1.2 |
| p3 | 0.096 | 3 rollbacks: lr × 0.125 |
| i78 | 0.652 | by it 74–78 |

- Every run starts at 0.078. In i78, training pass@1 also falls, from 0.927 (it65) to 0.876 (it78).
- The two worst knobs (g64, T 1.2) also have the fewest no-signal groups (6% and 24% skipped vs about 45%), so
  the most failure-driven gradient.
- Hypothesis: negative-advantage mass flattens the think distribution, and that hurts long proofs, where per-step
  errors compound. This is consistent with the observation that negative-sample updates redistribute probability
  mass (Zhu et al. 2025). Untested.

**E6. Where proofs die differs between methods (observed; `analysis/hazard.py`).** Hazard of the first invalid
step, per 1,000 steps reached, on t1320:
- GRPO arms: about 7 at steps 1–5, rising to 22–28 by steps 11–20, then flat.
- EI + verifier: 11 early, 13–18 in the middle, 31–34 after step 30.
- KL 0 s0 is lower than merged s0 at steps 11–20 (17.7/21.7 vs 22.0/27.8). g64 is higher at every step.
- So pass@1 averages over different failure mechanisms. The per-attempt `fail_step`/`status` fields already in the
  rows support this analysis at no GPU cost.

**E7. Checkpoints that exist and reads that do not.**
- Every-5-iteration L3 checkpoints exist for krr, KL 0 s0–s2, KL 0.1, g64, T 1.2, ch4, x66, p3, merged s1 and i78
  (it 0–78) (`runs/grpo_v2/*_L3/ckpts`).
- L2 runs kept only it39.
- Never read on the v2 sets: any intermediate checkpoint, any L2 checkpoint (= L3 it0), the S1g3 base, and either
  ID test_1k.

## Findings by question

### Q1. Will the runs separate generalization from competence?

No, not as planned.
- A knob that only changes competence moves ID and OOD together. Telling that apart from a generalization effect
  needs an ID→OOD reference trend, with each knob judged by its residual at matched ID. This is the "accuracy on
  the line" / effective-robustness logic (Miller et al. 2021; Taori et al. 2020).
- A trend needs many (run, checkpoint) points that span ID. The plan gives about 12 endpoints, one per knob, which
  is too few to estimate a line and its residuals.
- Wherever the mini run resembles the full recipe, ID is close to its ceiling (E4).

What it needs:
1. **Points along trajectories.** A fixed problem panel read at 4–6 checkpoints per run (Q2), so that each run
   traces its own ID→OOD path. Residuals are then measured at matched ID. Fit the reference trend on the baseline
   replicates' trajectories, in logit space.
2. **An ID measure with headroom:**
   - ID pass@1 at B0, which is lower: 0.75–0.86 in L3 training;
   - a held-out ID subset the L2 checkpoint solves under 50%;
   - teacher-forced per-step NLL on held-out ID reference proofs (forward passes only; the canary code already
     does this for long states).
3. **The level/slope decomposition** (E2) for length, the per-step hazard profile (E6), and pass@1 vs pass@k
   (coverage vs sharpening; Yue et al. 2025, and v1's own "pass@1 up, pass@5 flat" in `RL_RESULTS.md` § 1).
   For example, KL 0 raised 21–30 pass@5 (+3.4) more than pass@1 (+1.2), which is a coverage gain on the longest
   set.
4. **Separate structure from length.** Report the ladder L3 test_1k (ID length, ID structure) → t1320 13–15 (ID
   length, OOD structure) → t1320 16–20 → t2130. A pool-family 16–20-line test set would isolate length without
   composition. It needs generator time (CPU) and Leon's call, since the pool generator would make it.
5. **Keep the two OOD families separate (E1).** A single "OOD" score averages two nearly independent axes.

The planned read set covers most of the axes: L2/L3 test_1k, t1320 with 16–20 split out, t2130, Charles. Missing:
the base and L2 reads, the train subset (memorization), an ID measure with headroom, B0/B256 reads for knobs that
act through thinking, and an eval-temperature grid for the temperature knobs.

### Q2. Dynamics or endpoints?

Endpoints. The one time-resolved element is a single run, one seed, 5 checkpoints, L3 only (its it0 is the L2
end). That run (merged + KL 0 s0) also had a canary rollback at it29, so its it30 is a rolled-back state. Two
time-resolved schedules are recommended.

**(a) Existing full-scale checkpoints (reads only).**
- Runs: krr (the baseline trajectory), KL 0 s0/s1/s2, KL 0.1, g64, T 1.2, merged s1, and i78 (10–70).
- Checkpoints: it 0, 10, 20, 30, plus ID reads at it39.
- This gives dose-response *in time* (KL 0/0.02/0.1), a strong negative control (g64), and prolonged training,
  in the regime the recipe actually uses. It is also the reference for judging whether the mini runs transfer (Q4).

**(b) Mini runs.**
- Save every 2 iterations (`--save_rolling 2 --keep_rolling 100`; roughly 15–20 GB in all).
- Read a fixed dynamics panel at L2 it 3 and 6 and L3 it 3 and 6, plus the full read at the end.

**Dynamics panel** (paired across all checkpoints and runs):

| set | k | time per read |
|---|---|---|
| t1320, all 986 | 4 | ~6 min |
| t2130, all 1,000 | 4 | ~7 min |
| L3 test_1k | 4 | ~2–3 min (estimate) |
| L2 test_1k | 2 | ~1 min (estimate) |
| Charles149 | 16 | ~1.5 min |

- That is about 18 minutes per checkpoint at it39 strength, less for weaker checkpoints, whose failed attempts end
  early.
- The times are scaled from the kkl0 s0 logs on serrano: t1320 k16 1,426 s, t2130 k8 804 s, Charles173 k64 307 s.
- Use all problems at low k, not a subset at high k. Per-problem differences dominate the variance: t1320 paired
  SE is 0.86 at k4 on all 986 but 1.5 at k4 on 300 (`analysis/power.py`).
- For per-problem flip analyses, keep a fixed subset at k16 at 2–3 checkpoints only.

Costs:

| schedule | reads | GPU-h | wall time on 5 GPUs |
|---|---|---|---|
| (a) existing checkpoints | 31 panel reads + 3 full reads of L2 it39 (seeds 0–2) + ~12 ID-only reads at it39 | ~13 | ~2.6 h |
| (b) mini runs, per run | 4 panel reads + 1 full read | ~1.75 | — |

### Q3. Statistical power

**Read noise** (paired SE between two models, problems and attempts bootstrapped, krr vs merged s0;
`analysis/power.py`):

| set | k | paired SE (points) |
|---|---|---|
| t1320 | 16 / 8 / 4 | 0.55 / 0.62 / 0.86 |
| 16–20 only | 8 | 0.79 |
| t2130 | 8 / 4 | 0.41 / 0.53 |
| Charles173 | 64 / 16 | 0.91 / 1.02 |

**Training noise:**
- Two draws, both with a shared L2 (−1.2 and +1.3 on 13–20), imply SD(Δ) ≈ 1.3 points for a single-seed
  knob − baseline comparison at k8. Two draws cannot pin down a variance: one noise rerun says almost nothing about
  the noise band.
- The single-seed minimum detectable effect (80% power, α 0.05) is then about **3.7 points** on 13–20 pass@1.
  With *n* seeds per arm it is about 3.7/√n: about 2.6 at n = 2 and 2.1 at n = 3.
- At full scale only g64 (−5.5) clears the single-seed bar. KL 0 (+1.7) needed 3 seeds, and Charles149 resolves
  nothing below about 3 points.

**In the mini design the noise is probably larger.** Each run re-samples L2, and E3 says L2/Stage-1 is the
dominant variance source at full scale (seeds differ by 4–8 points on 13–20 and by 3× the KL 0 effect on the
slope). The size of this variance in the mini setting is unknown and has to be measured first.

Changes:
- Replace the single noise rerun with **3–4 baseline replicates**, which also become the reference trajectories.
- Give each knob at least 2 seeds. To keep the cost flat, test fewer levels: the extremes first (T 0.7 and 1.3;
  length −0.1 and +0.2 or more), and fill in the middle only if the extremes differ beyond noise.
- For knobs meant to act in L3, **fork from a shared mini-L2 checkpoint** (one L2 per seed, then the L3 knobs).
  This removes L2 variance from the comparison, as the full-scale pilots did.
- Analyse per problem, paired and hierarchical (seed → problem), as `code.thinking.stats` already does. Report
  pass@1 and pass@5, and treat 21–30 pass@5 as a primary long-set metric: it has the smallest SE relative to the
  effects.
- Statistically: few seeds, interval estimates, and no single-seed "†" calls (Agarwal et al. 2021).

### Q4. External validity of 2,000-problem, 20-iteration runs from the base

How the mini setup differs from the full recipe:

| | full recipe | mini |
|---|---|---|
| pool | 5,000 L2, 5,000 L3 + 1,796 3-premise rewrites | 1,000 + 1,000; the 3-premise extra pool for L3 is not specified (subset or full?) |
| iterations | 39 + 39 | 10 + 10 |
| start of L3 | training pass 0.81, near-saturated sharpening regime | about 0.6 (a guess from the full L2 at it10), learning regime |
| warmup share (10 AdamW steps) | 6% of a stage | 25% |
| WSD cooldown | 8 iterations | 2 |
| L3 think-entropy rise and drift | mostly after it 20 (E5) | 10 iterations; probably not reached |

- The draws per (target, budget) pair are similar: 0.67 per pair in the full runs, 0.85 in the mini runs.
- **Interpretation.** Knobs that act through late drift (KL anchor, entropy growth, prolonged training) will tend
  to look null in mini runs. Knobs that act on early learning (temperature, group shape) may show effects that do
  not carry over to the saturated regime. Both regimes are worth studying, but mini results should not pick knobs
  for the full recipe without a transfer check.

**Controls before trusting any new knob** (one wave of 5 GPUs, about 2.75 h):
1. Two more baseline seeds (with seed 0 and the planned rerun, 3–4 replicates).
2. **Positive control A:** KL 0.1 vs 0 *in L3 only*, as in the full-scale pilots. Expected at full scale: a lower
   level, about −4 points on 13–20 (−2.7 for 0.1 vs 0.02, +1.6 for 0 vs 0.02), slope unchanged.
3. **Positive control B:** g64 (64 × 64). Expected: about −5 points, a level shift (Δ logit about −0.27).
4. A temperature control is already implied: T 1.15/1.3 bracket the known T 1.2 result (−2.2).
5. The pool size is a free bonus check: the mini run's train-subset vs test_1k gap.

**Go/no-go gate (pre-register it):** mini knobs proceed only if
- the baseline replicate SD on 13–20 pass@1 is ≤ about 1.5 points, and
- both controls reproduce the sign and at least half the full-scale magnitude, beyond 2 replicate SDs.

Otherwise run the knobs as L3 forks from the full L2 it39 checkpoints: about 1.9 h of training each, a validated
setup.

### Q5. Per-knob confounds

**Temperature 0.7 / 0.85 / 1.15 / 1.3.** Interpretable only with an eval-temperature control; this is the most
serious confound.

What the knob changes besides exploration:
- **Train/eval mismatch.** The model is trained so that π_T is good, but it is read at T 0.8. A model trained at
  0.7 is read hotter than it was trained; one trained at 1.3 is read colder.
  - Read temperature has large effects on long sets in this repo. The stop arm s2 read at T 1.0 instead of 0.8:
    13–20 20.3 vs 24.5 (§ 1 and § 8).
  - So the T knob's long-set effect is confounded with how per-step errors compound at the read temperature.
- **Gradient and clip.** The gradient is scaled by 1/T. Adam largely normalizes that away, but the PPO ratio is
  computed at T: the same logit change gives a smaller ratio at high T, so the clip binds later.
- **Curriculum and signal.** Pass rates fall at high T, so the p(1−p) curriculum shifts toward easier pairs and
  the signal fraction rises (T 1.2: 24% of groups skipped vs about 45%).
- **Alarms and rollbacks.** The think-ban mass and entropy are measured at T, so the alarms, rollbacks and
  `lr_scale` are affected.

Fix:
- Read every T run on t1320 (k4) and L3 test (k4) at T_eval 0.6, 0.8 and 1.0: about 30 min per run.
- Report both the fixed-0.8 number and the best-T_eval number.
- Log rollbacks as an outcome, and flag runs with lr_scale < 1.
- The manipulation check is entropy at T_eval on OOD states.

**Length reward −0.05 / +0.05 / +0.15 per 1k tokens on successes.** Interpretable as token-budget pressure, but
probably too weak, and the code fix needs checking.

- **Magnitude.** L3 attempts are about 410–460 tokens. A ±150-token within-group spread gives about ±0.008
  (at 0.05) to ±0.02 (at 0.15) of reward. The unlikeliness deduction already in the recipe averages 0.14
  (`row['unlikely'].pen_mean`). v1 needed 0.2 per 1k over 78 iterations for a 40% token cut (`RL_RESULTS.md` § 3).
  Over 20 iterations, ±0.05 is likely null by construction.
- **What it penalizes.** The term covers all tokens, think and action, across all steps. It pushes toward shorter
  proofs as well as shorter thinking, and at B0 it acts on actions only.
- **The fix changes the batch.** Letting all-solved groups (about 45% in L3) contribute a length-only gradient
  changes which groups train and the loss normalization. Verify that `--len_penalty 0` stays bit-identical (the
  digests in `test_grpo_pilots.py`), and log the gradient share of all-solved groups.
- **The bonus (−0.05)** rewards padding: watch `forced_by_budget` and `capped_frac`.

Fix:
- Use −0.1, +0.2 and +0.5 if you want an effect.
- The manipulation check is think tokens per block by budget and proof steps relative to the reference.
- Read B0 and B256 as well as B64 (t1320 k4), because the knob may act through the effort-conditional gap.

**Group shape 512 × 8 vs 256 × 16.** Interpretable as breadth vs depth at fixed samples. It changes:
- the distinct pairs per stage: 5,120 draws over 3,000 pairs, so about 1.7 per pair vs 0.85, which means more
  repetition of the 1,000-problem subset (this matters for memorization);
- the probability of a mixed group: at p 0.05, 34% vs 56%; at p 0.9, 57% vs 81%;
- baseline noise (r − mean over 8);
- the granularity of the unlikeliness rank.

Fix:
- Add g64 (64 × 64) for a 3-level dose response, which is also positive control B.
- Log distinct pairs drawn, signal share, and gradient share by pass band (`--monitor`).

**Updates per sample batch 1 / 8 (vs 4).** Not interpretable as an on- vs off-policy knob.
- `--updates` splits one batch into that many minibatches with one AdamW step each, so each sample is used once
  whatever the setting. What the knob changes:
  - the number of optimizer steps per stage: 10 / 40 / 80;
  - the minibatch size, hence the gradient noise;
  - the staleness of later minibatches;
  - the warmup. The ramp is 10 AdamW steps (`--warmup 10`, `warmup_iters 0`, `args.json`). At updates 1 the whole
    10-iteration stage is inside warmup; at updates 8 the warmup lasts 1.25 iterations.
- The lr-weighted step totals are therefore roughly **0.15 : 1 : 2.1** for updates 1 : 4 : 8. With Adam, that
  approximates the update magnitude. It is effectively the lr/optimizer knob Leon excluded.
- At updates 1 the clip is also inactive (ratio = 1).

Fix: hold the steps and samples fixed and vary only staleness.
- On-policy arm: 64 prompts × 16 per sample batch, 1 update per batch, 40 iterations per stage.
- Off-policy arm: the current 256 × 16 with 4 updates.
- Same 40 AdamW steps, same samples, same minibatch size.
- Set `--warmup_iters` so the warmup is in iterations for both arms.
- Otherwise drop the knob.

**Common to all knobs.**
- Rollbacks (lr halving) are a hidden lr confound. In the full pilots: KL 0 s0 1, KL 0 s1 2, p3 3 (lr × 0.125),
  ch4 1, merged s2 1. In a 10-iteration stage one rollback is a large perturbation. Report `lr_scale` per run, and
  either rerun or flag affected runs.
- Each knob needs a pre-declared **manipulation check**. A null outcome without one is uninformative.
- Each knob needs a stated hypothesis about *how* it should change generalization.

### Q6. Is Charles149 a valid proxy for textbook72?

Directionally yes; for knob-sized effects no; and it is not a proxy for length generalization.

**For:**
- Across 31 models r = 0.82 with textbook72, and 0.81 among GRPO models only (E1).
- Both sets show the same GRPO − EI human-written advantage (`length_split_ynez.md` § 1a, § 2a).
- Both are much closer to the pools than the generated sets:

  | measure | textbook72 | Charles149 | generated sets |
  |---|---|---|---|
  | subformula shapes in the pool | 0.66 | 0.69 | 0.32–0.33 |
  | proof skeleton in the pool | 45% | 31% | 0–7% |

**Differences** (`source_split_ynez.md` feature table): Charles149 is somewhat more compositional and longer.

| feature | Charles149 | textbook72 |
|---|---|---|
| 3+ premises | 19% | 6% |
| 4 atoms | 27% | 11% |
| A∧B goals | 36% | 19% |
| ∨E with a trivial branch | 53% | 25% |
| problems of 13+ lines | 49 of 149 | — |
| problems of 15+ lines | — | 18 of 72 |

**Sensitivity:**
- The paired SE is about 1.0 point at k64, with about 50 of the 149 problems in an informative difficulty band
  (§ 1d of `length_split_ynez.md`: 16 of the 221 human-written problems at 0%, 32 at 100%, 97 above 80%).
- Raising k past 64 buys almost nothing; the problem count is the limit.
- No GRPO knob moved Charles149 by more than ±1 apart from g64 (−2.2), whereas GRPO vs EI moved it by about 7.

Recommendations:
- Keep it as the tuning-safe proxy for the human-written axis.
- Report its 13–15 and ≥16 bins, to connect it to length.
- If source generalization becomes a target, build a larger human-written dev set from a separate, sealed source.
- Keep textbook72 sealed as planned.

### Q7. What is missing for a science of generalization dynamics

Ordered by value per cost; all are cheap relative to training.

1. **Exposure-conditioned memorization.**
   - Read the fixed 1,000-problem training subset alongside test_1k at the final checkpoint (k8, about 5 min).
   - Break the gap down by the number of times each pair was drawn (the curriculum counts are in `train_state`).
   - The mini design's fixed subset makes this possible: it is the RL analogue of the train/test level split in
     Cobbe et al. 2019.
2. **Distance from the start model, per checkpoint.**
   - KL(π_base‖π_θ) and the reverse, teacher-forced on held-out ID, t1320 and Charles reference steps, split into
     think and action and by state size.
   - Parameter distance, overall and per layer.
   - Then test whether OOD change tracks KL drift *on the training distribution* (RL's Razor: Shenfeld et al. 2025)
     or drift on long states (the E2 hypothesis). Forward passes only.
3. **Entropy at the read temperature on OOD states**, think and action, by step index. The logged training
   entropy is at the training temperature on training states (E5).
4. **Per-step hazard by step index and state size** (E6), from the existing `fail_step` and `status` fields, so
   "fails late in big contexts" can be told apart from "fails uniformly".
5. **Per-problem flips across checkpoints.** Learned and forgotten events, churn vs net gain (Toneva et al. 2019),
   and which `source_split` features predict flips.
6. **Solution diversity.** Distinct Lean-accepted proofs per problem, and pass@k curves (k 1–8 generated, 1–64
   Charles) over time, to see whether gains are sharpening or coverage.
7. **Thinking.** Think length by proof length and budget, and B0/B64/B256 gaps over time (`ADAPTIVITY.md` did this
   across 3 checkpoints in v1). Dead-end and backtrack counts parsed with `trace_dsl.py`.
8. **Where the gradient comes from.** `--monitor` already logs gradient share by target length and pass band; put
   it next to the read trajectories.
9. **Causal data levers.** Leon's question is what *causes* generalization; the most direct levers here are data,
   not optimizer knobs:
   - **pool size** at fixed iterations (250 / 1,000 / 4,000), the classic generalization lever (Cobbe et al. 2019);
   - pool diversity (the 3-premise share, partly tested);
   - training-problem length (the pending 16–20-line generator pool and mid-proof starts, both waiting on Leon).

## Prioritized changes

| # | change | cost | what it buys |
|---|---|---|---|
| 0 | Formalize the existing-data analyses (level/slope, hazard, set correlations, per-problem flips) in `v2_results.py`-style scripts | CPU only, about half a day of agent work | Shows now which known knobs are "level" and which "slope"; settles the analysis plan before new runs |
| 1 | Read the existing L3 checkpoints on the dynamics panel (Q2a), plus L2 it39 for seeds 0–2 and the S1g3_s0 base on the full set | ~13 GPU-h ≈ 2.6 h wall; no training | Full-scale dynamics for 9 runs; how much OOD gain comes from L2 vs L3; the reference for mini-run transfer |
| 2 | Fix the knobs: `updates` → fixed-step staleness contrast (or drop); T runs with a T_eval grid and rollback reporting; length reward at −0.1 / +0.2 / +0.5 with bit-identity check and B0/B256 reads; g64 added to group shape; specify the 3-premise subset; `--save_rolling 2` | ~0 GPU-h (+~2 GPU-h of extra reads) | Makes every knob interpretable and gives each a manipulation check |
| 3 | Validate the mini setup: baseline × 3–4 seeds, KL 0.1 in L3, g64; panel reads; go/no-go gate | ~14 GPU-h ≈ 2.75 h wall (one wave) | A measured noise band, and evidence that the mini runs reproduce known effects |
| 4 | Knobs at 2 seeds, extremes first (T 0.7/1.3, length −0.1/+0.5, 512 × 8, staleness), with panel reads at 4 checkpoints each | ~33 GPU-h ≈ 8 h wall (3 waves) | Effects detectable at about 2.5 points; about 100 (run, checkpoint) points for the ID→OOD trend and residuals |
| 5 | L3-only forks from a shared mini-L2 for L3-acting knobs | Same per-run cost as #4; replaces part of it | Removes the dominant (L2) variance from L3 comparisons |
| 6 | Per-checkpoint KL-from-base, eval-T entropy and memorization reads | ~5 min forward passes per checkpoint, plus ~5 min train-subset reads per final | Measures *what changed in the policy* next to the outcomes |
| 7 | New causal levers (Leon's call): pool size 250/1,000/4,000; pool-family 16–20-line test set | 3 mini runs × 2 seeds ≈ 16 GPU-h; the test set is CPU generator time | Separates length from composition, and tests the textbook "data breadth causes generalization" hypothesis |

Totals:
- Rows 1–4: about 60 GPU-h, roughly 13–15 h wall on 5 GPUs, against about 21–25 GPU-h for the plan as written.
- Cheaper path: rows 0–3 first (about 27 GPU-h, under 6 h wall), then decide on row 4 from the measured noise
  band. Row 1 alone may answer much of Q1/Q2 for the existing knobs.
- All of rows 1, 3, 4 and 7 are new runs or reads, so each needs Leon's yes under the consult rule.

## Suggested analysis plan (pre-register before the runs)

1. **Unit:** (run, checkpoint). **Metrics:**
   - ID: L2 and L3 test pass@1 at B64, plus B0; teacher-forced NLL.
   - Generated OOD: level and slope (logit, 13–30 lines), 16–20 and 21–30 pass@1/pass@5.
   - Human-written OOD: Charles149 pass@1/pass@5.
   - Policy: entropy at T_eval, KL from base, think tokens.
2. **Competence trend:** fit logit(OOD) on logit(ID) using the baseline replicates' trajectories. A knob's
   generalization effect is its residual at matched ID, averaged over its checkpoints, with a hierarchical
   bootstrap over seeds and problems. Report the competence effect (movement along the line) separately.
3. **Length:** report level and slope changes separately. Only slope changes count as length generalization.
4. **Source:** the Charles149 residual, reported separately from the long-set residual; do not average them (E1).
5. **Mechanism checks,** each pre-declared as a prediction: hazard profile, entropy and KL trajectories,
   memorization gap vs exposure.
6. **Multiple comparisons:** about 6 knobs × 4 outcomes. Treat single-seed or single-outcome hits as screening
   only.

## Literature (titles, authors, years and arXiv ids checked by web search; claims kept modest)

- Miller, Taori, et al. 2021, "Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and
  In-Distribution Generalization", arXiv 2107.04649. OOD accuracy is often a linear function of ID accuracy (in
  probit space) across models and training settings, which motivates measuring residuals from that trend.
- Taori et al. 2020, "Measuring Robustness to Natural Distribution Shifts in Image Classification", arXiv
  2007.00644. Robustness metrics should control for baseline accuracy ("effective robustness").
- Cobbe, Klimov, Hesse, Kim, Schulman 2019, "Quantifying Generalization in Reinforcement Learning", arXiv
  1812.02341. Train/test level splits; agents overfit even with many training levels; the number of training
  levels is a primary lever.
- Zhang, Vinyals, Munos, Bengio 2018, "A Study on Overfitting in Deep Reinforcement Learning", arXiv 1804.06893.
  Agents with identical training can differ a lot on test environments even at optimal training reward.
- Agarwal, Schwarzer, Castro, Courville, Bellemare 2021, "Deep Reinforcement Learning at the Edge of the
  Statistical Precipice", arXiv 2108.13264. Few-run evaluation needs interval estimates and aggregate metrics.
- Andrychowicz et al. 2020, "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study",
  arXiv 2006.05990. A large controlled study of on-policy design choices; a model for one-knob-at-a-time studies
  with many seeds.
- Chu, Zhai, et al. 2025, "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training",
  arXiv 2501.17161. Outcome-reward RL generalized to rule and visual variants where SFT did not.
- Yue et al. 2025, "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base
  Model?", arXiv 2504.13837. RLVR mainly improves sampling efficiency; base models can overtake at large k. This
  motivates tracking pass@1 vs pass@k.
- Shenfeld, Pari, Agrawal 2025, "RL's Razor: Why Online Reinforcement Learning Forgets Less", arXiv 2509.04259.
  Forgetting tracks the KL from the base model measured on the new task.
- Cui et al. 2025, "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models", arXiv
  2505.22617. An empirical entropy–performance relation, with entropy change tied to the covariance of probability
  and logit updates.
- Zhu, Xia, Wei, Chen, Chen, Meng 2025, "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning",
  arXiv 2506.01347. Positive-only reinforcement raises pass@1 but hurts high k; negative-sample updates
  redistribute mass to other candidates.
- He, Fried, Welleck 2025, "Rewarding the Unlikely", arXiv 2506.02355 (already used by the recipe).
- Toneva et al. 2019, "An Empirical Study of Example Forgetting during Deep Neural Network Learning", arXiv
  1812.05159. Per-example forgetting events as a dynamics measure.
- Anil et al. 2022, "Exploring Length Generalization in Large Language Models", arXiv 2207.04901. Naive
  fine-tuning length-generalizes poorly; failure-mode analysis.
- Huan et al. 2025, "Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM
  Reasoning", arXiv 2507.00432. RL-tuned models transferred better than SFT-tuned ones, with less
  representation and output drift.
- Mukherjee, Yuan, Hakkani-Tür, Peng 2025, "Reinforcement Learning Finetunes Small Subnetworks in Large Language
  Models", arXiv 2505.11711. RL updates are sparse, which is relevant if parameter distance is tracked.

None of these is in `references.bib`. Adding them would go through `code/tools/literature.py` and is not done here.

## Caveats on this review

- E2, E5 and E6 are new, exploratory analyses: one read per model, simple logit fits on per-line means, bootstraps
  over problems only. The KL 0 slope effect is consistent across 3 seeds, but no single seed resolves it.
- The noise estimates rest on two rerun draws. The noise in the mini setting is unknown; that is the main reason
  for the baseline replicates.
- The ID ceiling (E4) is inferred from training pass rates at T 1.0 on training problems. The planned L3 test_1k
  read at T 0.8 will settle it.
- Read times are from kkl0 s0 it39 logs on serrano. Earlier checkpoints may read faster; the ID-set times are
  estimates.
