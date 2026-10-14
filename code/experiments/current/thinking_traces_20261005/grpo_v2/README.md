---
written_on: 2026-10-09
written_by: agent:claude
---

# GRPO v2: length curriculum and GRPO with the verifier

Branch `leon/grpo-v2` (worktree `/scratch/cowifornia/nd-rl/think/nd-rl-v2`), built on `leon/thinking-auto`.
Launcher: `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/line.sh GPU SEED ARM`.

## What Leon asked for (2026-10-09, after the group meeting)

- **GRPO + verifier:** redraw invalid steps during GRPO sampling, and penalise the rejected steps.
- **Curriculum:** RL on 10–12-line problems, then 13–15. The unseen-length tests become 16–20 lines (the
  16–20 part of the 13–20 set) and 21–30.
- **Both models:** GRPO on the thinking model (Stage-1 S1g3) and on the no-think model (T0S).
- **Priorities:** GRPO vs EI in a fair setup, then what each change buys, then the model-chosen think budget.

## What the code does

All of it is opt-in; with default arguments the sampler's results, GRPO's metrics rows and the trained weights
hash identically to `leon/thinking-auto` (checked on a tiny model).

- **Guided rollouts** (`grpo.py --arm logical --max_rej 10`). GRPO samples with the arm that eval and EI already
  have (`sample.think_generate`):
  - a draw the environment rejects, or one `step_check` rejects, is redrawn from the same state;
  - redraws are without replacement over the whole think + action, at most 10 per attempt;
  - every decoded token counts as sampled compute, rejected draws included.
- **`step_check` is sound with respect to Lean.**
  - Dan's guided-tts validation: 0 false rejects in 314,541 (state, step) pairs, on Lean 4.34.1.
  - Re-checked here on our Lean (4.34.0) with the round-2 thinking model's own steps (13–20 problems, plain
    sampling): 166 of 166 checker rejects were rejected by Lean itself.
- **Step penalty** (`--rej_penalty P`). Every rejected draw gets advantage −P whatever its group's outcome:
  - each redrawn one (`keep_rejected`). By default (`--rej_penalty_scope diverge`) only its tokens from the
    first one where it departs from the draw accepted at the same state are scored: the shared prefix led to a
    valid step too, and penalising it would push down valid structure. With no accepted draw at that state
    the whole draw is scored. `draft` scores the whole think + action, `action` the action only.
    - On the Stage-1 model on 10–12-line problems, 71% of redrawn draws have an accepted draw at their state.
    - The shared prefix is 24% of a redrawn draw, so `diverge` scores 84% of the tokens `draft` would.
  - and the draw that ended a failed attempt, on top of that attempt's advantage.
  - A group is skipped only if its outcomes give no signal and it has no rejected draw. So groups where every
    attempt failed (or every attempt succeeded) still teach the model to avoid its invalid steps.
- **Scoring.**
  - The rejected draws are scored like any sampled tokens, under the sampler's phase masks.
  - The without-replacement exclusion of a redraw is not modelled. A redrawn step is scored as if pi_T had drawn
    it, a small bias toward accepted redraws.
  - The penalised draws' ratios fall below 1 − ε on purpose. So with the penalty, the clip-low alarms read the
    accepted items' clip-low (`clip_low_accepted`), which keeps the collapse signal they were built for.
  - With the penalty the loss is normalised by the accepted items' tokens: (sum over accepted and penalised
    tokens) / accepted tokens. An accepted token gets plain GRPO's per-token weight, the penalty comes on top,
    and plain vs guided is not confounded by a diluted outcome signal. `tokens_accepted_share` logs the
    normaliser's share of the scored tokens.
- **Monitoring** (`--monitor`, `row['monitor']`; `matched_eval/rl_dashboard.py` plots it). Per iteration:
  - per target length and per group pass-rate band: the groups, the share with a gradient, the pass rate, the
    share of the |advantage|-weighted token mass ("where the gradient comes from"), and its think and
    rejected-draw parts;
  - overall: the think / action and accepted / rejected split;
  - rejected draws per draw and per attempt, with the top causes;
  - `diverge_match`: the share of redrawn draws that found the draw accepted at their state (about 71% on the
    Stage-1 model; a systematic offset would show here), and `shared_prefix`.
  - The dashboard also plots think tokens per block per budget and the think / action entropy per iteration,
    since the penalty could push think length down in groups with no outcome signal.
- **Guided EI** is `ei.py --arm logical`. It fine-tunes on accepted proofs only: the found records keep the
  trace's non-rejected steps, and redraws never enter the trace (test `test_guided_ei_round_trains_on_accepted_steps`).

## Choice of P = 0.1

- With Dr. GRPO advantages (r − group mean), a group at pass rate p has a mean |A| of 2p(1 − p), which is
  0.2–0.5 here. So P = 0.1–0.25 is "comparable to a typical advantage" per token.
- But a guided attempt has many rejected draws. In the CPU smoke runs, the Stage-1 models on 10–12-line problems
  had 26–33% of draws rejected (4–6 per attempt), and at P = 0.25 the rejected draws carried 58–70% of the
  gradient mass.
- P = 0.1 brings that to roughly a third to two fifths at the start (same arithmetic; the smoke runs at
  P = 0.1 showed 13–46%), and the share falls as rejections do. Every invalid step is still pushed down, and
  saturated or all-failed groups still give a gradient.
- `REJ_PENALTY=0.25 bash line.sh ...` runs the stronger version. Watch `mass_rejected`, the accepted clip-low
  and the canary in the dashboard.

## Arms (one stage = 160k sampled attempts: 39 GRPO iterations × 256 × 16, or one EI round at k 32 over 5,000)

| arm | init | method |
|---|---|---|
| `grpo_think` | S1g3 | plain GRPO |
| `grpo_think_v` | S1g3 | GRPO, logical arm + step penalty |
| `ei_think` | S1g3 | plain EI |
| `ei_think_v` | S1g3 | guided EI (logical arm) |
| `grpo_nothink_v` | T0S | GRPO, logical arm + step penalty |
| `ei_nothink_v` | T0S | guided EI |
| `grpo_nothink`, `ei_nothink` | T0S | optional plain controls |

- **Stages:** L2 (10–12 lines), then L3 (13–15), from `runs/pools/curriculum/L{2,3}/pool.jsonl` (`POOL_L2` /
  `POOL_L3` override).
- **Pre-flight** (`grpo_v2/preflight.py`): both pools exist, are in band, and share nothing with each other or
  with any evaluation set. "Nothing" means no prompt, no renaming key, and no renaming or curried-normal-form
  class: `classes()`, copied from `charles_eval/build_clean.py`, Charles's audit rule. The class check is the
  one that covers textbook72 and Charles's rows, which carry no keys. It found 94 shared classes between the
  first L2 split (`L2/old_split2`) and hard712 test_1k / dev_500, so that split would have been refused. The evaluation sets are the sealed 13–20 and 21–30 sets, the hard12 test,
  hard712 test/dev, textbook72, charles173, hand_proved_14, and the curriculum test splits.
- **Step cap:** `--max_steps 96` for both GRPO and EI (eval's value). v1 used the default of 48. On
  13–15-line problems 3–6% of B64 attempts exceed 48 steps at T 0.8, more at GRPO's T 1.0 and with redraws.
  Capped attempts count against the 5% capped alarm, so v1's cap would have raised false collapse alarms
  (and rollbacks) for GRPO only.
- **GRPO settings** are tonight's (`grpo_think/line.sh`) plus `--monitor`:
  - lr 1.5e-4, `--overlong fail`;
  - canary + rollback, and a KL anchor of 0.02 to each stage's init;
  - budgets 0/64/256 with effort tokens for thinking.
- **EI settings** are `seed2/line.sh`'s (`--select shortest`, the default `--max_per_thm`).

## Learning-rate schedules (2026-10-13, branch `leon/grpo-v2-sched`)

- **Today** each stage is a fresh `--init`, so it gets a new AdamW and step count. The lr ramps linearly over the
  first 10 AdamW steps (`--warmup`, which `line.sh` does not set). At 4 steps per iteration that is 2.5 iterations:
  iteration 1 runs at 0.1–0.4 × lr, iteration 2 at 0.5–0.8 ×, iteration 3 at 0.9 × and then 1 ×. After that the lr
  stays at 1.5e-4 to the stage's end. The only decay is a rollback's halving (at most 3), which in practice fired
  late in L3.
- **`grpo.py --lr_schedule {constant,cosine,wsd}`** (module docstring) multiplies that lr by a factor m(t) for each
  iteration t = 1–39 of the stage:
  - `cosine`: m goes from 1 down to `--min_lr_frac` along a cosine after the warmup. This is EI's fine-tune
    schedule, counted in iterations.
  - `wsd`: m stays at 1 until the last round(`--decay_frac` × 39) iterations, then falls to `--min_lr_frac`.
    `--decay_shape` sets how: `linear`, `cosine`, or `sqrt` (the 1−sqrt cooldown of Hägele et al. 2024).
  - `--warmup_iters W` replaces the 10-step ramp with a ramp of t / W over iterations. Without it the step ramp
    stays.
- **Rollbacks** still halve the lr, and the halving multiplies the schedule. A rollback restores the training state,
  step count included, but not the iteration number. So the schedule keeps following the loop's counter, and a
  stage with rollbacks still ends at `--min_lr_frac` × 0.5^rollbacks.
- **Resume:** `--resume` continues the schedule, and refuses a checkpoint whose run had a different one.
- **Logged:** `lr`, `lr_mult`, `lr_ramp` and `alarm_cfg.lr_schedule` in the rows, and `lr L = 1.5e-4 x kind m x
  scale` in the guard line.
- **Default unchanged.** Without `--lr_schedule` the rollouts, rows, log lines, weights and args.json match
  fa28f6e: digests of four tiny-model configs, in `code/tests/thinking/test_grpo_sched.py`. Before its decay
  starts, WSD is bit for bit the constant run.

Arms in `line.sh`. The schedule flags are set in one place, `SCHED_C` / `SCHED_W`:

| suffix | L2 | L3 |
|---|---|---|
| `ARM_c` | cosine, 3-iteration warmup, down to 0.1× | the same |
| `ARM_w` | ARM's command, no schedule | WSD with today's warmup: constant to iteration 31, then 1−sqrt over the last 8 iterations down to 0.3× |
| `ARM_wb` | none: starts from ARM's finished L2 checkpoint, checked as for `ARM_b` | as `ARM_w` |

All three are refused for EI and in combination with `_h` or `_b`. The dry-run harness (`launch/dry/`) covers
them, and every earlier arm's commands are byte-identical.

## L3 pilots: refill, pass-rate filter, unlikeliness reward (2026-10-13, branch `leon/grpo-v2-pilots`)

**Why.** Late in L3 about half of the groups give no gradient. In the three `grpo_think_vs` L3 runs, iterations
30–39 skip 46–51% of groups, and almost all of them are all-solved: 46–51% all solved, 0.5–0.8% all failed, at a
training pass@1 of 90–91%.

**The prompt sampler.** It works as coded, but its prior for unseen pairs is optimistic, and that prior sets most
draws.
- Weights are p(1 − p) + 0.03, with p a Beta(1, 1) posterior mean. So a pair never drawn sits at p = 0.5, which is
  the largest weight.
- L3 has 15,000 (target, budget) pairs, and a stage draws 39 × 256 = 9,984 of them. Pairs drawn for the first time
  are 81–86% of the stage's draws, and still 67–76% in iterations 30–39 (the curriculum's `seen` count).
- The seen pairs' mean p is about 0.87 at B64/B256 and 0.75 at B0.
- In the final L3 checkpoints, unseen pairs hold 63–70% of the draw weight.
- Interpretation: the weighting can only steer the remaining third of the draws, so most groups come from a
  near-uniform sweep of a pool that the policy mostly solves.

**`--max_pass` only acts on seen pairs.** In the final checkpoints:
- p > 0.9 holds about 9% of the draw weight: some 3,400–3,500 pairs, those last seen solved 16 of 16 times.
- p > 0.6 holds 26–30% of the weight: 92% of the seen pairs, since one group at 10/16 already gives p = 11/18.
- At 0.6 the filter would therefore move draws onto unseen pairs, the main source of all-solved groups.
- Olmo 3's 62.5% is an offline filter on the SFT model, not a running estimate.
- The pilot uses **0.9**.
- A fix aimed at the cause would be an informed prior for unseen pairs (for example the mean p of seen pairs at
  that budget, or of the same target at other budgets). It is not implemented: it changes the sampler for every arm.

**What `grpo.py` adds** (all opt-in, left out of args.json unless given; module docstring, "Pilot options"):
- **`--refill`** (DAPO's dynamic sampling, Olmo 3's active sampling), with `--refill_max` 2.
  - After the rollouts, it draws ceil(deficit / signal share) more pairs, at most `--prompts`, with the same
    curricula, excluding pairs already drawn this iteration. Up to 2 extra rounds, until 256 groups have an outcome
    signal.
  - Groups are trained in draw order up to the target. Later ones (the surplus) only update the curriculum.
  - Every sampled group counts in the curriculum and in the sampling rows.
- **Compute accounting for matching EI.** `row['refill']` logs, per iteration:
  - groups, attempts and tokens sampled. `tokens` is the sampler's count: think + action, rejected draws included.
  - the run totals (`groups_cum`, `attempts_cum`, `tokens_cum`), with rolled-back iterations included and carried
    through a resume.
  - per round: the pairs drawn, those with a signal, and those never drawn before.
  - `skip_before` (round 0) and `skip_after` (the share of the 256 target left without a signal group).
  - A plain stage samples 39 × 4,096 = 159,744 attempts, the same as one EI round at k 32 over 5,000 rows (160,000). A
    refill stage samples more, and `attempts_cum` says how much. A matched-compute EI comparison then needs EI's k
    or rows raised to that total, or GRPO cut to the same `attempts_cum`.
- **`--max_pass x`.** Pairs with p > x are never drawn again.
- **`--unlikely_beta β`** (He, Fried & Welleck 2025, arXiv 2506.02355, Sec. 4.1).
  - A solved attempt's reward is 1 − β ρ / n. Here ρ is the number of the group's n unmasked attempts that are less
    likely than it under π_old, ties counting half, so 0 is the least likely.
  - This equals the paper's R(1 − β (G − rank)/G) with rank 1 the most likely, and its released code's
    `argsort(argsort(logp)) / G`.
  - Likelihood is the mean log π_old per scored token (think + action, under the phase masks, forced `</think>`
    excluded), as in the released code. A sum would mostly rank by length and reward long proofs.
  - The log-probs are the ones the PPO ratio already uses, so there is no extra forward pass.
  - Only groups with an outcome signal change. All-solved groups stay skipped, as in the paper.
  - The advantage stays Dr. GRPO's r − mean.
  - `--rej_penalty`'s −P stays on top for rejected draws. Under `--overlong fail` capped attempts are failures and
    are ranked.
  - `row['unlikely']` logs the deductions and the within-group correlation of rank with length.
- **Default unchanged.** Without the flags, digests of tiny-model runs equal those from a clean export of c4b8548:
  rollouts, rows, log lines, weights, args.json and batch dumps (`code/tests/thinking/test_grpo_pilots.py`, plus
  `test_grpo_sched.py`'s).

**Arms** (`line.sh`; flags in `PILOT_RFB` / `PILOT_ULB` only). Each is stage L3 only, from ARM's finished L2
checkpoint, checked as for `ARM_b`. Each is refused for EI and with any other suffix.

| suffix | L3 |
|---|---|
| `ARM_rfb` | `--refill --refill_max 2 --max_pass 0.9` |
| `ARM_ulb` | `--unlikely_beta 0.25` |
| `ARM_3pb` | the L3 3-premise rows as `--targets_extra` at 0.33, exactly as `_vs3` passes them, after `_vs3`'s extra-targets preflight; refused for `*_vs3` |

`line.sh` still runs the code in `nd-rl-v2` on `leon/grpo-v2`. `ARM_rfb` and `ARM_ulb` refuse until that worktree's
`grpo.py` has the pilot options (fast-forward `leon/grpo-v2` to this branch: the default path is digest-identical).
`ARM_3pb` needs no new code.

`grpo_think_vs3` is now a full arm: `grpo_think_vs` plus the 3-premise rows in both stages, as `grpo_nothink_vs3`.
An arm with 3-premise rows (`_vs3`, `_3pb`) is refused with "no 3-premise pool file …" while a `curriculum_3p`
file is missing.

```
bash line.sh GPU 2 grpo_think_vs_rfb    # L3 of grpo_think_vs seed 2, from runs/grpo_v2/grpo_think_vs_s2_L2
bash line.sh GPU 2 grpo_think_vs_ulb
bash line.sh GPU 2 grpo_think_vs_3pb
```

## Unseen-pair prior (2026-10-13, branch `leon/grpo-v2-prior`)

The fix aimed at the cause in the sampler finding above, as its own opt-in pilot: `grpo.py --unseen_prior mean` and
the launcher arm `ARM_pb`.

**Shrinkage is already in the estimate.**
- A pair's p is the Beta(1, 1) posterior mean (s + 1) / (n + 2) of its decayed counts. Its rate s / n is pulled
  toward 1/2 with weight 2 / (n + 2), and a never-drawn pair sits at 1/2.
- `half` (the default) keeps this.
- `mean` swaps the prior for Beta(2 m_b, 2 (1 − m_b)): the same strength 2, but with mean m_b. So
  p = (s + 2 m_b) / (n + 2). A never-drawn pair sits at m_b, and seen pairs are pulled toward m_b instead of 1/2, so
  the shrinkage stays consistent. With one group of 16 behind it, a pair moves by at most 2/18 of (m_b − 1/2).

**m_b, per budget and per curriculum.**
- m_b is the mean over budget b's seen pairs of their raw decayed rate s / n. It is not shrunk, so the estimate of
  the prior does not feed on itself.
- The extra targets' curriculum has its own m_b.
- It is recomputed at every draw.
- Fallback: 1/2 until 32 of the budget's pairs are seen (`PRIOR_MIN_SEEN`). In practice that only holds before the
  first update, since iteration 1 draws about 85 pairs per budget.

**Why budget and not target.**
- Budget is the curriculum's own summary grouping, and the budgets' means differ: m_b is 0.76–0.81 at B0 against
  0.90–0.93 at B64/B256 in the final v2 L3 states.
- A pair's same-target pairs at other budgets do correlate with its rate (r = 0.36–0.41 in the final v2 L3
  checkpoints). But their offset-corrected rate predicts it worse than the budget mean: squared error 0.043–0.053
  against 0.035–0.043.

**`--max_pass` now applies to seen pairs only.** Under `half` this is unchanged, because an unseen pair at 1/2 is
never above a threshold of at least 1/2.

**Expected effect.** This replays the final L3 curriculum states of the three `grpo_think_vs` runs under each prior,
so it is not a run of the new arm.
- The unseen pairs' mean weight falls from 0.28 to 0.13–0.19.
- Their share of the total draw weight goes:
  - from 90–91% to 87% at iteration 10;
  - from 80–85% to 74–80% at iteration 20;
  - from 63–70% to 53–58% at iteration 39.
- The effect is modest. Most seen pairs are solved too: their mean weight is 0.10–0.14 under `mean`, against about
  0.25 for a frontier pair. The weight also has a floor of 0.03.
- The effect compounds in a run, since fewer first draws leave more seen frontier pairs to redraw. The pilot
  measures it.

**Logged.** `row['prior']` is written whenever the option is given, `half` included. It has:
- the pairs drawn this iteration, and how many were never drawn before (every refill round);
- at the first draw: m_b per budget (`fallback`: still at 1/2), the seen and unseen pair counts, their mean weight,
  and the unseen pairs' share of the total weight;
- an `extra` entry for the extra curriculum.

The `prior it N:` line prints the same.

**Default unchanged.** `test_grpo_prior.py` checks it:
- without the option, the digests equal c4b8548's and 0de954e's;
- with `--refill`, `--max_pass` and `--unlikely_beta` together, the digest equals a clean export of 0de954e;
- `--unseen_prior half` gives the same rollouts and weights.

**Arm.** `ARM_pb` (`PILOT_PB="--unseen_prior mean"`) is stage L3 only, from ARM's finished L2. It has the same
checks and refusals as `ARM_rfb`, and refuses until the code worktree's `grpo.py` has `--unseen_prior`.

```
bash line.sh GPU 2 grpo_think_vs_pb     # L3 of grpo_think_vs seed 2, from runs/grpo_v2/grpo_think_vs_s2_L2
```

## Merged recipe (2026-10-14, branch `leon/grpo-v2-merged`)

`ARM_mb`, approved by Leon, at least for `grpo_think_vs_mb`. It is stage L3 only, from ARM's finished L2 checkpoint,
with the same checks and refusals as `ARM_3pb`: refused for EI, for `*_vs3`, and with any other suffix. It combines
three pilots. The flags come from the existing constants, so each value is set in one place:

- `_wb`'s cooldown (`SCHED_W`: `--lr_schedule wsd --decay_frac 0.2 --min_lr_frac 0.3 --decay_shape sqrt`);
- `_3pb`'s 3-premise rows (L3_3p at `--targets_extra_frac` 0.33, after `_vs3`'s extra-targets preflight);
- `_ulb`'s unlikeliness reward (`PILOT_ULB`: `--unlikely_beta 0.25`).

It has no refill and no unseen prior.

**Reads.**
- First the usual reads, plain and guided, on the final checkpoint.
- Then `line.sh` averages the stage's L3 checkpoints at it 25, 30 and 35 (rolling) and 39 (final). This is
  `pilot4.sh`'s averaging:
  - every floating-point tensor of `state` is averaged in float64 and cast back to its dtype;
  - the other fields come from it 39, `train_state` is dropped, and `extra.averaged_from` lists the inputs;
  - the result is written to a temporary file first and then renamed.
- The averaged checkpoint goes to `runs/grpo_v2/ARM_avg_sSEED_L3/ckpts/ARM_avg_sSEED_L3_it39.pt`.
- It is read plain only, on t1320 (k 16), tb72 (k 64), charles173 (k 64) and t2130 (k 8). Reads are at B64 (B0
  for no-think) and T 0.8, under the arm name `ARM_avg`, so rows land in
  `eval/v2/<set>_ARM_avg_sSEED_plain_k<k>/`.

**Reruns and failures.**
- A rerun skips the averaging when the averaged checkpoint exists, and skips each read whose rows exist.
- If a rolling checkpoint is missing, the averaged reads are refused with "no … to average (an alarm iteration?)".
  An iteration with an alarm saves an alarm checkpoint instead of its rolling one.
- Under `SMOKE=1` the mean is of the last checkpoint alone, since a 2-iteration stage keeps no rolling checkpoint.

```
bash line.sh GPU SEED grpo_think_vs_mb    # L3 of grpo_think_vs, from runs/grpo_v2/grpo_think_vs_sSEED_L2
```

## Reads (after the last stage, on the same GPU; thinking at B64, no-think at B0; rows under `runs/eval/v2/`)

| set | k | arms |
|---|---|---|
| t1320 (986 valid; 16–20 = a `min_lines` filter of the rows) | 16 | plain, logical |
| textbook72 | 64 | plain, logical |
| charles173 (Charles's 2026-10-02 release: candidate_v0 87 + candidate_v1 72 + batch3 14) | 64 | plain, logical |
| hand_proved_14 (20–102-line hand proofs; `--max_steps 160`) | 64 | plain |
| t2130_n1000 (a fixed random 1,000 of test_21_30, seed 2130) | 8 | plain |

charles173 and hand_proved_14 are eval only:
- every prompt loads through `code.thinking.eval` unchanged and tokenises for both models (atoms P–S only);
- none of them appears in K12, the RL pools or textbook72, by prompt or by canonical hash;
- `derived.jsonl` is not used.

## Launch

```
cd /scratch/cowifornia/nd-rl/think/runs/grpo_v2
nohup setsid bash line.sh GPU SEED ARM > /scratch/cowifornia/nd-rl/think/runs/logs/grpo_v2_ARM_sSEED.log 2>&1 < /dev/null &
SMOKE=1 bash line.sh 0 0 ARM    # a tiny CPU dry run of the whole line (2 iterations, k 2, 2 eval problems)
```
