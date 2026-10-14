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
