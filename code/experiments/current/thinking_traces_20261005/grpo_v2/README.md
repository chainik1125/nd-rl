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

## In-context verifier feedback (2026-10-14, branch `leon/grpo-v2-feedback`)

Dmitry asked at the 2026-10-09 group meeting: "Combine Robbie's in-context verifier idea with GRPO to provide gradient
signal from failures, not just successes." This adds code, tests and a dry run only. Nothing was trained.

**Robbie's format.**
- His verifier-in-context work is `retry_ei_20261008` on `robbie-experiments` (summary
  `experiment-summaries/2026-10-08-retry-ei`). Its episodes are whole-proof retries:
  `STATEMENT attempt_1 <eos> <err:CODE> [name] STATEMENT attempt_2 <eos> ...`.
  - It adds 32 new error-code tokens. The `blank` arm has a single `<fail>` token instead.
  - Retries were accepted no more often than independent samples.
  - Failed proofs in context cut exact repeats: attempt 2 equalled attempt 1 in 20% of episodes, against 30% for
    two independent failed samples. The error codes and `<fail>` did the same.
- His test-time-scaling resampler, `engine._swor_shift`, is the without-replacement shift our guided arms use. The
  "50–55% of rejections redraw a segment already rejected" figure comes from that work.
- The `plain-state-feedback` autoresearch commits (705/707) feed hidden states back (Coconut-style), not verifier
  output.

**Ours, and the rendering chosen.**
- S1g3 was trained only on `searchtrace` traces. Their think blocks are goal lines `ctx ⊢ G := SKEL`. A candidate
  tried and found dead is the same line with `<no>`, plus a countermodel.
- In the first 20,000 e64 traces, 93% of the blocks with a dead end have it on their first line, before the viable
  reduction.
- So a rejected action is written as the dead reduction it tried, and the retry's think block starts with these
  lines (`code/thinking/feedback.py`, `render`):

```
rejected action                                        line prefilled in the retry's think block
have n21 : R := n20 .2 ;                               ⊢ R := n20 .2 <no>
have n1 : S := h1 ;                                    ⊢ S := h1 <no>
have n3 : ( Q → S ) := ( fun ( n4 : Q ) => by          ⊢ ( Q → S ) := fun ( n4 : Q ) => ⊢ S <no>
have n39 : P := ( fun ( n40 : ( P ∧ S ) ) => by        ⊢ P := fun ( n40 : ( P ∧ S ) ) => ⊢ P <no>
have n2 : ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : Q ) => by
          ⊢ ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : Q ) => ⊢ ( Q ∨ P ) ) ( fun ( n4 : Q ) => ⊢ ( Q ∨ P ) ) <no>
exact n2                                               ⊢ G := n2 <no>        (G: the focused goal)
```

- **The retry's prompt.** It is `state <eB> <think> LINE_1 <nl> … LINE_m <nl>`, and the model's own think tokens
  follow. At B = 0 it is `<think> LINES </think>`, then the action.
- **Which lines.** One per distinct action rejected at this state, the newest 8 at most. They reset when a draw is
  accepted.
- **Trace syntax.** Each line's reduction parses with the trace DSL's own skeleton parser, `trace_dsl.Skel` (tested).
- **No reason.** The checker's reason is not written: the vocabulary has no tokens for it, and Robbie's error codes
  did no better than `<fail>`.
- **No countermodel.** The verifier says that the step is invalid, not why the claim fails, and a truth-table
  countermodel would be a second oracle. In the first 20,000 e64 and e256 traces, 99.7% of the `<no>` lines carry one,
  so the bare form is the rare one.
- **No line** for:
  - actions outside the grammar, and truncated ones;
  - `Or.elim` on a non-disjunction, whose second branch would have no binder type.
  The retry is then a plain resample, counted as `unrendered`.

**Design** (`sample.think_generate(feedback='context')`, `grpo.py --feedback context`, `eval.py --arms feedback`):
- **Retries.** Under feedback, `--max_rej` counts retries per step. A state's (`max_rej` + 1)-th rejection ends the
  attempt (`fail:max_rej`).
  - So `--max_rej 0` is the stop arm. Tested: the same rollouts, and the same GRPO weights as the stop digest.
  - There is no without-replacement exclusion. Each retry is an exact sample of π_T given its prompt, so the PPO
    ratio is exact. The repeat rate is measured, not suppressed.
- **Masking.** The feedback tokens sit in the retry's decoder input, an item's prompt. So they are out of the loss,
  the PPO ratio and the think ban mass, exactly as prompt tokens are. The KL anchor reads only its reference steps.
  - A retry's input ends inside an open think block (`grpo.open_think`). Its own think tokens are therefore scored as
    think tokens, under the think ban.
  - Retries and rejected drafts are policy tokens.
- **Cost of a rejected draft.** `--rej_penalty` P works as before:
  - every retried draft is kept and scored at −P, whatever the outcome;
  - the draw that ends an attempt gets −P on top of the failure;
  - the `diverge` scope compares a draft with the draw accepted at its state. It matches them by state, because the
    prompts differ by the lines.
- **The redraw-collapse risk.**
  - A recovered invalid first draw costs −P only. In the stop arm the same draw ends the attempt: it gets the
    failure's advantage (0 minus the group's pass rate, about −0.9 late in L3) and −P on top.
  - The per-step cap is the other cost: past `FB_MAXREJ` retries the attempt fails, as in the stop arm.
  - At the L3 init below, two retries left 41% of rejected first draws unrecovered at B64, 52% at B0 and 32% at B256.
- **Logged.** Per iteration in `row['feedback']` and the `fb it N:` line (`feedback.metrics`), overall and per budget:
  - rejections and retries per attempt;
  - the retries accepted, overall and per retry index;
  - the repeat rate of identical rejected actions at a state, of the rejected retries and of all rejections
    (Robbie's denominator);
  - feedback tokens and lines per attempt, and the unrendered share;
  - the share of solved attempts that needed a retry.
  - Eval cells add the same: `feedback` in `summary.json`, and `fb` in the rows.

**At the pilot's init.** These are observed CPU numbers on the checkpoint `grpo_think_vs_s2_L2_it39`: 24 L3 pool rows ×
8 attempts, T 1.0, `--max_rej 3`, one seed. "Blind" is the same per-step retry loop with no lines.

| B | retries | pass@1 | retry accepted | rejected retries that repeat an action rejected at the state |
|---|---|---|---|---|
| 64 | feedback | 85.9% | 31% (43/137) | 39% |
| 64 | blind | 89.6% | 43% (70/162) | 25% |
| 0 | feedback | 66.1% | 25% (76/305) | 44% |
| 0 | blind | 69.8% | 31% (114/362) | 19% |
| 256 | feedback | 87.0% | 40% (68/171) | 31% |
| 256 | blind | 89.1% | 47% (87/187) | 17% |

- The model has not been trained to use the lines, and it often copies a dead line's term into its action (examples
  in the smoke below).
- So at init, feedback does worse than blind resampling at every budget.
  - The B0 repeat gap is large (two-proportion z ≈ 6). The other gaps are z ≈ 1.3–2.3. Attempts of one problem
    are correlated, so with 24 problems these z values overstate the evidence.
- The pilot measures whether GRPO teaches the model to use the lines. Compare the repeat rate in its `fb it` lines with
  the blind numbers here.
- Retries recover mostly on the first try: 32/70, 9/38 and 2/29 at retries 1, 2 and 3 (B64). The pilot uses
  `FB_MAXREJ=2`.

**Smoke on the Stage-1 model** (CPU, `S1g3_s2/model.pt`, 4 L2 + 4 L3 pool rows, 2 iterations × 4 prompts × group 4,
the launcher's L3 flags for `grpo_think_vs_fbb` at SMOKE=1 sizes; output in `/scratch/cowifornia/nd-rl/think/fb_smoke/`):
- It runs end to end with the real Lean gate, the canary, the KL anchor and rollbacks. The log lines are:

```
fb it 1: rej/att 4.12 retries/att 3.12 retry acc 28% (14/50) repeat 36% of rejected retries (20% of rejections) fb tok/att 28.8 lines/att 1.62 unrendered 34% assisted - of solved
fb it 2: rej/att 3.94 retries/att 3.19 retry acc 39% (20/51) repeat 29% of rejected retries (14% of rejections) fb tok/att 34.0 lines/att 2.06 unrendered 24% assisted 25% of solved
```

- Iteration 1 (pass@1 0%) raised a clip-low alarm on the accepted tokens and rolled back. With no solved attempt, the
  gradient is the penalised draws only. The draws that end attempts count as accepted items in the alarm's basis, as
  they do in the stop arm; the retried drafts do not. Most likely an effect of the tiny batch (interpretation).
- Two retries from its rollouts (B0, the action only after the block):

```
state … n33 : ( ¬ R ) <nl> n35 : ( ¬ Q ) <nl> ⊢ False <act>
rejected   have n36 : ( ¬ ( ¬ ( S ∧ Q ) ) ) := n51 ;                  (S:unbound)
retry      <e0> <think> ⊢ ( ¬ ( ¬ ( S ∧ Q ) ) ) := n51 <no> </think>
           have n36 : ( ¬ ( ¬ ( S ∧ Q ) ) ) := ( fun ( n37 : ( ¬ ( S ∧ Q ) ) ) => by        accepted

state … n11 : P <nl> n13 : ( P → Q ) <nl> ⊢ S <act>
rejected   have n14 : ( ¬ P ) := n7 n13 ;                             (L:app-arg)
retry      <e0> <think> ⊢ ( ¬ P ) := n7 n13 <no> </think>
           have n14 : ( ¬ P ) := n7 n13 ;                             copied at both retries: fail:max_rej
```

**Arm `ARM_fbb`** (`line.sh`; its flags are in `PILOT_FBB` and `FB_MAXREJ` only). It applies to `grpo_think_v`,
`grpo_think_vs` or `grpo_think_vs3`; `grpo_think_vs_fbb` is the one asked for.
- **Stage.** L3 only, from ARM's finished L2 checkpoint, with `--feedback context --arm logical --max_rej 2
  --rej_penalty 0.1`. It is checked and refused as for `ARM_rfb`.
- **Refusals.** EI, no-think arms (they have no think block), arms without the verifier, and any other suffix. It is
  also refused until the code worktree's `grpo.py` and `eval.py` have the options; the guard checks `FEEDBACK_MODES`
  and `READ_ARMS`.
- **Reads.** The usual plain and logical reads, plus the feedback arm on t1320 and charles173 (`--max_rej 2` retries
  per step).
  - These run alongside the logical reads, not in their place: logical vs feedback on the same checkpoint is the
    test-time comparison.
  - The plain read stays the headline.

```
bash line.sh GPU 2 grpo_think_vs_fbb    # L3 of grpo_think_vs seed 2, from runs/grpo_v2/grpo_think_vs_s2_L2
```

## EI knob pilots (2026-10-16, branch `leon/grpo-v2`)

`ARM_kKNOB` screens one EI setting at a time, as `ARM_mb_kKNOB` does for GRPO (`launch/README.md`, item 9). It takes an
EI arm (`ei_think`, `ei_think_v`, `ei_nothink`, `ei_nothink_v`); `ei_think_v_kKNOB` at seed 0 is the one asked for.

**Stage.** L3 only, from ARM's finished L2 checkpoint, `runs/grpo_v2/ARM_sSEED_L2/ckpts/ARM_sSEED_L2_r1.pt`. The arm is
refused when that checkpoint is missing; L2 is never retrained. Output goes to `runs/grpo_v2/ARM_kKNOB_sSEED_L3/`.
ARM's own L3 is one round: 5,000 targets x 32 attempts (160k, matched to 39 GRPO iterations), then one 600-step
fine-tune. Each knob changes one setting, set in `line.sh`'s knob block and nowhere else:

| KNOB | change from ARM's L3 |
|---|---|
| `rr` | nothing: ARM's L3 again, the noise control (GPU runs are not bit-reproducible) |
| `3p` | `--targets runs/pools/curriculum_3p/L3_mix_ei.jsonl`: 3,333 L3 pool rows and 1,667 L3 3-premise rows (below) |
| `ret0` | `--retain 0`: the fine-tune mix is the found proofs alone (ei.py's default adds 20,000 Stage-1 records) |
| `t10` | `--temperature 1.0` for the training samples (0.8); the reads stay at 0.8 |
| `r2` | `--rounds 2 --k 16 --ft_steps 300`: the same 160k attempts and 600 fine-tune steps, in two rounds of 80k and 300 (below); the stage ends at `_r2.pt` |

**Reads.** The GRPO knob pilots' reads: plain only, on t1320 (k 16), t2130 (k 8) and charles173 (k 64), under the
full arm name (`eval/v2/<set>_ARM_kKNOB_sSEED_plain_k<k>/`). Knob selection never sees textbook72, so there is no
tb72 read and no guided or hand14 read. `KNOB_FULL=1` gives a finished pilot ARM's usual reads instead. The finished
L3 and the reads already there are skipped, so on a finished pilot this adds tb72, the guided reads and hand14. HEAD
then goes to `ARM_kKNOB_sSEED.reads_commit.txt`, and the training's `base_commit.txt` is kept.

**Refusals.** An unknown knob or a GRPO knob (`ei_think_v_kx66`), anything after the knob, and any suffix before it
(`ei_think_v_h_krr`). `_mb_k` on an EI arm gets `_mb_k`'s refusal, and an EI knob on a GRPO arm
(`grpo_think_vs_kret0`) is refused with a pointer to `ARM_mb_kKNOB`.

**The `3p` mix** (`grpo_v2/build_mix_ei.py`). GRPO's merged recipe draws a third of each iteration's groups from the
L3 3-premise rows (`--targets_extra L3_3p.jsonl --targets_extra_frac 0.33`). EI samples every target k times in one
round, so `3p` gets the same share from a fixed file instead.
- The builder samples 3,333 rows of `curriculum/L3/pool.jsonl` and 1,667 of `curriculum_3p/L3_3p.jsonl` (1,843 rows),
  without replacement. It shuffles them together with seed 20261016, so every run seed trains on the same mix.
- Rows are copied byte for byte and keep their own names, which are checked unique (EI keys found proofs by name).
- The mix built on 2026-10-16 has md5 `8748b4b17d452485fe45095b30263ecc`. Building it again gives the same bytes.
  - Shortest-proof lines (`min_lines`) of the 3-premise rows: 11: 167, 12: 1,143, 13: 312, 14: 45 (mean 12.14).
    Of the pool rows: 13: 2,742, 14: 455, 15: 136 (mean 13.22).
  - 1,108 of the 1,667 3-premise rows have their original in the mix as well. GRPO's draws can pair them the same way,
    since every L3_3p row's original is in the L3 pool.
  - Its first 8 rows (what `SMOKE=1` uses) are 4 of each kind.
- Preflight: the same as the GRPO extra targets get (`preflight.py` on the L2 pool and the mix). It checks prompt,
  renaming-key and renaming / curried-class overlap with the L2 pool and with all 11 evaluation sets, including
  textbook72 and the sealed 13-20 and 21-30 sets. Only the band line may fail, and all 22 evaluation-set lines must be
  `ok`. It passed: 0 shared prompts, keys or classes anywhere. The only failure is `1310 rows outside 13-15`, the
  3-premise rows below the band (`L3_mix_ei.preflight.log`).
- The pass record `L3_mix_ei.preflight_pass.txt` holds `PREFLIGHT PASSED <md5 of the mix>`. The builder writes it only
  when the preflight passes, and deletes it before writing a new mix. `line.sh` refuses `ARM_k3p` without the mix or
  without a pass record naming the mix's current md5. At launch it also runs the same preflight again, into
  `preflight_ARM_k3p_sSEED_extra.log`, as it does for the GRPO extra targets.
- `L3_mix_ei.json` has the counts, the seed and the md5s of the mix and its two sources.

**ei.py, checked for `ret0` and `r2`** (no code change):
- `ret0`: `build_mix` takes `rng.sample(train_records, min(0, n))`, which is empty and draws nothing from the rng.
  The mix is then the found proofs alone: at most 4 per target, x `--rl_weight` 4 (78,156 records in ei_think_v s0's
  L3). The fine-tune still takes 600 steps of 128 records, now about one pass over those records instead of over
  98,156. A round with no found proof skips the fine-tune as before, for any `--retain`. `--train` is still loaded
  (vocabulary check, effort relabelling), but nothing from it is used.
- `r2`: within one invocation, round 2 loads round 1's checkpoint. It samples 5,000 x 16 again (round seed
  `seed * 1000 + 2`) and adds the new proofs to round 1's. `found_2.jsonl` is cumulative, one entry per action
  sequence, and with `--select shortest` a shorter think variant replaces the kept one. Round 2 then fine-tunes `_r1`
  again, on the cumulative picks plus a fresh 20,000 retained records.
  - `--ft_steps 300` (a follow-up the same day) makes each round's fine-tune 300 steps, so the two total one round's
    600, as the sampled attempts total one round's 160k. The difference meant to be tested is that round 2 samples
    from `_r1` (more on-policy). The code also differs in three smaller ways. The lr schedule is two cycles (each
    fine-tune has ei.py's 50-step warmup and its cosine from 3e-4 to 3e-5) instead of one. Each fine-tune draws
    300 x 128 records from its mix, instead of 600 x 128 once. Round 2's mix is the cumulative picks with a fresh
    retained sample.
  - `line.sh` does not resume. An interrupted `r2` stage reruns from round 1 on the next launch, because an `_r1.pt`
    alone does not end the stage. ei.py could resume (`--resume --start_round 2` reads `found_1.jsonl` and `_r1.pt`),
    but the other EI stages do not use it either.
  - Timeout: the attempts and fine-tune steps are one round's, plus a second checkpoint load and round setup, so
    `r2` should take about as long as ARM's one-round L3: 7,709 s for ei_think_v s0 on ynez, and 11,838 s for the
    slowest so far (ei_think_v s2). That is well inside the stage's 21,600 s timeout. The watchdog keeps EI's 4 h
    expectation, so its OVERRUN alert comes at 6 h, the same as the timeout.

```
bash line.sh GPU 0 ei_think_v_k3p    # L3 of ei_think_v seed 0 on the mix, from runs/grpo_v2/ei_think_v_s0_L2
```

## Mini runs (2026-10-16, branch `leon/grpo-v2`)

Leon's knob sweep, for a science of what makes GRPO generalise (to longer proofs, to other sources). Each run is the
primary recipe in miniature, the knob active for the whole run, with in- and out-of-distribution reads. Nothing here
has been trained yet. The knob list is still open (the values are parameters of the launcher).

**Data** (`grpo_v2/build_mini.py`, into `runs/pools/mini/`; seed 20261016 for L2, 20261017 for L3, never a run seed).
- `L2_mini.jsonl`: 1,000 of the L2 pool's 5,000 rows (min_lines 10 / 11 / 12: 273 / 417 / 310, the pool's shares).
- `L3_mini.jsonl`: 1,000 of the L3 pool's 5,000 (13 / 14 / 15: 804 / 159 / 37).
- `L3_3p_mini.jsonl`: the 369 3-premise rewrites of those L3 rows, the same fifth of `L3_3p.jsonl`'s 1,843.
  - Every L3_3p row rewrites exactly one L3 pool row (`orig_name`), and in the full recipe every 3-premise target's
    original is an L3 target. So the L3 subset is drawn in two strata in the pool's proportions: 369 of the 1,843 rows
    with a rewrite and 631 of the 3,157 without. The 3-premise subset is exactly those 369 rows' rewrites.
- Rows are copied byte for byte, in file order. A rebuild gives the same bytes (md5s `e22652a9…`, `f5c95cd9…`,
  `74469403…`, in `mini.json`).
- Preflight, as line.sh runs it: the two subsets (bands; no shared prompt, key or renaming / curried class with each
  other or any of the 11 evaluation sets, the L2 / L3 `test_1k` splits included) passed. The L2 subset against the
  3-premise subset failed only the band line (281 of the 369 rewrites are under 13 lines). All 22 evaluation-set lines
  are `ok` in both.
- The pass record `preflight_pass.txt` names each subset's md5. line.sh refuses a mini arm while a subset is missing or
  differs from its record, and runs both preflights again at launch.
- Nested subsets for the pool knobs: `L2_mini{250,500}.jsonl` are the first 250 / 500 of a fixed shuffle (seed
  20261018) of `L2_mini`'s rows, so 250 ⊂ 500 ⊂ 1,000. `L3_mini{250,500}` are nested the same way within each stratum
  (seeds 20261019 with a rewrite, 20261020 without), 92 / 184 rows with a rewrite. `L3_3p_mini{250,500}` are exactly
  those rows' rewrites (92 / 184). Rows stay in file order. Each size passed the same two preflights, and the pass
  record has all nine files. The rebuild left the three 1,000-row files byte for byte as they were.

**Stages.** Both 10 iterations (`MINI_ITERS`), chained. The commands come from `grpo_stage`, the same code as the full
recipe's, and differ from it only in what is listed here (the dry run checks this).
- **L2-mini** = `grpo_think_vs`'s L2 command: verifier stop (`--arm logical --max_rej 0 --rej_penalty 0.1`), KL anchor
  0.02 to its init, the constant lr after the 10-step warmup, all the safeguards. Init: always `stage1/S1g3_s0`, at every
  seed. The seed sets the RL sampling and data order only.
- **L3-mini** = `grpo_think_vs_mb_kkl0`'s L3 command, the primary recipe: KL anchor 0, SCHED_W, `--unlikely_beta 0.25`,
  the 3-premise subset as extra targets at 0.33. Init and anchor reference: L2-mini's `_it10`.
- Differences from the full commands: `--iters 10` / `--save_every 10` (39), the subsets as `--targets` /
  `--targets_extra`, `--save_rolling 2` (grpo.py's default 5; `--keep_rolling 100` keeps them all: it2, it4, it6 and
  it8, with it10 the stage's own checkpoint), the stage timeout 2 h (6 h) with a 1 h watchdog expectation (3.5 h), and
  L3-mini's init.
- **The cooldown scales by itself.** `--decay_frac 0.2` of 10 iterations is the last 2 (grpo.py rounds half up:
  floor(0.2 × 10 + 0.5) = 2; 39 gives 8). The lr factor is then 0.505 at it 9 and 0.3 at it 10 (1-sqrt, down to 0.3), so
  no schedule flag changes.
- **The warmup is kept at 10 AdamW steps**, the recipe's. At `--updates 4` that is 2.5 iterations: 25% of a mini stage
  instead of 6%. Iteration 1 trains at 0.1-0.4x the lr, iteration 2 at 0.5-0.8x, iteration 3 at 0.9x and then 1x. Each
  stage is a fresh `--init`, so both mini stages warm up, as both full stages do. Shortening it would change the
  per-step protection the recipe has, which is the thing kept fixed here.
- After each stage the line logs its `ALARM` lines (each alarm, the rollback it led to or why none) and its last guard
  line's rollback count, as `GRPO <stage> ALARM it N: …` and `GRPO <stage> rollbacks: lr xS (N rollbacks)`.
- A stage that ends without its last checkpoint (for example `--ban_stop`, exit 3, when the think-ban mass passes
  0.005) logs `GRPO <stage> FAILED: exit 3 (--ban_stop: …), no <ckpt>: the run counts as failed`, and the line stops
  with no reads.

**Knobs** (`grpo_think_vs_mini_kKNOB`, both stages unless noted, in line.sh's knob block only):

| KNOB | change |
|---|---|
| `rr` | nothing: the noise control |
| `t07`, `t085`, `t115`, `t13` | `--temperature 0.7 / 0.85 / 1.15 / 1.3` (training only; the reads stay at 0.8) |
| `lpNN`, `lpmNN` | `--len_penalty` +NN / −NN hundredths, two digits (`lp20` 0.2, `lp50` 0.5, `lpm10` −0.1; `lp00` refused) |
| `kl1` | a fork (below): `--kl_anchor_coef 0.1` in L3-mini (0), mirroring the full-scale `kl1` pilot (a negative control) |
| `g64` | a fork: `--prompts 64 --group 64` in L3-mini (the same 4,096 attempts per iteration; about 5 points worse at full scale) |
| `g64b` | `--prompts 64 --group 64` in both stages |
| `pool250`, `pool500` | the nested 250- / 500-row subsets in both stages, the same iterations and attempts (each row seen 4x / 2x as often) |
| `size` | the size-twin mix in both stages: a fixed half of the rows as their big-statement twins (below); also the size reads |
| `lossact`, `lossthink` | `--loss_tokens action / think`: the policy loss on that phase's scored tokens alone (below) |
| `advpos`, `advneg` | forks (below): `--adv_sign pos` at `--rej_penalty 0` / `--adv_sign neg`: one side of the outcome advantage (below) |

**Forks** (`kl1`, `g64`: the controls, seeds 0 and 1; `advpos`, `advneg` since 2026-10-18, so that L2-mini's KL
anchor keeps its relative weight and each is paired with the reference). A fork has no L2-mini of its own. Its L3-mini starts from the
reference `grpo_think_vs_mini`'s `_L2_it10` at the same seed, which is also its KL reference, so it differs from the
reference's L3-mini only in its knob. This removes L2-mini's run-to-run noise from the comparison.
- The fork waits until that L2-mini has finished: its `_it10.pt` is there *and* the reference's job log
  (`logs/v2_grpo_think_vs_mini_s<seed>.log`, pull_queue.sh's) has the line `GRPO grpo_think_vs_mini_s<seed>_L2 exit 0:`.
  `save_ckpt` is not atomic, so the file alone is not enough. It logs the wait, checks once a minute for up to 3 h, then
  logs FAILED and stops.
- It refuses unless the reference's L2-mini log ends at `lr x1 (0 rollbacks)` (run the control at another seed).

Dropped: `--updates` (it changes the AdamW steps per iteration and the warmup's length, an lr-like knob) and the lr and
unlikeliness knobs.

**Reads** (plain only, never textbook72; under the full arm name, `eval/v2/<set>_<arm>_s<seed>_plain_k<k>/`), of
L3-mini's last checkpoint (`MINI_READS`):

| set | k | what |
|---|---|---|
| `l2test` = `pools/curriculum/L2/test_1k.jsonl` | 8 | in distribution: the L2 pool's held-out split |
| `l3test` = `pools/curriculum/L3/test_1k.jsonl` | 8 | in distribution: the L3 pool's held-out split |
| t1320 | 8 | length OOD |
| t2130 | 8 | length OOD |
| charles173 | 64 | source OOD |

- The `test_1k` files (1,000 rows each) have the `prompt` eval.py reads and `min_lines` (10-12, 13-15). The preflight
  checks every pool and subset against both, as evaluation sets; they are not sealed.
- The reference, `rr` and the pool arms also read their own training subsets (`MINI_TRAIN`: `l2train`, `l3train`, k8;
  the pool arms' 250 / 500 rows), memorisation against the held-out splits.
- The T arms also read `MINI_PANEL` (`l3test` k8, t1320 k8, t2130 k4, charles173 k16) at their training temperature,
  as `<set>_<arm>_s<seed>_plain_k<k>_T07` (`_T085`, `_T115`, `_T13`), next to the 0.8 reads.
- Time-resolved reads hook: `MINI_ROLL="L2:4 L3:2 …"` (env) also reads `MINI_PANEL` of those rolling checkpoints, as
  `<set>_<arm>_L3it2_s<seed>_plain_k8`. One missing (an alarm iteration saves none) is skipped. It is off by default.
- A read named in two of a line's lists runs once.
- Rerunning a finished line (both stages there: to add `MINI_TRAIN` or `MINI_ROLL` reads) trains nothing and writes HEAD
  to `ARM_sSEED.reads_commit.txt`, as `KNOB_FULL=1` does; the training's `base_commit.txt` stays. `base_commit.txt` is
  written once (2026-10-18): a rerun that trains again (after a crash or a stop) records HEAD in `rerun_commit.txt`.
- `grpo_think_vs_mini_szpanel` (2026-10-18, a job, no training) reads the reference's L3-mini last at seed SEED on
  `SIZE_READS`: the L2 / L3 test_1k twins (`l2test_sz`, `l3test_sz`, k8) and their matched originals (`l2test_om`,
  `l3test_om`, k8: the test_1k rows whose twin was built, in the twins' order). `_ksize` reads them too. So both arms
  have paired size effects in distribution. About 0.5 GPU-h.
- `grpo_think_vs_mini_tpanel` (2026-10-18, a job, no training) reads the reference's L3-mini last at seed SEED with
  `MINI_PANEL` at T 0.7 and 1.3 (`TPANEL_TS`), under the reference's names with `_T07` / `_T13`: the reference point
  for the T arms' training-T reads (`t07`'s `_T07`, `t13`'s `_T13`). About 0.5 GPU-h (8 reads).
- A GRPO stage that an earlier run left without its last checkpoint (its dir has `args.json` or `metrics.jsonl`: a
  crash, a stop) has its dir, rolling checkpoints included, and its log moved aside to `<dir>.crashed_<UTC>` before
  it reruns, and the line says so (grpo.py would otherwise append to the old `metrics.jsonl`). Every GRPO arm, not
  only the mini ones.

**The size knob** (`_ksize`, 2026-10-18; `grpo_v2/build_size.py`, into `runs/pools/mini/`). The twins are in
`runs/pools/size_twins/` (branch `leon/size-twins`, its README): the same problem under one uniform substitution, a
bigger statement and the same proof, `min_lines` re-certified.
- Each mini subset keeps its rows and order. A row is in the twinned half when the first byte of sha256(salt + its
  name) is even: a fixed function of the original name, never a run seed. A chosen row is replaced by its twin when
  it has one. A 3-premise extra is replaced by its own twin exactly when its L3 parent was.
- The twin share: `L2_mini_size` 504 of 1,000 (50.4%; 1 chosen row has no twin and stays original), `L3_mini_size`
  476 of 1,000 (47.6%), `L3_3p_mini_size` 183 of 369 (49.6%).
- Statement tokens (whitespace tokens of the statement, without THM / SEQ / PRF; mean, p10 / p50 / p90, max):

  | file | original rows | twin rows | the mix |
  |---|---|---|---|
  | `L2_mini_size` | 26.9, 23 / 27.5 / 30, 32 | 60.2, 46 / 60 / 75, 87 | 43.7, 25 / 42 / 70 |
  | `L3_mini_size` | 28.5, 25 / 29 / 31, 32 | 60.8, 46 / 60.5 / 76.5, 87 | 43.8, 27 / 31 / 71.1 |
  | `L3_3p_mini_size` | 27.4, 25 / 28 / 30, 30 | 57.7, 43.2 / 57 / 74, 85 | 42.4, 26 / 30 / 69 |

  (The size-twin README's statement-size table counts the whole prompt, 3 tokens more.) `size.json` has the
  10-token histograms.
- Checks: line.sh's two preflights on the mixed files (the L2 and L3 files pass, 22 of 22 evaluation-set lines ok;
  the L2 file with the 3-premise file fails only its band line, as the original mini does), and no mixed row shares
  a prompt, `key` or `pkey` with the twin test sets. The pass record `preflight_pass_size.txt` is separate from
  build_mini.py's; line.sh refuses `_ksize` without it, and both `_ksize` and `szpanel` without the four test sets.
  A rebuild gives the same bytes.

**`--len_penalty` in grpo.py** (needed for the `lp` knobs; with `--len_penalty 0` nothing changes, see the tests below).
- An all-solved group's rewards 1 − lp·n/1000 differ by length alone. Such a group is *length-only*: it is trained,
  with the Dr. GRPO advantage −lp·(n − n̄)/1000 (no std division, so nothing amplifies it). Its items are kind `len`.
- (a) The outcome signal now leaves the penalty out. Before, these groups counted as having a signal, so the recipe's
  unlikeliness reward (mean |A| about 0.06 in a group of 16) applied to them and swamped the length term. It no longer
  does (the paper skips such groups), and `--refill` counts them as having no signal.
- (b) The loss's divisor leaves their tokens out, as it does for the step penalty's draws. Otherwise the about 45% of L3
  groups that are all solved late in a stage would dilute the outcome signal's per-token weight by about half, an lr
  cut.
- `row['len_only']` (groups, items, tokens, their share of the scored tokens), `row['monitor']['len_only_groups']` and a
  `len it N:` line, only with the penalty.
- Sizes, at about 460 tokens per attempt and a within-group mean |n − n̄| of about 70 tokens (0.15 × the mean: the kl0
  pilot's all-solved t1320 groups at B64). The reward change of a 460-token success and the length-only |A|:
  - `lpm10`: +0.046, 0.007;
  - `lp20`: −0.092, 0.014;
  - `lp50`: −0.23, 0.035.

  For comparison, an outcome |A| at pass rate 0.9 is 0.18.
- At `lp50` a success over 2,000 tokens scores below a failure (r < 0). In the kl0 pilot's B64 t1320 reads that is 0.7%
  of the solved 13-15-line attempts. B256 attempts run longer. Nothing in grpo.py assumes rewards in [0, 1] or "solved"
  = r of 1 (`ok` decides the curriculum, pass@1, `--max_pass`, the monitor's bands and who pays the unlikeliness
  deduction), so negative and >1 rewards (`lpm`) run as they are.
- Tests: `tests/thinking/test_grpo_len_penalty.py`. `test_grpo_pilots.py`'s c4b8548 digests still hold at lp 0, for
  `plain` and for `mixed0` (the `mixed` config without its penalty, new). `mixed` itself (lp 0.05) keeps c4b8548's
  rollouts and args.json, and its rows, lines, weights and dumps are pinned anew.

**`--loss_tokens` and `--adv_sign` in grpo.py** (the `loss` and `adv` knobs; opt-in, left out of args.json unless
given, so without them nothing changes: c4b8548's digests hold, and given at their defaults `all` / `both` they change
only args.json; `tests/thinking/test_grpo_mini_knobs.py`). line.sh refuses these arms unless the worktree's grpo.py has
`LOSS_TOKENS` and `ADV_SIGNS`.
- `--loss_tokens action|think` trains only the scored tokens of one phase. The think phase is the think block, its
  `</think>` included (`think_len`); the action phase is the rest. The loss's divisor counts the trained tokens alone,
  so the loss is a mean over them and each weighs more than in a full run (about 2x for `action`, whose tokens are about
  half of an attempt's). This is the "normalise over the selected tokens" choice. The clip and ratio statistics, and
  so the clip-low alarm, count those tokens too.
  - Unchanged: the entropy and the think-ban mass are over every scored token, so the ban alarm, the restore check
    and `--ban_stop` are unchanged (under `action` the think tokens get no gradient of their own, but the weights are
    shared). The KL anchor is a separate term on the anchor set's tokens. The unlikeliness ranks use each attempt's
    whole sequence.
  - The step penalty's −P falls on the trained phase's tokens of its draws.
  - Budget 0 has no think block, so `think` trains nothing there. A minibatch with no token of the phase takes no
    step, and so no KL-anchor step either (L3-mini's anchor is 0 anyway). `trained_tokens` is the phase's count, and
    the logged entropies stay per scored token (`LossStats.tokens_all`).
- `--adv_sign pos|neg` keeps one side of the outcome advantage, r minus the group mean after the unlikeliness
  deduction and the length terms, and zeroes the rest.
  - `pos` keeps the positive advantages (by sign). The deduction can make a solved attempt's advantage negative in a
    mostly solved group; `pos` drops it.
  - `neg` keeps the failed attempts' advantages (by outcome, 2026-10-18): every solved attempt's is zeroed, also one
    the deduction made negative, and a failed attempt's is kept up to 0. It is ≤ 0 anyway unless a length penalty
    takes long successes below 0 and the mean with them.
  - The `advpos` arm runs at `--rej_penalty 0`: otherwise the step penalty's −P would be the one negative signal left,
    at twice its relative weight. Both adv arms are L3-mini forks.
  - The zeroed items stay in the batch, their tokens in the divisor, so a kept token weighs exactly what it weighs in
    the full run. The update is about half as large on average, an lr-like effect that comes with the knob.
  - The step penalty is not an outcome advantage. Its −P stays on the redrawn draws and on the draw that ended a
    failed attempt (under `pos` that draw keeps −P alone, which is why `advpos` runs without it).
  - The KL anchor and the ban mass are unchanged. All-solved and all-failed groups are skipped as before.

**Temperature** (`t` knobs). grpo.py uses `--temperature` for sampling, the PPO ratio's old and new log-probs, the
unlikeliness ranks, the entropy and think-ban mass in the loss, and the ban restore check. The canary is always at
T = 1, against its T = 1 base. The KL anchor tempers both distributions, and in L3-mini it is off. The think-ban alarm
reads the tempered mass (what is sampled), unchanged. At T = 1.2 the full-scale pilot's mass rose from 4.2e-4 to 8.2e-4
by its last iteration (the alarm is at 1e-3), so `t13` may raise ban alarms and rollbacks, which the stage lines record.
The gradient on the logits scales as 1/T (1.43x at 0.7, 0.77x at 1.3), partly absorbed by the grad-norm clip at 1.

**Checkpoint reads of the full-scale L3 runs** (`ARM_ckreads`, for every finished GRPO L3 that started from another
arm's L2: the knob pilots, `ARM_mb`, `ARM_fbb`, …). There is no training and no preflight. Each distinct checkpoint is
read once:
- `ARM_ckreads`: `CKR_PANEL` (`l2test` k8, `l3test` k8, t1320 k4, t2130 k4, charles173 k16) at `CKR_ITS`, it10, it20 and
  it30 (`_ki78`: it10 to it70). Each is read from the rolling checkpoint there or, after an alarm iteration (which saves
  none), from the nearest earlier one. That substitution is logged, the reads are named after the checkpoint read
  (`ARM_itJ`), and `ARM_ckreads_sSEED.ckpts.txt` records each iteration and its file. No rolling checkpoint at or before
  an iteration skips its reads. Then `CKR_FINAL` (`l2test`, `l3test` k8) of the last checkpoint, under the arm's own
  names; the pilot's own reads hold its out-of-distribution ones.
- `ARM_l2_ckreads` (`grpo_think_vs_l2_ckreads`): a base arm's own last L2 checkpoint, the init every pilot of that seed
  shares, as `ARM_L2`; `s1g3_ckreads`: Stage-1 `S1g3_sSEED`, as `s1g3`. Both read `CKR_PANEL` and `MINI_READS` (8 reads:
  the two test splits are in both). That covers the "it0" of every pilot once per seed.
- HEAD goes to the job's own `<job>_sSEED.base_commit.txt`; the arm's training record stays. A rerun reads only what is
  missing. Refused: other arms, KNOB_FULL=1, no finished L3 (no last L2 checkpoint), nothing written.
- Holes in the existing runs (no rolling checkpoint at the iteration; the nearest earlier one read instead): `kp3` s0
  it10 → it5 and it30 → it25; `kkl0` s0 it30 is the post-rollback checkpoint (it25 plus one iteration). Every 5
  iterations is saved, so a missing one is an alarm iteration.

```
bash line.sh GPU 0 grpo_think_vs_mini          # L2-mini then L3-mini from S1g3_s0, then the reads
bash line.sh GPU 2 grpo_think_vs_mini_klp20    # the same with --len_penalty 0.2 in both stages
bash line.sh GPU 1 grpo_think_vs_mini_kkl1     # seed 1: L3-mini from grpo_think_vs_mini_s1_L2_it10, KL anchor 0.1
bash line.sh GPU 0 grpo_think_vs_mb_kkl0_ckreads
bash line.sh GPU 0 grpo_think_vs_l2_ckreads    # shared by every pilot of the seed
bash line.sh GPU 0 s1g3_ckreads
```

## Reads (after the last stage, on the same GPU; thinking at B64, no-think at B0; rows under `runs/eval/v2/`)

| set | k | arms |
|---|---|---|
| t1320 (986 valid; 16–20 = a `min_lines` filter of the rows) | 16 | plain, logical |
| textbook72 | 64 | plain, logical |
| charles173 (Charles's 2026-10-02 release: candidate_v0 87 + candidate_v1 72 + batch3 14) | 64 | plain, logical |
| hand_proved_14 (20–102-line hand proofs; `--max_steps 160`) | 64 | plain |
| t2130_n1000 (a fixed random 1,000 of test_21_30, seed 2130) | 8 | plain |

`ARM_fbb` adds the feedback arm on t1320 and charles173 (`--max_rej 2`, retries per step).

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
