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

## Model-chosen think budget (auto effort; branch `leon/grpo-v2-auto`, 2026-10-11)

Worktree `/scratch/cowifornia/nd-rl/think/nd-rl-v2auto`. Without the new options the GRPO runs are e027354's:
rollouts, metrics rows, log lines, weights and args hash identically on three tiny-model configurations
(`tests/thinking/test_grpo_auto.py::test_default_path_is_e027354`).

- **GRPO** (`grpo.py --auto_levels 0,64,256`, module docstring): at every step the effort token is sampled from the
  policy restricted to `<e0> <e64> <e256>`, then the step is decoded at that level.
  - The effort token is trained as an action. It is a one-token item with its attempt's advantage, scored with the
    restricted log-softmax, the distribution it was sampled from. It enters the PPO ratio and clip, the loss
    normaliser and the clip-low alarms.
  - The step penalty does not reach the effort token, and the think ban mass does not read it.
  - With the KL anchor, the level choice at the anchor's states is anchored too; each choice weighs one action
    position.
  - Plain arm, or the verifier stop (`--arm logical --max_rej 0`). Redraws are refused.
  - `--choice_lambda` is ei_auto's token cost (every attempt pays lambda per 1,000 sampled tokens). It is 0 in the
    arms below.
  - Logged per iteration: `level_share`, `choice_entropy`, the scored choices (`tokens_choice`, `entropy_choice`,
    `clip_low_choice`), `kl_anchor_choice` and `mass_choice`.
- **EI** (`ei_auto.py`, unchanged): it runs with the launcher's EI settings, with the plain arm and with the logical
  arm (accepted proofs only).
- **Reads** (`eval --budgets auto`): the plain and the logical arm. With a guided arm, a redraw picks its level again,
  and the without-replacement exclusion is per level.
- **Arms** (launcher candidate `/scratch/cowifornia/nd-rl/think/smoke_auto/line.sh.auto`):
  - `grpo_think_auto` and `grpo_think_vs_auto`: as `grpo_think` and `grpo_think_vs`, with `--auto_levels 0,64,256`
    instead of `--budgets 0,64,256`.
  - `ei_think_auto`: `ei_auto` at k 11/11/10.
  - All three read with `--budgets auto`. Only these arms use the auto worktree; the dry run reproduced every
    existing arm's commands byte for byte.
- **The Stage-1 start** (`smoke_auto/measure_e/first_step_levels.py`; 200 random L2 problems, first state):
  - In S1g3_s0, s1 and s2, the level choice is uniform: mean shares 0.333 / 0.333 / 0.333, entropy ln 3, and no
    state above 90% at T 1 or 0.8.
  - The effort tokens were inputs only in Stage 1. Their output rows ended up nearly identical (pairwise distance
    about 4e-3, against about 2 between other tokens' rows), so the choice starts state-independent.
  - GRPO-auto therefore starts with full exploration and no prior over levels.
