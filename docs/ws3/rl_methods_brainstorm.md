# RL methods that fit ws3: brainstorm

Written 2026-09-24. This is a menu, not a plan; `plan.md` still sets the order of work. Anything
picked from here needs its own `proposal.md`.

**What ws3 lets us vary:** the RL library, the algorithm, the RL hyperparameters, and which
allowed prompts are sampled and when. **Fixed:** the checkpoint, tokenizer, pre-training data,
checker, and the 72 textbook problems (`CLAUDE.md`). **Out of scope:** writing an algorithm
from scratch. Anything that needs more than a thin wrapper around a library is marked 🟡 (ask
Charles first) or 🔴 (out of scope as the plan stands).

## What about our setting shapes the choice

- **The model is tiny** (3.2M params, 99-token vocab, proofs under about 100 tokens), so
  rollouts are cheap and the optimizer step is cheap. We can afford large groups, many samples
  per prompt, and many seeds. vLLM and LoRA buy us little.
- **Reward is binary and sparse at the frontier.** Robbie's round-1 sample accuracy for the new
  checkpoint on `rl_targets` (lengths 7 to 14) is about 3 to 5%. At 4% with group 16, about half
  the groups are all-zero and give GRPO no gradient. Most choices below are about getting signal
  out of rare successes.
- **The checker is exact and works line by line.** `nd_verify` can say which lines are valid, so
  dense or process rewards cost nothing to compute. They are also easy to hack.
- **Expert iteration is the bar to beat.** Robbie's harness turns this checkpoint's recipe from
  about 4% round-1 accuracy into 254 to 291 of 1,108 dev theorems with 4 rounds of plain
  rejection-sampling fine-tuning. Any on-policy method we pick has to be compared against EI
  run at the same number of samples.

## A. Policy gradient with group baselines (TRL `GRPOTrainer` switches)

All of these are config changes in TRL 1.13 (`GRPOConfig`), so each costs one run to try.

| Method | Switch | Why it might matter here |
| --- | --- | --- |
| GRPO, length-normalized | `loss_type="grpo"` | The textbook form; has a known length bias. Useful only as a reference point. |
| DAPO token aggregation | `loss_type="dapo"` (the TRL 1.13 default) | No length bias. Our current default. |
| Dr. GRPO | `loss_type="dr_grpo"`, `scale_rewards="none"` | Drops the std division that inflates advantages on near-all-zero groups, which are most of ours. |
| Clip-higher | `epsilon_high` > `epsilon` | Lets rare positive samples move further; aimed at early entropy collapse. |
| Overlong masking | `mask_truncated_completions=True` | Stops cap-hitting junk from counting as ordinary failures. |
| CISPO | `loss_type="cispo"` | Clips importance weights instead of the update, so rare tokens in successful proofs still get gradient. |
| GSPO, LUSPO | `importance_sampling_level="sequence"`, `loss_type="luspo"` | Sequence-level ratios. Matters more with several updates per batch (`num_iterations` > 1). |
| SAPO, VESPO | `loss_type="sapo"` / `"vespo"` | Soft gating instead of hard clipping. New; worth one run each at most. |
| GMPO | `trl.experimental.gmpo` | Geometric-mean objective, meant to be more stable. Experimental module. |
| RLOO | `RLOOTrainer` (already wired: `train_rloo.py`) | Leave-one-out baseline. The PPO-family alternative in the library comparison. |
| Batch-level reward scaling | `scale_rewards="batch"` | Middle ground between group and none. |
| Multiple updates per batch | `num_iterations`, `steps_per_generation` | Reuses each batch of rollouts for more updates; cheap rollouts make this less needed. |

**Not in TRL 1.13:** PPO with a learned critic, REINFORCE++, ReMax, and PRIME. They are in verl
(and PPO is in OpenRLHF). Try them only through the plan's fallback rule: record what failed in
TRL first.

### Why PPO with a critic is not on the list

1. **TRL 1.13 doesn't have it.** The installed package has no `PPOTrainer` or `PPOConfig`, not
   even under `trl.experimental` (checked 2026-09-24). Getting PPO means pinning an old TRL
   release or moving to verl or OpenRLHF. Pinning old TRL swaps the library under every other
   run. Moving library is a switch away from TRL, which `CLAUDE.md` says to ask about first and
   the plan allows only after a recorded TRL failure. We have no such failure yet.
2. **Wiring cost is high for our model.** Old `PPOTrainer` and verl expect an HF causal LM
   they can add a value head to. Our `GPT` from `fork_pin/` isn't one. We would need a value
   head wrapper, a choice between sharing the trunk or running a separate critic, and, in verl,
   custom-model registration plus Ray and FSDP. That is more than the thin wrapper this
   workstream allows.
3. **The critic mostly does what a group baseline already does, and does it worse here.** With
   one binary reward at the end and short proofs (under about 100 tokens), PPO's advantage is
   roughly `R - V(prefix)`. GRPO and RLOO replace `V` with the mean reward of other samples on the
   same prompt. Rollouts are cheap for us, so we can use groups of 32 to 64 and get a good
   per-prompt baseline with no model to train. A critic has to learn P(success | prefix)
   from data where only 3 to 5% of rollouts succeed. Early on it will be poorly fit, and a
   poor critic adds bias as well as noise to every advantage.
4. **More knobs, one week.** A critic adds value lr, `vf_coef`, value clipping, GAE lambda, a
   warm-up schedule, and an init choice. Each is a way for a run to fail for reasons unrelated
   to the question "which library gives the most lift".
5. **Published results point the same way.** Critic-free methods (GRPO in DeepSeekMath, RLOO in
   Ahmadian et al. 2024, "Back to Basics") match or beat PPO on LLM RL with verifiable rewards,
   at lower cost. RLOO is our PPO-family entry for the library comparison (shortlist item 5).

**What a critic could give that we would lose.** Two things, both worth naming:

- **Signal from all-zero groups.** When every sample in a group fails, the group baseline gives
  zero advantage and no gradient. A critic that predicts `V > 0` still pushes those samples down.
  At 4% accuracy about half our groups are all-zero, so this is the real argument for PPO.
- **Credit per line.** A good critic could learn which line sinks a proof. But the checker
  already tells us which lines are valid (see F), so a critic is a roundabout way to get this.

**When to revisit.** If GRPO stalls with most groups still all-zero after the sparse-reward pack
(shortlist item 3), the curriculum (item 4), and the replay buffer (item 6), record that as the TRL
failure and ask Charles about a verl PPO run.

## B. Rejection sampling and expert iteration

| Method | How | Notes |
| --- | --- | --- |
| Expert iteration / RFT / STaR | sample k, keep verified, `SFTTrainer`, repeat | startup.md exp-05. Replicates Robbie's harness inside TRL. Needed as the comparison baseline in any case. |
| ReST-EM style | same, but fine-tune from the base checkpoint each round | Less drift than continuing from the previous round. |
| EI with a replay mix | mix verified RL proofs with pre-training records | What Robbie does (`mix_rl_records`). Protects lengths 2 to 6. |
| Shortest-proof EI | keep only the shortest verified proof per prompt | Pushes toward minimal proofs, and possibly toward generalising to longer lengths. |
| Self-distillation (SSD) | `trl.experimental.ssd` | Fine-tune on the frozen model's own samples at a chosen temperature, with no filter. A useful control for how much the filter matters. |
| Hindsight relabelling 🟡 | when an attempt derives valid lines but not the goal, make the last valid line the new goal and keep it as a verified proof | Turns failures into training data, the biggest lever against sparse reward. Risks: the new sequents are generated data (a dataset change?), and each needs a renaming-class leak check against the 72 textbook problems and every evaluation set. |

## C. Preference methods using the checker

| Method | How | Notes |
| --- | --- | --- |
| Online DPO | `trl.experimental.online_dpo` with the checker as judge | Needs a pass and a fail in each pair, the same scarcity problem as GRPO. |
| Offline DPO / KTO / BCO | pairs or unpaired labels from EI samples | KTO and BCO accept unpaired failures, so every sample counts. Cheap to try after an EI round. |
| Nash-MD, XPO | `trl.experimental` | Built for learned preference models; weak fit for a binary checker. Low priority. |

## D. Exploration and sparse-reward fixes

- **Bigger groups on hard prompts.** 32 to 64 samples per prompt at lengths 9+. Cheap for us.
- **Dynamic sampling (DAPO).** Drop all-zero and all-one groups and top up with more prompts.
  This is a prompt filter in our driver, not a TRL flag.
- **Temperature and truncation.** Temperature 1.0 to 1.3, `min_p`, `top_k`. Sweep in T5.
- **Entropy bonus.** `entropy_coef`, or `use_adaptive_entropy` with `entropy_target`.
- **High-entropy tokens only.** `top_entropy_quantile` < 1 updates only the most uncertain tokens,
  which are probably the rule choices.
- **Replay buffer of rare successes.** `trl.experimental.grpo_with_replay_buffer` keeps
  high-advantage groups and replays them. A direct fix for "one success in 64 samples".
- **Pass@k objective** 🟡. Reward that credits a group for any success, or advantages based on
  pass@k. Needs a custom advantage transform beyond a reward function.
- **Unlikeliness bonus** 🟡. Extra reward for correct proofs the model assigns low probability,
  to stop it collapsing onto one proof per prompt. Doable as a reward function that reads log-probs.

## E. Curriculum and prompt selection

The allowed pools are `train` (lengths 2 to 6) and `rl_targets` (7 to 14). Never `heldout`,
`transfer`, the 72 textbook problems, or `validation_36`.

- **Length ladder.** Unlock length L+1 once pass rate at L passes a threshold (Robbie's `lstar`).
- **Adaptive by group variance.** Sample lengths in proportion to reward variance, re-measured
  every N steps (startup.md exp-03).
- **Aim for about 50% success per prompt.** Track each prompt's success rate and sample prompts
  in the middle band.
- **Replay easy lengths.** Keep 10 to 20% of prompts at lengths 3 to 6 to guard against forgetting.
- **Drop solved prompts.** Once a prompt is solved every time, stop sampling it.

## F. Reward design (each one a separate, labelled experiment, per plan T3)

- **Invalid vs wrong.** Score unparseable output below a well-formed proof of the wrong goal
  (for example -0.1, 0, 1).
- **Partial credit by valid lines** 🟡. The fraction of lines the checker accepts. Dense, but
  easy to hack by padding with valid irrelevant lines.
- **Progress toward the goal** 🟡. Credit for deriving a subformula of the goal. Needs a
  hand-built heuristic; high risk of hacking.
- **Length.** A small bonus for shorter verified proofs, or a penalty per line past the reference
  length.
- **Process reward model** 🟡. `trl.experimental.prm`, with labels straight from the checker per
  line. It's a learned model, though, so why not use the checker directly? It would only help if
  it generalises to lines the checker can't judge in isolation (it can judge them all).

## G. Regularisation and reference model

- **KL:** `beta` at 0, 0.001, 0.01, or 0.05. With `beta=0` there's no reference model, which saves
  memory and time.
- **KL estimator:** `use_bias_correction_kl`.
- **Moving reference:** `sync_ref_model` with `ref_model_mixup_alpha` (TR-DPO style), or a
  BEMA reference (`trl.experimental.bema_for_ref_model`). The trust region follows the policy
  instead of anchoring to pre-training.
- **Learning rate:** at 3M params the right lr is probably much higher than LLM defaults
  (exp-00 at 1e-5 barely moved, KL 0.0017). Sweep 1e-5 to 1e-3.
- **Optimizer:** AdamW by default. Muon for RL is untested here, and changing it is an RL
  hyperparameter, not a pre-training change.

## H. Search at training time

- **Beam search or best-of-n rollouts.** HF `generate` supports beams. Generate with search,
  train on verified results (EI with a stronger sampler).
- **Checker-guided line-by-line rejection** 🟡. Generate one line, check it, resample if invalid.
  Needs a custom generation loop.
- **MCTS / AlphaZero-style** 🔴. Writing a search algorithm is out of scope.

## I. Distillation (confounds the lift question)

- **On-policy distillation** from Robbie's `039_*/final.pt` (the post-EI model), via
  `trl.experimental.gkd`, `iw_opd`, or `gold` 🟡. Probably a strong result, but it imports
  another pipeline's RL, so it answers "how good can this model get", not "which library gives
  the most lift". Report separately if run at all.

## J. Other libraries

| Library | What it adds over TRL | Cost to switch |
| --- | --- | --- |
| verl | PPO with critic, REINFORCE++, ReMax, PRIME, DAPO recipe, fast rollouts | High: Ray and FSDP setup, custom-model registration for our `GPT` |
| OpenRLHF | PPO, REINFORCE++, RLOO | Similar to verl |
| open-instruct | GRPO in the Olmo stack | Medium; built around HF models, vLLM-first |
| Tinker | Managed LoRA RL on open-weight models | Can't load our checkpoint; a reference number only (plan T6) |

For a 3M-parameter model the main thing these add is algorithms, not speed. So switching needs a
specific algorithm TRL lacks, plus a recorded TRL failure.

## K. Literature scan: arXiv, ICML, NeurIPS (2026-09-24)

A web scan of RL with verifiable rewards (RLVR) from 2025 and 2026, filtered for our setting.
Venues were checked against conference pages where possible, and "arXiv" means no acceptance was
found. Fit: **cfg** is a TRL 1.13 config switch, **wrap** is a thin wrapper (a reward function,
advantage transform, or prompt sampler), and **custom** needs its own algorithm code.

**How much fixing dead groups is worth.** At p = 0.04 the chance a group is all-zero is
0.96^G: 0.52 at G=16, 0.27 at G=32, 0.07 at G=64. Most papers below attack this.

### K1. Does RL beat rejection sampling?

| Paper | Venue | Finding | For us |
| --- | --- | --- | --- |
| Yue et al., "Does RL Really Incentivize Reasoning Beyond the Base Model?" | NeurIPS 2025 | RLVR mostly sharpens sampling; base pass@k catches up at large k | Report pass@k up to 256, not only pass@1, against base and EI |
| RAFT++ / Reinforce-Rej ([2504.11343](https://arxiv.org/abs/2504.11343)) | arXiv | Rejection sampling roughly matches GRPO; most of GRPO's gain is from dropping all-wrong prompts | Strengthens EI as the bar and zero-group filtering as the key knob |
| Interplay of pre-/mid-training and RL ([2512.07783](https://arxiv.org/abs/2512.07783)) | ICML 2026 spotlight (per authors) | Small models on synthetic tasks: RL helps only on prompts at the edge of competence | Closest match to our setup; supports a curriculum at the frontier |
| From f(x), g(x) to f(g(x)) ([2509.25123](https://arxiv.org/abs/2509.25123)) | ICLR 2026 | RL learns new compositions of skills from pre-training | Argues RL can help on longer, multi-rule proofs |
| ProRL | NeurIPS 2025 | Long runs with KL and periodic reference resets reach new strategies | cfg (`beta`, `sync_ref_model`) plus a wrapper for resets |
| Invisible Leash ([2507.14843](https://arxiv.org/abs/2507.14843)) | arXiv | RLVR stays in base support; answer diversity drops | Track distinct verified proofs per prompt |

### K2. All-zero groups and sparse reward

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| BroRL | ICML 2026 | Scale rollouts per prompt from 16 to 512; gains keep going after step-scaling saturates | cfg (`num_generations`) |
| DAPO | NeurIPS 2025 | Clip-higher, dynamic sampling, token-level loss, overlong penalty | cfg, plus wrap for refilling batches |
| MaxRL ([2602.02710](https://arxiv.org/abs/2602.02710)) | ICML 2026 oral | Optimises log pass rate: divide the advantage by the group mean reward, which puts weight on low-pass prompts | wrap (one-line advantage transform) |
| AVSPO; NGRPO ([2509.18851](https://arxiv.org/abs/2509.18851)) | ICML 2026; arXiv | A virtual max-reward sample, so all-wrong groups get a negative advantage | wrap |
| BREAD | NeurIPS 2025 | When all rollouts fail, prefix a short piece of an expert proof | custom; prefixes only from allowed training prompts |
| QuestA ([2507.13266](https://arxiv.org/abs/2507.13266)); Scaf-GRPO ([2510.19807](https://arxiv.org/abs/2510.19807)) | ICLR 2026; arXiv | Partial solutions or tiered hints on hard prompts | wrap; same data caveat, and possibly a dataset change (ask) |

### K3. Curriculum and prompt selection

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| Online Difficulty Filtering | EACL 2026 | Improvement is bounded by the variance of per-prompt pass rates, so sample prompts near 0.5 | wrap (sampler) |
| GRESO | NeurIPS 2025 | Skip prompts whose recent groups had zero variance, with an exploration floor | wrap (sampler) |
| Prompt Replay ([2603.21177](https://arxiv.org/abs/2603.21177)) | arXiv | Re-serve prompts with pass rate near 0.5; only the prompt is reused, so it stays on-policy | wrap (sampler) |

### K4. Exploration and pass@k objectives

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| PKPO | NeurIPS 2025 | Unbiased reward transform that optimises pass@k | wrap (group-level reward) |
| Pass@k Training ([2508.10751](https://arxiv.org/abs/2508.10751)) | arXiv | Analytic pass@k advantage | wrap |
| Entropy mechanism: Clip-Cov / KL-Cov ([2505.22617](https://arxiv.org/abs/2505.22617)) | arXiv | Entropy collapse comes from the covariance between log-prob and advantage; clip or add KL on high-covariance tokens | custom loss |
| Beyond the 80/20 rule | NeurIPS 2025 | Update only the top 20% of tokens by entropy | cfg (`top_entropy_quantile`); unclear benefit on 100-token proofs |
| Outcome-based exploration ([2509.06941](https://arxiv.org/abs/2509.06941)) | arXiv | UCB bonus for rarely seen outcomes plus a repetition penalty within the batch | wrap; the "outcome" could be a canonicalised proof |
| Entropy advantage shaping ([2506.14758](https://arxiv.org/abs/2506.14758)) | AAAI (per agent) | Add a clipped, detached entropy term to the advantage | wrap |

### K5. Negative samples and rank bias

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| NSR / W-REINFORCE | NeurIPS 2025 | Training on negatives alone improves the whole pass@k curve; down-weight positives (λ ≈ 0.1) | wrap |
| Rewarding the Unlikely | EMNLP 2025 | **On formal theorem proving:** GRPO over-reinforces already-likely proofs; a bonus for rare correct proofs plus fewer PPO epochs improves pass@N | wrap (reward) plus cfg (`num_iterations`) |

### K6. Off-policy and replay

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| ExGRPO ([2510.02245](https://arxiv.org/abs/2510.02245)); AR3PO; RLEP | ICLR 2026; arXiv | Replay past correct rollouts into online groups; AR3PO swaps one stored success into an all-fail group | partly cfg (`grpo_with_replay_buffer`), otherwise wrap |
| LUFFY | NeurIPS 2025 | Mix off-policy correct traces into groups with importance-sampling shaping | custom |
| ScaleRL ([2510.13786](https://arxiv.org/abs/2510.13786)) | arXiv | 400k GPU-hour recipe: CISPO, zero-variance filtering, no resampling of solved prompts, batch-level advantage normalisation | cfg (`cispo`, `scale_rewards="batch"`) plus wrap |
| Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)) | COLM 2025 | Drop the length and std normalisation | cfg |

### K7. Credit per step (the checker marks valid lines)

| Method | Venue | Mechanism | Fit |
| --- | --- | --- | --- |
| VinePPO | ICML 2025 | Value of each step from Monte Carlo rollouts restarted there; cheap when rollouts are cheap | custom |
| Rewarding Progress (PAVs) | ICLR 2025 | Step reward = change in success probability across the step | custom |
| TreeRL | ACL 2025 | Branch at high-uncertainty tokens; advantages from the tree | custom |

### K8. Prover recipes

DeepSeek-Prover-V2 ([2504.21801](https://arxiv.org/abs/2504.21801)): GRPO with a binary checker
reward after subgoal SFT. Goedel-Prover-V2 (ICLR 2026): synthetic tasks of rising difficulty,
self-correction from checker feedback, and model averaging to recover diversity. Kimina-Prover
RL: an open GRPO pipeline in verl. Absolute Zero (NeurIPS 2025): self-proposed tasks, out of
scope because prompts must come from the allowed pools.

## L. Tinker scan (2026-09-24)

**It can't run our model.** Tinker trains only LoRA on its listed open-weight models. It has no
custom-weight upload and no full fine-tuning. Its smallest model is Qwen3.5-4B, and its cheapest
dense model is Qwen3-8B. Built-in losses are `importance_sampling`, `ppo`, `cispo` and `dro`;
`forward_backward_custom` takes any other loss. The cookbook has no proof, Lean or curriculum
recipe. No new RL posts on the Thinking Machines blog in 2026.

**Findings that transfer:**

- **Capacity isn't our limit** ("LoRA Without Regret"). Policy gradient learns about 1 bit per
  episode, and rank-1 LoRA matches full fine-tuning for RL. Our 3.2M params are plenty. Lift
  depends on how often a group gives non-zero signal.
- **Cookbook defaults:** mean-centred advantage with no std division (per Dr. GRPO), KL 0,
  temperature 1.0, group 16 to 64, Adam betas (0.9, 0.95). They log `frac_degenerate`, the share of
  groups where every sample got the same reward. Their learning rates (1e-5 to 1e-4) are for LoRA
  on 8B models and don't transfer to us.
- **CISPO clip:** the default is one-sided (0, 4). The docs warn that a positive lower clip lets
  sampler/trainer KL grow out of control off-policy.
- **Sampler/trainer mismatch** ("Defeating Nondeterminism"): uncorrected mismatch collapsed reward
  in their run. Log it every step, and if we use vLLM, turn on
  `vllm_importance_sampling_correction` (in TRL 1.13).
- **Logging worth copying:** `kl_sample_train`, entropy, a rollout dump for a few groups each
  iteration, and `config.json` plus `code.diff` saved with each run.
- **On-policy distillation** is 50 to 100 times more compute-efficient than RL, but needs a
  teacher, so it confounds the lift question (see I).

**A reference run (plan T6), if Charles approves the spend.** Put our proof syntax and a few worked
examples in the prompt for Qwen3-8B or Qwen3.5-9B-Base. Measure sampling-only accuracy on
training-pool prompts with our checker first, for a few dollars. Run RL only if base accuracy is
a few percent or more. The agent's rough estimate, not checked against a bill, is about $0.35 a
step with 64 groups of 16, so $35 to $100 for 100 to 200 steps. It measures an 8B model's RL
ceiling, not lift over our pre-training.

## What K and L change in the shortlist

- **Item 1 (GRPO baseline):** use G=64 from the start, since rollouts are cheap. Log the fraction
  of all-zero groups each step. Start with `scale_rewards="none"`.
- **Item 3 (sparse-reward pack):** add **MaxRL** and **W-REINFORCE** (optionally with a virtual
  positive per group) as advantage transforms. Both are one-line changes aimed at our exact case,
  binary reward with a low pass rate.
- **Item 4 (curriculum):** use a p(1−p) prompt sampler with a skip rule like GRESO's. It is well
  supported (EACL 2026, NeurIPS 2025, and the edge-of-competence result from ICML 2026).
- **Item 6 (replay):** AR3PO-style: swap one stored verified success into an all-fail group. It
  is effectively a GRPO and EI hybrid.
- **New item: Rewarding the Unlikely** (unlikeliness bonus, `num_iterations=1`). It was tested
  on formal proofs and targets the loss of pass@k.
- **Evaluation:** add pass@k for k = 1 to 256 against base and EI (Yue et al.). The 72 textbook
  problems stay evaluation only.
- **New questions for Charles:** are hint or prefix methods (BREAD, QuestA) a dataset change,
  even when the hints come from allowed training proofs? Is the Tinker T6 spend worth an 8B
  number that says nothing about our lift?

## Suggested shortlist, in order

1. **GRPO, TRL defaults** (`dapo` loss), with a lr sweep up to 1e-3. The baseline everything is
   compared against.
2. **EI in TRL** at matched sample budget (exp-05). Tells us whether on-policy RL beats what
   Robbie already gets.
3. **Sparse-reward pack on top of 1:** group 32+, dynamic sampling, `dr_grpo` with no reward
   scaling, clip-higher, overlong masking. One run with all of them, then drop them one at a time.
4. **Adaptive curriculum** (exp-03).
5. **RLOO** as the second algorithm for the library comparison.
6. **Replay buffer** (`grpo_with_replay_buffer`) for rare successes.
7. **Wildcard: hindsight relabelling**, if Charles rules it in scope.

## Questions for Charles

- Is hindsight relabelling a dataset change (out of scope), or RL data augmentation (in scope)?
- Is shaped reward (partial credit, progress) allowed as a labelled experiment, or binary only
  this week?
- Is distillation from Robbie's `final.pt` worth a side run, reported separately?
