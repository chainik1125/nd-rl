---
written_on: 2026-09-23
written_by: agent:claude
status: proposal, awaiting Leon's approval; nothing implemented
---

# Proposal: dashboards for monitoring RL runs

Dmitry asked in the 2026-09-20 group meeting (transcript, 49:38) for "a decent set of dashboards that we monitor ... so that it's less of a black box". His test is whether we can point at a curve and say "the model stopped learning here, and this is why". Leon owns this. The dashboards read the per-step `metrics.jsonl` that the A/B harness writes (`/scratch/cowifornia/nd-rl/ab/common/SPEC.md`, section "Metrics schema"). The host-side `wandb_sync.py` mirrors that file to the W&B project `nd-rl-hillclimb`.

The choice of dashboards rests on a survey of how other groups monitor RL runs. The survey is at `/scratch/cowifornia/nd-rl/ab/dash/RL_monitoring_survey.pdf`, with its Markdown source beside it. It covers the public reports (DeepSeek-R1, DAPO, Dr. GRPO, ProRL, Kimi, Qwen3, OLMo 3, Tülu 3, MiniMax) and the metric sets of verl, OpenRLHF, TRL, NeMo-RL and the Tinker cookbook. It describes each failure mode and gives a procedure for finding what limits a run. This note keeps only the parts needed to decide what to build.

## What the survey found

Almost every RL stack logs the same seven kinds of number. They are task reward, the share of prompt groups that carry any learning signal, response length and truncation, token entropy, drift from the starting model, time per phase, and the disagreement between the sampler and the trainer. Most well-known failures show up in these seven.

Five failures matter for our setting. The first is diversity loss, where pass@1 rises while pass@32 does not. That matters because our dev score is pass@32. The second is wasted groups, where every sample for a target gets the same reward. In run 4, mean reward rose from 0.37 to 0.74 while the share of groups with any reward variance fell from 0.31 to 0.08. The third is forgetting of short proofs. In run 4, held-out greedy accuracy on proofs of six lines or fewer fell from 0.87 to 0.56 without retained data. The fourth is a plateau that never ignites. Several arms in the ignition study found no pattern proofs for many rounds, and an intervention at round 4 started 11 of 11. The fifth is running out of the 600-second budget during sampling.

Several failures that dominate LLM reports do not apply to us. We have no learned reward model, and our verifier is exact, so reward hacking can only come from verifier bugs. Chain-of-thought monitoring, language mixing and mixture-of-experts routing do not apply either. Grokking is a supervised-learning effect. Our late ignition is an exploration problem, because the training reward is also near zero before it takes off.

## Finding the limiting factor

The first dashboard should answer one question, which is what is holding the run back. The survey gives five candidates, checked in order. A sampling-bound run spends most of each round sampling and verifying. An exploration-bound run almost never finds a correct proof at the frontier length. An exploitation-bound run finds proofs, but its per-round solve rate stays at the frozen control. A learning-bound run shows loss and gradient spikes, with held-out accuracy falling at the same step. A capacity-bound run looks healthy everywhere, but its solve rate and pass@k have stopped growing.

A frozen control is the same sampling loop with training switched off. Indexing every curve to round 1, or to the frozen control, shows which one went flat first. Table 3 of the survey lists the two or three metrics that tell each case apart and what to change in each case.

## What we log now, and what to add

The schema already logs a good deal. Each sampling round logs acceptance and QED rates, targets solved per round and in total, new distinct proofs, solved counts by length, written and pruned proof length, samples per second, the top failure reasons, and sampling and verification time. Each training step logs loss, learning rate, gradient norm, records, tokens and batch size. Each evaluation logs held-out greedy accuracy on a fixed slice of proofs of six lines or fewer. Entropy and KL to the starting model are optional fields that the baseline `rl.py` does not fill.

Table 1 lists the proposed additions in order of value. Most take a few lines in `rl.py`, because `judge_samples` already has each target's success count.

Table 1. Proposed schema additions.

| Field | What it is | Why |
|---|---|---|
| `frac_all_fail`, `frac_all_pass`, `frac_mixed` | Share of targets whose k samples all failed, all passed, or were mixed | This is the wasted-group signal. It separates the exploration limit from the capacity limit |
| `pass_at` for k = 1, 4, 32 | Unbiased pass@k from each target's success count | It shows diversity loss at no extra sampling cost |
| `distinct_per_solved` | Distinct proofs per solved target | It is our answer-level entropy, and it works for EI |
| `entropy` | Mean token entropy on sampled proofs | It gives early warning of collapse in GRPO |
| `heldout_greedy_by_len` | Held-out accuracy by proof length 2 to 6 | Forgetting shows first at the longest retained lengths |
| `len_accepted`, `len_rejected`, `trunc_rate` | Length of accepted and rejected samples, and the share that hit `max_new` | It checks length drift and truncation |
| `n_rl_records`, `n_retained` | Found-proof and retained records in each training mix | It is the exploitation check |
| `train_secs`, `gpu_mem_gb` | Fine-tune time and peak GPU memory | It completes the time split. It would have caught the 12 GB GRPO update that killed seven arms |
| `kl_to_init` | KL to the starting model on a fixed prompt set | It is the drift check, and it is cheap at 3.2M parameters |
| GRPO only, `reward_std`, `adv_abs_mean`, `clip_frac`, `ratio_max` | Group reward spread, advantage size and update-size statistics | These are the standard instability checks |
| `examples` | Eight accepted and eight rejected proofs per round | Reading samples catches what the numbers miss |
| `ref_logp_unsolved` (optional) | Mean log-probability of known proofs for a fixed set of unsolved targets | It should rise before ignition. It needs a non-sealed target set with known proofs |

The W&B sync also needs to flatten `fail_reasons` to `fail/<reason>` and `solved_by_len` to `solved_by_len.7` and so on, and to use `step` as the x-axis.

## Dashboards

The earlier draft of this note proposed five dashboards. Four survive in changed form and one is new. The throughput dashboard is folded into the first, because time per phase is one of the five limit checks.

1. The limiting-factor dashboard is the page to open first. It shows the per-round solve rate against the frozen control, the all-fail and mixed fractions, new proofs per round, the time split per round, and loss with gradient norm. All curves are indexed to round 1.
2. The diversity dashboard shows pass@1, pass@4 and pass@32 together, distinct proofs per solved target, and token entropy.
3. The reward-integrity dashboard shows written against pruned length, the failure-reason mix over rounds, accepted against rejected length with the truncation rate, and the sample table. It catches padding, verifier bugs and length drift. The padded share of new transfer proofs rose from 0.42 to 0.61 over the 16-round EI run `ei_abs_s0`.
4. The retention dashboard shows held-out accuracy overall and by length, KL to the starting model, and the ratio of RL to retained records.
5. The comparison dashboard overlays the limiting-factor and retention curves across arms, and joins them with the dev score from `logs/scores.jsonl`.

There are two places to show these. People on the host have W&B. The agents inside the arm containers have only `metrics.jsonl`, since no W&B key enters a container. I propose W&B first, because the sync already exists, and then a small script, `code/tools/rl_dash.py`, that renders the same panels to static HTML from `metrics.jsonl`. W&B saved workspaces for all five dashboards should take a few hours. The HTML renderer for the first four should take about a day.

## Recommendation

Build the limiting-factor dashboard first, in W&B, with one frozen-control run per target pool as its baseline. It needs only the three group fractions and `train_secs` added to the schema. The diversity dashboard should come second, because pass@k and distinct proofs come free from data `rl.py` already computes. Retention, reward integrity and comparison follow.

## Decisions for Leon

1. Which schema additions in Table 1 should we accept?
2. Should we run one frozen-control run per target pool (about five minutes each) as the baseline?
3. Should we build a non-sealed set of unsolved targets with known proofs for the ignition progress measure?
4. How often should held-out evaluation run within the 600-second budget, every round for EI or every N steps for GRPO?
5. Should we set alerts, for example on a drop in held-out accuracy, on the mixed-group fraction, or on samples per second?

## What was not done

Nothing is implemented. No W&B panels, workspaces or reports exist, no code was changed, and the SPEC was not edited. The historical numbers come from the take-home fork's notes and `round_*.json` files in `/scratch/cowifornia/nd-rl/external/nd-takehome/`. Run 4's per-step GRPO traces are not stored locally, so its per-step claims come from `log.md` and `run4.md`.

## Sources

- The meeting transcript is `docs/research_calls/group_meeting/group_meeting-20-09-2026.md` (49:38).
- The harness spec is `/scratch/cowifornia/nd-rl/ab/common/SPEC.md`.
- The historical runs are in `/scratch/cowifornia/nd-rl/external/nd-takehome/` (`run4.md`, `ignition.md`, `ladder.md`, `log.md`, and `artifacts/*/round_*.json`).
- The monitoring survey, with 37 references and URLs, is `/scratch/cowifornia/nd-rl/ab/dash/RL_monitoring_survey.pdf`. Its source is `RL_monitoring_survey.md`, and the figure and build scripts are in `survey/`.
