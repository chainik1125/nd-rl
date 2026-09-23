---
written_on: 2026-09-23
written_by: agent:claude
status: proposal, awaiting Leon's approval; nothing implemented
---

# Proposal: dashboards for monitoring RL runs

Requested by Dmitry in the 2026-09-20 group meeting (transcript, 49:38): "have a decent set of
dashboards that we monitor ... so that it's less of a black box", with the heuristic goal of being
able to say "the model stopped learning here, that's why". Owner: Leon. This note proposes
dashboards that consume the per-step metrics JSONL defined in the A/B harness spec
(`/scratch/cowifornia/nd-rl/ab/common/SPEC.md`, "Metrics schema"), mirrored to W&B project
`nd-rl-hillclimb` by the host-side `wandb_sync.py`.

## Summary

Every RL run in this project is one loop: sample proofs, verify, train, repeat. The questions a
person debugging a run asks are correspondingly few. **Has it started learning, and has it
stopped?** (frontier length and per-round solve rate vs. the frozen-control baseline: the
ignition-study arms that never ignited looked identical to igniters for the first three rounds).
**Is reward rising for the right reason?** (padding, QED-without-validity, one failure mode
dominating, or GRPO groups with no reward variance: run 4's mean reward rose 0.37 to 0.74 while
the fraction of informative groups fell 0.31 to 0.08). **Are we damaging the base model?** (held-out
cap-6 greedy fell from 0.87 to 0.56 in run 4 with no retained data). **Has exploration collapsed?**
(distinct proofs per solved target, entropy). **Is the GPU the bottleneck?** (samples/s, time split
by phase, memory; one GRPO update at 12 GB OOM-killed seven arms). The five dashboards below map
one to one onto those questions. Each panel states what healthy and unhealthy look like, using the
historical runs in `/scratch/cowifornia/nd-rl/external/nd-takehome/` as the reference.

## Survey of what existing stacks show

W&B has no RL-specific default layout: automatic workspaces make one line panel per logged key,
grouped by key prefix, so what a dashboard shows is decided by what the trainer logs. The RL stacks
converge on the same families (a subagent web survey, 2026-09-23; source URLs in the footnote):

| family | TRL `GRPOTrainer` | verl | Tinker cookbook | transfers to us? |
|---|---|---|---|---|
| reward mean / std | `reward`, `reward_std`, `rewards/{fn}/mean` | `critic/rewards/{mean,max,min}` | `env/all/reward/total` | yes: `accept_rate` is the mean; add `reward_std` when reward is shaped |
| wasted groups | `frac_reward_zero_std` | (dynamic sampling in DAPO) | `frac_all_good`, `frac_all_bad` | yes, central: for EI "all fail" means no winners to train on; we already had `frac_groups_with_variance` in run 4 |
| KL to reference | `kl` (if beta > 0) | `actor/kl_loss` | `kl_policy_base` | diagnostic only (no KL term in EI); `kl_to_init` is in the schema |
| sampler/learner mismatch | `sampling/importance_sampling_ratio/*` | `actor/ppo_kl` | `optim/kl_sample_train_v1` | should be about 0 in a single-process run: a bug detector, not a dial |
| clip fractions | `clip_ratio/{low,high}_mean` | `actor/pg_clipfrac` | none | no: only for multi-epoch PPO updates |
| entropy | `entropy` | `actor/entropy` | `optim/entropy` | yes, cheap (`entropy` in schema); the more readable collapse signal here is distinct proofs per solved target |
| response length | `completions/mean_length`, `completions/clipped_ratio` | `response_length/{mean,clip_ratio}` | `ac_tokens_per_turn` | yes: written vs pruned proof length, plus truncation at `max_new` |
| timing / throughput | `step_time`, `num_tokens` | `timing_s/gen`, `perf/throughput` | `time/sampling_time_*` | yes; at 3M params the verifier can be the bottleneck |
| critic / value | none | `critic/vf_explained_var` | none | no value model |
| sampled text | completions table (`log_completions`) | none | `logtree.json` per rollout | yes, every round, capped |

Two heuristics from the literature carry over directly: DAPO's dynamic sampling exists because
all-same-reward groups give zero advantage, so `frac_reward_zero_std` is the first thing to plot
for GRPO; and the entropy-mechanism result (reward is "bought" with entropy, and a flat reward
curve at near-zero entropy is the ceiling) says entropy and distinct-proof counts belong next to
the solve-rate curve. Tinker's console itself is not publicly documented beyond run and
checkpoint management; its cookbook writes `metrics.jsonl` and mirrors to W&B, which is the
design the SPEC already adopts.

## Dashboard designs

Field names below are the SPEC schema unless marked **add**. Effort is human-equivalent time
assuming the W&B sync already works and `metrics.jsonl` exists per run.

### D1. Learning pulse (is it learning, stalled, or saturated?)

| panel | x / y | healthy | unhealthy |
|---|---|---|---|
| Targets solved, cumulative and per round | `elapsed` (or cumulative `samples`) / `targets_solved_cum`, `targets_solved_round / n_prompts` | per-round rate climbs above the frozen control within 2-3 rounds (EI: 0.29 to 0.51 to 0.62) | per-round rate flat at the control's level: cumulative still creeps up from resampling alone (frozen control 1,431 to 1,691 over 16 rounds), which is why the per-round line matters |
| Solved by length | `round` / `solved_by_len["7"|"8"|"9"]`, stacked | 7 rises first, then 8, then 9 | length 9 stays at zero while 7 saturates (frontier stuck) |
| New distinct proofs per round | `round` / `new_proofs` | large early, decaying smoothly (22.8k, 28.4k, 17.9k, ...) | zero or near-zero while `accept_rate` is high: policy re-emits memorised proofs |
| Acceptance and QED | `round` / `accept_rate`, `qed_rate` | both rise together | `qed_rate` far above `accept_rate`: confident invalid proofs |
| Train loss, grad norm, lr | `step` / `loss`, `grad_norm`, `lr` | loss falls per round, grad norm bounded | loss spikes after a round with few new records; grad-norm clipping every step |
| Budget | `elapsed` / 600 s line | rounds finish with margin | sampling round 1 alone consumes half the budget |

Metrics: `sample.*`, `train.loss/grad_norm/lr`. **Add:** `samples_cum` (or the sync computes
it), and a per-pool `frozen_ref` overlay: one no-train run per target pool (about 5 minutes) whose
`targets_solved_round` curve is drawn as a dashed baseline.
Would have caught: the depth-3 ignition arms s5 and s7 (0 pattern proofs through round 8) and s4
(ignited at round 7) are indistinguishable from the control by round 3, which is when a human
would stop them or inject data (the round-4 intervention ignited 11 of 11). Ladder arm T4's flat
rounds 2-3 (544, 664 solved vs 1,020, 1,584 for T1) would have been visible within 20 minutes
instead of at analysis time. Effort: W&B saved workspace 1-2 h; JSONL renderer half a day.

### D2. Reward integrity (is reward rising for the right reason?)

| panel | x / y | healthy | unhealthy |
|---|---|---|---|
| Padding gap | `round` / `mean_written_len`, `mean_pruned_len` and their ratio | ratio matches the frozen control (padded fraction 0.47 in the 16-round control) | written grows while pruned is flat: the padded fraction of new transfer proofs rose 0.42 to 0.61 over the 16-round EI run `ei_abs_s0` + `_cont` (`log.md` 07:50); with any length-shaped reward this is the reward-hacking signature |
| Failure mix | `round` / top-5 `fail_reasons` as stacked fractions | mix shifts from "final formula is not the conclusion" toward specific rule checks as proofs gain structure | one reason grows to dominate late (e.g. "bad box cite" exploding means the model has learned to open boxes it cannot close) |
| Proof-length histograms | `round` / **add** `written_hist`, `pruned_hist` (as in `round_*.json`) | mass moves right one bin per few rounds | mass piles at `max_new` (truncation) or collapses to one length |
| GRPO group signal | `step` / **add** `reward_mean`, `reward_std`, `frac_groups_with_variance` | variance fraction 0.3-0.5 | run 4: 0.31 to 0.08, i.e. 92 % of groups contribute zero gradient (the DAPO dynamic-sampling failure); mean reward alone looked great |
| Sample viewer | table of 8 accepted and 8 rejected proofs per round, **add** `examples` (optional, capped) | proofs use premises, boxes close | trivial proofs of trivial targets, repeated proof text across targets |

Metrics: `sample.*`, additions above. Would have caught: run 4's wasted-group decay from step 50
onward; the padding drift; a reward-shaped run that games length. Effort: schema additions half a
day (the histogram and group fields are one-liners in `rl.py`; `examples` needs a size cap);
panels half a day.

### D3. Retention (are we damaging the base model?)

| panel | x / y | healthy | unhealthy |
|---|---|---|---|
| Held-out cap-6 greedy | `step` / `heldout_greedy`, horizontal line at `base - 0.03` | flat 0.948-0.959 (every ladder arm, 20k retained records) | 0.87 to 0.56 by the first round-equivalent (run 4, no retention, no KL); `ei_abs_s0`'s dip to 0.931 at round 4 |
| Held-out by length | `step` / **add** `heldout_greedy_by_len["2".."6"]` | lengths 5-6 lowest but stable | length-6 falls first: the model is forgetting exactly the proofs nearest the frontier |
| Drift | `step` / `kl_to_init`, `entropy` (optional in schema) | KL grows sub-linearly; entropy falls slowly | entropy to near zero (exploration collapse) or KL jump after a round with few records |
| Trade-off | scatter `targets_greedy` vs `heldout_greedy`, one point per eval | moves right without moving down | moves down-right |
| Mix composition | `step` / `n_records`, **add** `n_retain` | retained fraction as intended | RL records swamp the retained set as `new_proofs` accumulate |

Would have caught: run 4's degradation by step 50 (it was pre-registered as R4-E4 and only read
at the round boundary); a retention ratio that silently shrank as found proofs accumulated.
Effort: half a day including the by-length eval (the 1,000-record slice already exists; bucket
by `n_lines`).

### D4. Throughput and budget (is the GPU the bottleneck?)

| panel | x / y | healthy | unhealthy |
|---|---|---|---|
| Sampling throughput | `elapsed` / `samples_per_sec` | about 350/s at batch 384 (SPEC) | falls as proofs lengthen or a second process shares the GPU (ladder round 9: 1,436 s vs 946 s) |
| Time split | `round` / seconds in sample, train, eval, **add** `phase_secs` (or derive from `t` deltas) | sampling dominates, eval small | eval or verification wall time comparable to sampling |
| Memory | `step` / **add** `gpu_mem_gb` (max allocated) | headroom | the 12 GB GRPO update that OOM-killed seven arms at start |
| Cost of a proof | `round` / `samples / new_proofs` | rises slowly | jumps: the pool is saturated, move the target window |
| Projected finish | `elapsed` vs budget, rounds completed | finishes under 600 s with margin | on course for `status: timeout` (no score at all) |

Effort: half a day. Would have caught: the OOM before seven arms were queued; the resume-run
slowdown; runs that would time out under the 600 s harness clock.

### D5. Run comparison (which arm or config is winning?)

Overlay D1's cumulative-solved and D3's held-out curves across runs, grouped by `arm`, joined with
the dev score from `logs/scores.jsonl` (**add** to the `summary` record: `dev_score`, `by_len`).
W&B does this natively (group by, config table); the JSONL renderer would need a small leaderboard
table. Effort: 2 h in W&B, half a day self-hosted.

## Implementation surface

Two consumers exist and they cannot share one surface: humans on the host have W&B; the agents
inside the arm containers have only `metrics.jsonl` (no W&B key enters a container, per the SPEC
addendum). I therefore propose **both**, with W&B first because the sync makes it nearly free:

| surface | what | cost | notes |
|---|---|---|---|
| W&B saved workspace per dashboard + one Report | D1, D3, D4, D5 line panels; D2 histograms and tables | hours | the sync must flatten `fail_reasons` (list of pairs) to `fail/<reason>` and `solved_by_len` to `solved_by_len.7`, and set `step` as the x-axis; W&B free tier limits logged tables, so `examples` should be sampled per round, not per step |
| `code/tools/rl_dash.py`: reads `runs/*/metrics.jsonl`, writes one static HTML (Plotly) | D1-D4 for agents and for offline review | 1 day for D1-D4 | runs in the container; the same script can render PNGs for experiment summaries |

## Recommended first build and ordering

**First: D1 in W&B**, plus the `frozen_ref` baseline. It answers Dmitry's "stopped learning here"
question directly, needs no schema change, and the ignition and ladder evidence says it pays off
in the first hour of any multi-round run. Then D3 (one metric already logged; the run-4 failure
mode is the most expensive one to discover late), D2 (needs the schema additions, so it can be
built while the first runs go), D4, D5. The JSONL renderer follows D1 once its panels are agreed,
so the in-container agents see the same view.

## Decisions for Leon

1. Surface: W&B only, self-hosted only, or both (recommended both, W&B first).
2. Schema additions to accept: `samples_cum`, `written_hist`/`pruned_hist`, `reward_mean`,
   `reward_std`, `frac_groups_with_variance`, `heldout_greedy_by_len`, `n_retain`, `phase_secs`,
   `gpu_mem_gb`, `examples` (capped), `dev_score` in `summary`; and flattening `fail_reasons`.
3. Per-theorem tracking: log first-solve round per target (a side file `first_solved.jsonl`, not
   in `metrics.jsonl`) to enable ignition-by-class and "which targets flipped" panels; or not.
4. Eval cadence: `heldout_greedy` every round (EI) or every N steps (GRPO); it costs budget.
5. Whether to run one frozen-control reference per target pool for the D1 baseline.
6. Alerts: thresholds for `heldout_greedy` drop, `frac_groups_with_variance`, `samples_per_sec`,
   via `wandb.alert` or the renderer.

## What I did not do

Nothing was implemented: no W&B panels, workspaces or reports were created, no code was changed,
and the SPEC was not edited. The schema additions above are proposals for Leon to accept or
reject. The historical evidence cited is from the takehome fork's notes and `round_*.json` files;
run 4's per-step GRPO traces are not present locally (only its round summaries and the log), so
the per-step claims for it come from `log.md` and `run4.md`.

## Sources

- Meeting transcript: `docs/research_calls/group_meeting/group_meeting-20-09-2026.md` (49:38).
- Harness spec: `/scratch/cowifornia/nd-rl/ab/common/SPEC.md`.
- Historical runs: `/scratch/cowifornia/nd-rl/external/nd-takehome/{run4.md,ignition.md,ladder.md,log.md}`
  and `artifacts/{ei_abs_s0_all,frozen_abs_s0_all,ladder/la_*}/round_*.json`.
- Prior curves the group cares about: `experiment-summaries/2026-06-09-nd-bootstrapping-sprint/README.md`.
- Survey sources: TRL GRPO docs <https://huggingface.co/docs/trl/main/en/grpo_trainer>; verl
  `metric_utils.py` <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/metric_utils.py>;
  Tinker cookbook `rl/metrics.py` <https://github.com/thinking-machines-lab/tinker-cookbook>;
  W&B limits <https://docs.wandb.ai/guides/track/limits>; DAPO (arXiv 2503.14476); the entropy
  mechanism of RL for reasoning LLMs (arXiv 2505.22617). Metric names were read from current docs or
  source; the Tinker `env/*` names are from the subagent's memory and unverified.
