---
written_on: 2026-10-07
written_by: agent:claude
---

# Thinking GRPO collapse on hard12: masked sampling, full-vocabulary scoring

**Run:** `runs/restart2_hard12/grpo_T1v1e`. Code at `8644bf4` (branch `leon/thinking`), started 2026-10-07 01:15 UTC and stopped by me at iteration 92 (about 04:10 UTC). Log: `runs/logs/h_grpo_T1v1e.log`.

The audit was read-only, with CPU probes in the job's scratch directory. Its findings were checked against the code.

## What happened (observed)

| Iterations | Effort levels e16–e256 | e0 | Think tokens per attempt | Training metrics |
|---|---|---|---|---|
| 0–28 | 65–78% batch pass@1 | ~55% | 90–105 | clip fraction ~0.003, grad norm ~0.02 |
| 28–33 | no collapse yet | no collapse yet | no collapse yet | clip fraction 0.04–0.07, grad norm 0.2–1.1 |
| 34–37 | collapse to 5–27% | holds | falls to 11 | — |
| 39–56 | recover to ~65–72% | ~65–72% | ~0 | settle |
| 57 | — | — | — | grad norm 97.8, loss +0.17 |
| 58–61 | 0–6% | 24–55% | — | — |

- **After the second collapse:** by iteration 78 the run had settled at about 1 think token per attempt, so it is effectively a no-thinking policy.
- **Control:** the no-thinking GRPO run with the same recipe and learning rate (`grpo_T0`) stayed at a grad norm of 0.02–0.05 throughout.

## Cause (confirmed in code and by measurement)

**The sampler and the loss use different distributions.**
- The sampler, `sample.py` `decode_wave`, bans tokens inside a think block (`<eos>`, `<think>`, `<pad>`, the effort tokens) and in actions (the effort tokens). It then renormalises over what is left.
- The loss scores both old and new log-probs with a full-vocabulary softmax (`grpo.py` `_token_logp`).
- The logged entropy also uses the full vocabulary.

**Banned probability mass at think positions:**

| Checkpoint | Banned mass |
|---|---|
| Initial model (it0) | 4e-5 |
| Earlier run on a different pool, it39 | 2e-4 |
| This run, it39 | 0.91 |
| This run, it78 | 1.00 |

Almost all of it sits on `<eos>`.

**Consequences:**
- Sampled think tokens score log-probs of −17 to −28, against −1.4 to −6 under the distribution they were drawn from.
- The logged think entropy of "0.00" is an artefact: the sampler's entropy is 1.5–4.4 nats.
- The think text is word salad followed by an empty action. At it39, 65–77% of steps with a non-empty think block are rejected, against 2–3% with an empty think block.

**Interpretation (not proven).** With a full-vocabulary loss, a think token with negative advantage pushes probability onto every other token, `<eos>` included. `<eos>` is never sampled inside think, so it is never pushed back down, and the effect feeds itself. On hard12, 63–77% of think tokens carry a negative advantage, and think tokens make up 58% of scored tokens. That plausibly explains why this pool tipped over and the earlier pool did not. e0 is unaffected because it has no think positions.

**Second-order cause: the PPO ratio is unbounded when the advantage is negative.** The loss is min(rA, clip(r)A). When a sampled token's scored probability is about 1e-9, a small change to the `<eos>` logit makes r ≫ 1. That fits the grad-norm spikes (max 385 within an iteration) and the positive loss at it57.

**Ruled out as the primary cause:**
- 4 PPO updates at lr 1.5e-4. Emulated from it0, the minibatch clip fraction is 0.008–0.017 and the median |log r| is 0.02–0.04.
- Give-up or empty-action reward hacking. `<no>`, empty actions, rejected steps and Lean rejections are all full-weight failures with reward 0.
- Effort and prompt tokens: they are not scored.
- Text inserted by the environment: it is not scored.
- A temperature or autocast mismatch between old and new log-probs.

## What this invalidates

- **All thinking GRPO results** in `restart2_hard12` and `restart_20261006` were trained against the wrong distribution.
  - The bias grows with the share of negative-advantage think tokens.
  - The "thinking shrinks under RL" and "no-thinking GRPO improves faster" readings must be re-measured after the fix. They may be partly or entirely this bug.
- **Unaffected:**
  - No-thinking GRPO (`grpo_T0`): there are no bans without think and effort tokens.
  - Expert iteration: it fine-tunes on the target tokens, so it never scores banned tokens.
  - Stage-1 pretraining and all evals: these sample with the masked sampler and never use the GRPO loss.

## Fix (in progress, branch `leon/thinking-grpofix`)

- **Shared bans:** one ban helper used by the sampler and the loss, so scoring uses the masked distribution the sampler draws from.
  - Masked entropy.
  - A test that the scored log-probs equal the sampler's on every non-forced token.
  - NaN guards on masked positions.
- **Dual-clip** at 3A for A<0.
- **Alarms:**
  - Think banned mass above 1e-3.
  - Grad-norm max above 5× its running median.
  - Loss above 0.
  - Clip-low fraction above 0.01.
  - Every alarm also triggers a checkpoint save.
- **Saves:** rolling checkpoints every 5 iterations (keeping the last 4), plus the permanent ones every 39.

Checkpoints `grpo_T1v1e_it39` and `_it78` from this run are corrupted. Any relaunch starts from the Stage-1 initialisation.

## A separate problem: overlong masking rewards looping (no-thinking GRPO, observed 2026-10-07)

`grpo_T0` uses the same `--overlong mask` recipe and is unaffected by the scoring bug. Even so, its share of attempts ended by a cap rose steadily while batch pass@1 fell. Mostly the cap is `fail:step_cap`, i.e. 48 steps on proofs of at most 12 lines.

Averages over 10-iteration windows, from `h_grpo_T0.log`:

| Iterations | Batch pass@1 | Groups skipped | Attempts ended by a cap |
|---|---|---|---|
| 40–49 | 92.9% | 62% | 0.2% |
| 60–69 | 89.0% | 68% | 3.2% |
| 90–99 | 89.0% | 75% | 5.6% |
| 110–119 | 88.3% | 82% | 8.3% |
| 130 | 84.3% | 84% | 12.1% |

**Interpretation.**
- Under masking, an attempt that loops to the step cap gets no gradient and is left out of the baseline.
- A failure, by contrast, gets a negative advantage.
- So on the hard problems that still give signal, looping is strictly better for the policy than failing. The policy drifts toward it.
- DAPO's overlong filtering assumes a truncated answer might have been right. Here, reaching the step cap on a proof of at most 12 lines means the policy is looping.

**Proposal.** Use `--overlong fail` (cap = failure) in relaunches, and alarm when the share of capped attempts exceeds 5%.
