---
written_on: 2026-09-29
written_by: agent:claude
---

# Test-time scaling round 2: a learned value head

**Holdout rule (Robbie, 2026-09-29).** The holdout half of `transfer.jsonl` is touched only to
score the final configurations (pass@k, or solve rate at a budget for search). Nothing trains on
it, nothing is tuned on it, and no search runs on it before that final scoring. Tuning happens on
the dev half. Test-time training on evaluation theorems (e.g. expert iteration on them) is ruled
out.

Round 1 found that search ranked by the policy's own log-prob (beam, best-first) is no better than
plain sampling at matched tokens, and that SMC with 0/1 validity weights only matches resampling.
The question is whether a learned estimate of "will this prefix lead to a proof" makes search beat
resampling at matched compute. No expert iteration; the policy stays frozen (c005).

## Data: rollouts on `rl_targets` only

- **Theorems:** `data/ladder/rl_targets.jsonl`, 4,495 theorems of reference length 7-14.
  - Checked 2026-09-29: 0 overlap with dev or holdout (by `key`).
  - Excluded, as a precaution: the 1,500 the checkpoint's own RL stage trained on (the harness
    samples them with the run's seed), so the head never learns from theorems the policy has proofs
    memorized for. That leaves 2,995 per seed. On a 20-theorem smoke test the unseen ones still
    succeeded 81% of the time per resample rollout, so rl_targets are easier than dev or holdout
    either way.
  - Split by theorem into 2,663 for training and 332 for validation (`value.py` holds out
    min(495, n / 9)).
- **Rollouts:** 8 per theorem, from c005 with the round-1 resample sampler (T 0.8, cap 10).
- **Labels:** a rollout passes through a chain of valid prefixes (the root, then after each
  accepted segment). Every prefix on the chain is labelled with that rollout's outcome: 1 if it
  ends in a proof that Lean and nd_verify both accept.
  - Averaged over rollouts, the label estimates V(prefix): the chance that resampling from this
    prefix finishes a proof (Monte Carlo labels, as in Math-Shepherd 2024).
  - Expected size: about 290k labelled prefixes per seed.

## Model

- **Features:** the frozen c005 trunk's final hidden state (after `ln_f`) at the prefix's last
  token.
- **Head:** MLP 384 → 256 → 1 with a sigmoid, trained with binary cross-entropy. The trunk is not
  fine-tuned.
- **Cost:** one value costs one forward pass over the prefix, counted with prefill tokens, not
  sampled tokens.
- **One head per seed checkpoint,** since each checkpoint's hidden states differ.
- **Checks on the 495 validation theorems:**
  - AUC and calibration;
  - within-theorem ranking: does V rank prefixes whose rollout succeeded above those that failed,
    among prefixes of the same depth?

## Search with V

All three reuse the round-1 engine; only the ranking or weighting changes.

| method | change from round 1 | knob |
| --- | --- | --- |
| smc-v (twisted SMC) | a valid child's weight is V(child) / V(parent) instead of 1; resampling is proportional to weight | N |
| bfs-v | frontier priority is V(child) instead of length-normalised log-prob | budget (anytime) |
| beam-v | keep the top B children by V instead of by log-prob | B |

- **Tuning** on the round-1 300-theorem dev subset, seed 0 only: head size and training steps,
  plus E for beam-v and bfs-v.
- **Final scoring** on holdout for the chosen configurations, 3 seeds, once. Compared with
  round-1 resample and SMC at matched sampled tokens, with value forward passes reported alongside.

## Cost guess

- Rollouts: about 24k resample rollouts per seed, 15-20 A40-minutes.
- Head training: minutes.
- Dev tuning: about 1 pod-hour.
- Final holdout: 3 pods for about 1 hour.
- Total around $5.
