---
written_on: 2026-11-24
written_by: agent:claude
status: proposal 48, written at Dan's request 2026-11-24 as the behavioural counterpart to proposal 47 (`olmo-weights`), kept as its own proposal. Not approved or launched.
---

# Proposal 48: how much behaviour changes between checkpoints, next to how much the weights change (`olmo-kl`)

## The short version

[Proposal 47](2026-11-24-olmo-weight-trajectory.md) measures how far the weights move between consecutive Olmo 3
checkpoints, from pretraining to RL. Distance in weight space can mislead:
- a large change can do little, for example a rescaling that a normalisation layer undoes;
- a small change can redirect behaviour.

This proposal measures the behavioural side on the same checkpoints. For a fixed set of texts it asks how differently
two checkpoints predict the next token, and then compares that with the weight distance, interval by interval.

It answers questions 47 cannot:
- Is RL's small weight change also a small change in behaviour, or a large one?
- Which stages change behaviour on ordinary text, and which change it only on chat-formatted tasks?
- Does weight distance predict behavioural distance at all?

## Design

**Checkpoints:** a subset of Proposal 47's list. These intervals are coarser than 47's, so compute the weight
distance for each interval here in the same pass, while both checkpoints are loaded.
- **32B:** about 60 checkpoints:
  - 10 stage-1 pairs;
  - stage 2 every 4th, and the merge;
  - all of stage 3;
  - SFT every 2nd;
  - DPO;
  - RL every 100 steps.
- **7B:** about 90, at the same stages, plus the three RL-from-base runs every 200 steps.

**Fixed text sets,** teacher-forced, so each checkpoint scores the same tokens and nothing is generated:
1. **Plain text:** about 200 documents (≈ 100k tokens) of public web text published after Olmo 3's data cutoff, so it
   cannot be in the training data. This is the only set that suits every stage, base checkpoints included.
2. **Instruction following:** the 218 IFBench prompts with `s2300`'s answers, in the chat template (≈ 150k tokens).
3. **Math:** `olmo-think32b`'s 59 AIME problems with `s2300`'s solutions (≈ 150k tokens), in the chat template.

Sets 2 and 3 are off-distribution for checkpoints before SFT. Report them from SFT on, and for earlier checkpoints only
as context.

**Measures, per consecutive interval and from each stage's start:**
- mean per-token KL divergence in both directions, KL(a ‖ b) and KL(b ‖ a);
- top-1 agreement;
- each checkpoint's mean negative log-likelihood (NLL) on each set;
- the same split by position: thinking tokens vs answer tokens in sets 2 and 3.

**Analysis:**
- Plot behavioural distance on Proposal 47's x-axis, with the same colours and stage bands.
- Scatter behavioural distance against weight distance per interval, with a rank correlation within and across stages.
- Find the stages where the two disagree most: a large weight change with a small behavioural one, or the reverse.

## What I expect (guesses)

- **RL against SFT:** RL's behavioural change per step is larger, relative to its weight change, than SFT's or
  pretraining's: a small weight change with a large effect on chat-formatted text.
- **Where it shows:** on plain text, RL's change is small.
- **Weight vs behaviour:** weight distance predicts behavioural distance within a stage but poorly across stages.

## What it does not show

- **Whether the change is capability:** KL measures how much predictions shift, not whether the shift is a capability
  gain. Proposals 44 and 46 cover capability.
- **Generality:** the texts are fixed choices; another set could rank stages differently. One model family.

## Cost

Forward passes only, on GPU pods.
- **Load time dominates:** a 32B checkpoint takes about 3–5 minutes to load, and the ≈ 400k tokens of scoring per
  checkpoint take about 1–2 minutes on an H200.
- **32B:** about 60 checkpoints × 6 minutes ≈ 6 H200-hours, about $20–35 on secure or community pods.
- **7B:** about $5–10.

**Estimate ≈ $25–45; budget $60.** Re-cost at stage 0. It can reuse Proposal 47's checkpoint manifest and hash checks.
