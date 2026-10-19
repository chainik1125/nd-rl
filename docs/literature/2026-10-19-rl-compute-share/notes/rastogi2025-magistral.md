---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - rastogiMagistral2025
---

# Magistral: pure RL on Mistral Medium 3, with both compute terms undisclosed

Paper: [@rastogiMagistral2025] (proposed key)
Source: arXiv 2506.10910v1 (12 Jun 2025, the only version) and the Mistral Medium 3 announcement
(mistral.ai/news/mistral-medium-3). Both fetched 2026-10-19. Ledger rows: `A-MAG-*`, `A-YUE-MAG*`.

## Learnings

What the paper states:

- **Start checkpoint.** "We used Mistral Medium 3 Instruct [MistralAI, 2025] as the starting checkpoint for this run."
  This is an instruct model, not a pretrained base. It is multimodal.
- **Algorithm.** GRPO, and "We remove the KL penalty entirely." Zero-advantage groups are filtered out, and ε_high is
  0.26–0.28. Clip-Higher and the soft length penalty come from DAPO.
- **Schedule.** "Training was done in multiple stages with distinct hyper-parameters."
  - The batch is counted in completions and shrinks 8k → 4k → 2k.
  - The non-penalised length grows 16k → 24k → 32k.
  - There is one update per batch.
- **Data.** 38k math problems (filtered from 699k) and 35k code problems.
- **Results.** On AIME'24, pass@1/maj@64 goes from 26.8/43.4 to 73.6/90.0. The authors argue "increasing completion
  length is the main resource that improves the performance of the model".
- **Magistral Small.** Mistral Small 3 Instruct (24B) is SFT'd on Medium traces, then RL'd with batch 2048 and a 32k
  length. AIME'24 maj@64 falls from 90.0 (SFT) to 83.3 (SFT + RL) while pass@1 rises.

My interpretation:

- [interp.] **No RL/pretraining ratio is computable from primary sources.**
  - The RL side has no GPU type, count, hours, steps, group size or mean length.
  - The pretraining side has neither parameter count nor tokens. Yue notes "the model size is not disclosed", and
    Mistral says only that Medium 3 deploys on "four GPUs and above".
  - Any ratio needs grade C/D estimates for both terms.
- [interp.] **Yue's Magistral comparison is RL vs an instruct-tuned start, not vs a pretrained base.**
- [interp.] **Different metrics, compatible results.** Magistral's maj@64 jump and Yue's "little or no improvement at
  higher k" can both hold: majority voting rewards consistency, pass@k rewards coverage.

## Evidence and limitations

- **Off-policy degree is bounded but unmeasured.** Generation is asynchronous: "In the generators, weights are replaced
  mid-generation". n_async/n_batch is kept ≤ 2.
- **Dropped groups are uncounted.** Zero-advantage groups are generated and then discarded, and their fraction is not
  reported.
- **Max length is only partly given.** Only l_max − l_cache is stated, not l_max. Yue queried the API at 40k context
  (AIME24/25, k ≤ 1024).

## Connections and questions

- Related: [DAPO note](yu2025-dapo.md) and [SimpleRL-Zoo note](zeng2025-simplerl-zoo.md). Also
  [@yueDoesReinforcementLearning2025] and [@guoDeepSeekR1IncentivizesReasoning2025] (Table 2 baselines).
- Open: does any later Mistral source give Medium 3's size or tokens? Not checked.
- Open: Small's RL-lowers-maj@64 result is a create-vs-elicit data point that was never tested with pass@k.
