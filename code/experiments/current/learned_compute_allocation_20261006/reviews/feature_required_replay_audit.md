---
written_on: 2026-10-06
written_by: agent:codex
status: development_review
---

# Required replay work for the public feature critic

## Finding

The exact existing feature classifier consumes mean sampling log probability, mean base-model log probability, and mean base entropy produced by `selection/collect.py::likelihood_batch`. That function performs an additional BF16 full forward on every state prefix plus its entire sampled action. This replay is required inference work for that classifier. It cannot be excluded as diagnostic overhead in a total-compute comparison. The text-only BCE critic does not require these features or this replay.

Fixed one needs neither a selector nor replay. A staged feature threshold must compute its score, including replay, for every reached candidate. Fixed counts of at least two need the scored candidates' replay. Features obtained during offline pool collection are not free at deployment.

## Independent development accounting

Only public main-development candidates and collection summary counters were opened. Across 2,560 candidates there are 35,315 raw actions, 6,237,227 cumulative state-prefix tokens, and 802,096 generated tokens. Let these quantities be A, P, and G.

| Quantity | Forward input tokens |
|---|---:|
| Ideal cached generation proxy, P + G − A | 7,004,008 |
| Existing replay, unpadded P + G | 7,039,323 |
| Executed batched generation | 14,971,307 |
| Executed batched replay | 15,636,147 |

Executed replay is 104.44% of executed generation forward tokens. The executed totals are sums of `main_dev_forward/summary.json` and `main_dev_ore2/summary.json`, using the collector's model pre-forward hooks. They describe **full-pool collection with its actual batches**, not the execution cost of a sparse stopping policy. The public candidate totals come from `pools/main_dev/public.jsonl`.

The existing replay includes the final emitted token even though its output logit is unused. Its batching pads each action batch to the maximum prefix-plus-output length. Ideal generation P + G − A assumes prefix prefill plus exactly the cached decodes required for the sampled tokens. The upstream fast sampler checks completion every 16 steps, keeps completed rows until compaction, and can perform an unused final decode at the action cap. Therefore that generation formula is a lower-bound proxy, not its executed token count. Neither forward-token measure is itself a FLOP or wall-time measure: full replay and cached generation also differ in attention work and batching efficiency.

## Fair comparison

Use candidate inclusion probabilities to sum required unpadded generation and replay proxies, explicitly labeled as such. Freeze development mixtures under the chosen work definition and report residual held-out work mismatch. For measured total-work superiority, benchmark the actual staged execution, including its batching, replay, score extraction, and controller/critic passes; do not transfer collection padding factors mechanically to a stopping policy.

Generation logits can mathematically supply the same likelihood/entropy quantities without a second model pass. However, cached decoding and full BF16 replay are not bit-exact in this implementation; the collector itself records this distinction. An optimized cached-feature implementation needs development validation of numerical features and downstream decisions, or retraining/recalibration under the new features, before its lower work can be claimed for the current predictor. Its softmax/entropy and syntax-replay overhead also remain nonzero.

## Evidence and limits

Read `selection/collect.py`, pinned upstream `sample.py::generate_ids_fast`, public development rows, and saved collection counters. No GPU work, model training, final-test outcomes, or protected benchmarks were accessed. This corrects the earlier threshold audit's overly broad exclusion of diagnostic replay. The repository's referenced `docs/NOTE_TEMPLATE.md` was absent, so this note follows the existing review format and required date frontmatter.
