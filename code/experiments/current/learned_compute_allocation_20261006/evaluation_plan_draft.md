---
written_on: 2026-10-06
written_by: agent:codex
status: development_draft_not_test_authorization
---

# Evaluation plan before final test access

This document is a development draft. It does **not** authorize opening final test outcomes. A separate frozen protocol must name exact checkpoint, data and evaluator hashes, chosen hyperparameters and endpoints before test generation or scoring.

## Primary selection comparison

The main distribution is already frozen: 400 fresh test theorems, 200 per family, eight independent r16 attempts at temperature 0.8. Do not exclude all-invalid pools, malformed attempts, unusually long inputs or unfavorable theorem cases. Keep public outcomes and private Lean verdicts separate. Explicitly retain the two families in all saved per-theorem records.

Choose selector view, architecture, optimizer and checkpoint-selection rule on training/development only. Report three training seeds when completed, with exact starting checkpoint and independent training RNGs. Distinguish full-pool recognition from shared per-candidate scoring. Record the 8192-token full-pool context extension, actual overflow rate and outcome-blind fallback. A model-selected candidate must be one of the exact saved attempts.

Report random selection, first candidate, likelihood, entropy, shortest, public syntax, syntax-plus-likelihood, the train-only low-capacity feature predictor, learned model selection and oracle availability. Headline the learned model's difference from both random selection and the strongest comparator selected on development. Report selected accuracy, oracle-minus-random gap recovered and selection accuracy conditional on at least one valid attempt. Conditional accuracy must not replace all-theorem accuracy.

Average repeated runs within each theorem before paired theorem resampling. Report seed-level results as well as the cross-seed mean; a theorem bootstrap conditional on three observed seeds does not establish robustness over all possible training runs. Break out family results without choosing the headline family after seeing test outcomes.

## Cost-aware stopping

The intended primary live experiment uses the same trainable model for proof generation and stop/index decisions, initialized from outcome-blind format adaptation of r16. Keep the format-only initialization as a measured baseline. The frozen-pool controller remains a separately labeled diagnostic reduction.

Use correctness minus `lambda * (attempts - 1)`, with primary lambda 0.03; any other cost settings must be named before final evaluation. Compare learned stopping with fixed 1/2/4/8 attempts using the same checkpoint and selection rule, a separately trained fixed-count control when completed, and development-selected fixed-count mixtures. A test-fitted mixture envelope is descriptive and optimistic, not a deployable baseline. Also report the independent feature-based threshold policy on frozen pools, with its threshold chosen solely on development.

Report correctness, reward, attempt count, actual generated tokens, sampler forward work, controller work and inference wall time. Separate diagnostic likelihood replay and teacher-scoring overhead. Feature-baseline additive generation-token sums do not reconstruct batched GPU work. Count truncations and forced stopping separately from sampled acceptance. State whether sampling uses FP32 or BF16 and keep precision matched within each comparison.

The current public-feature predictor consumes likelihood and entropy from an extra full-forward replay. That replay is **required inference work** for this exact predictor, not removable diagnostic overhead. Charge it for every candidate scored, including the first candidate in a threshold policy; a fixed-one policy needs no selection or replay. The text-only BCE critic does not require these confidence features and instead pays its own scoring forwards. Pooled-record token sums are lower bounds/proxies because padding, EOS-check oversampling and kernel execution depend on deployment batching. Cached generation logits are a possible optimization, but their numerical equivalence must be checked before using them with the frozen feature predictor.

## Correctness-only stopping and causal continuation

The zero-cost model must not inherit a cost-trained or experiment-1 selector. It may use the same outcome-blind format initialization, explicitly described. No correctness-to-stop supervised labels or implicit summed-length penalty are permitted.

Acceptance frequency alone is not evidence for useful stopping. For retained policy states strictly before hard limits, estimate immediate-accept success using the model's selection probabilities over existing proofs. Force continuation from the identical state and sample future attempts without intermediate checker feedback. Keep two estimands distinct: one extra attempt followed by acceptance, and continuation followed by the original policy. Save replicate futures, raw decisions, selection probabilities, exact proof identities and final verdicts. The protocol must freeze the state panel, continuation replicate count and analysis before final evaluation.

The bounded development design recommended by independent review uses 64 prespecified theorems and two states per theorem, after one and two attempts. Force collection to the second state even if the policy would accept earlier; call this an intervention panel rather than natural policy occupancy. Draw eight independent one-extra-attempt branches per state, then integrate index probabilities exactly. Bootstrap whole theorems, retaining both states and every branch together. Examine acceptance versus continuation value within attempt-count and incumbent-validity strata. Individual noisy negative estimates do not establish that those states have negative true continuation value.

To investigate a context mechanism, on a prespecified subset repeat branches with prior history removed from the generator prompt, while retaining the original theorem and the complete selectable pool for the chooser. Match branch random seeds. This intervention is different from merely forcing continuation. Decompose post-continuation success into probability mass on previously valid candidates and mass on the new valid candidate; also report the new attempt's validity and oracle availability. This separates selection dilution from changes in generation quality. The primary index policy is stochastic categorical selection, evaluated by exact expectation; argmax results are secondary if included.

Report whether acceptance predicts negative continuation value after accounting for attempt count, alongside generation validity and selector retention. Oracle availability over retained candidates is monotonically nondecreasing. A decline in selected success must therefore be attributed to the policy's selection/generation behavior, not loss of an available valid proof. Unadapted marker-induced collapse does not establish the trained zero-cost hypothesis.

## Secondary transfer and reporting limits

Textbook72 may be a separately labeled historical benchmark if the final protocol names it; its known reuse and base-training overlap prevent a fresh-generalization claim. Protected hidden test files remain excluded. Any extension to pools larger than eight must explain how an eight-index chooser is evaluated and must remain separate from the primary test.

If any proposed component is unfinished or fails its implementation gate, state that directly and report the most informative completed test. Do not substitute a protocol smoke, teacher KL improvement or an oracle pass@k curve for evidence of learned model-selected success.
