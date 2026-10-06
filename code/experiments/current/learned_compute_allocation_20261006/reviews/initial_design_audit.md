---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Initial independent design audit

## Summary

The first decisive question is whether raw proof candidates contain semantic selection headroom after accounting for syntactic completion. Train only after establishing that headroom and that the proposed input fits the checkpoint's usable context. A frozen-generator classifier/controller is a useful preliminary experiment; it does not by itself demonstrate joint proof-generation training, contextual interference, or meta-learning.

Review context: `dmitry/main`, commit `315f36266bb4dd7feeff87228cb2662f8a36badc`. This review used the kickoff, proposal, Dan's two model cards from `origin/dan`, and read-only local proof-state source. The local source must still be matched to the downloaded checkpoint's provenance. No empirical candidate-pool results have been inspected at this stage.

## Ranked risks and decisive checks

1. **Verifier or termination information can solve the task accidentally.** The stock sampler returns strings prefixed with `LEANREJ` or `LEANPARSE`; never pass these to the selector. Capture raw generated actions, including the failing action. Store checking outcomes separately. Compare syntactic completion against strict validity, especially among finished candidates.
2. **Eight-proof selection is a large interface change.** The model was trained on focused proof states and single actions. Measure pool token-length quantiles, worst cases, truncation, and training-context exposure before declaring selection feasible. ALiBi's ability to execute at a length does not establish reliable behavior there. A per-candidate scoring head avoids pool-length limits but is a changed experiment and needs explicit labeling.
3. **Theorem overlap may inflate apparent transfer.** Exclude or group alpha-equivalent formulas and premise permutations; report stronger structural-family overlap as well. New atom names are not a structural holdout. Exposure includes K12 pretraining and all expert-iteration target sets, not just the selector's train split.
4. **Weak baselines obscure whether learning helped.** In the same pools compare random expected accuracy, first candidate, maximum mean action log likelihood, syntactic completion followed by likelihood, and a development-trained low-capacity predictor using visible features. Keep the strongest chosen baseline fixed before test. Untrained selector output is a secondary control, not the only baseline.
5. **Stopping claims can outrun the intervention.** Cost and no-cost policies must start independently from the declared initialization. A frozen independent pool removes history effects on generation. A policy that stops early under zero cost has not demonstrated harmful context. A classifier trained across tasks is not automatically meta-learning.

## Exact source observations

Read-only sources: `../proof-state/nd-takehome-dan/state_env.py` and `state_sample.py` relative to the sprint's parent folder (full paths below).

- `code/experiments/current/proof-state/nd-takehome-dan/state_env.py`: `Env._check_atomic` checks term shape and whether cited names are in scope. It does not verify the declared formula of an atomic `have` against its term. `exact` checks the declared formula against the focused goal. Therefore syntactically completed but semantically invalid proofs are possible. Their empirical frequency remains unmeasured.
- `code/experiments/current/proof-state/nd-takehome-dan/state_sample.py`: `env_generate(..., gate=True)` returns the Lean gate's strings. The documented contract marks rejected complete proofs `LEANREJ` and incomplete/malformed attempts `LEANPARSE`. `gate=False` does not suffice to preserve raw failures: `Env.nd()` still replaces them with a diagnostic string. The candidate collector must explicitly retain actions.
- `Env.state_tokens()` includes declared hypotheses and focused goals. These are environmental bookkeeping rather than full Lean correctness labels, but exposing transition failure flags or the reason text to a chooser adds privileged information. Distinguish text-only structural heuristics from privileged metadata diagnostics.
- The model cards describe 9,560,832 parameters, six layers, hidden dimension 384, proof-state SN-v2, and 16 expert-iteration rounds. They do not establish eight-proof context competence.

## Minimal pilot and baseline definitions

Use approximately 100 development theorems with eight fresh independent attempts each, before opening the final test. Record strict correctness, textual completion, actual generated tokens, number and size of state prefills, action entropy at a fixed diagnostic temperature, and mean log likelihood of sampled action tokens under their actual states. Do not score a concatenated reconstructed proof as though it were an in-distribution state-action sequence.

For each pool with correctness vector `v`, random expected success is `mean(v)` and oracle success is `max(v)`. Report their difference and the fraction recovered by each selector; abstain from an aggregate fraction when the denominator is negligible. Report full-pool accuracy and the subset with at least one valid candidate, retaining all-negative pools in the former.

Use two completion controls: (a) an inference-legal parser operating only on visible raw candidate text; (b) environment completion metadata, explicitly privileged and diagnostic. Among pools containing both valid and invalid finished proofs, test whether the learned model discriminates beyond completion. This is a stratified diagnostic, not a replacement for full-distribution evaluation.

Candidate order must be shuffled independently of validity and replayed under permutations. A scorer should show predictable permutation equivariance or its violations must be reported. Bootstrap whole theorems, including all pools/rollouts of each theorem. Selection training seeds and generator checkpoint seeds are different sources of variation.

## Stopping interpretation

For independent candidate generation and an incumbent-preserving selector, adding a candidate cannot lower oracle success. A decrease in model-selected accuracy may instead arise from a selector that forgets or reranks its incumbent. Include an incumbent-preserving confidence baseline and distinguish that mechanism from harmful proof-generation context.

Compute-matched mixtures interpolate fixed-budget policies using weights chosen on development data. Compare actual generated tokens plus repeated state prefills and chooser evaluation, not only attempt count. With zero attempt cost, a perfectly incumbent-preserving policy has no correctness-based incentive to stop before the cap. Zero-cost early stopping can still follow learned indifference, initialization, or imperfect selection; only replicated force-continue versus accept branches can show negative continuation value for the actual policy.

## Next steps

Audit the concrete collector schema and raw pilot samples; recompute the completion/validity table; verify independently initialized cost/no-cost lineage; inspect evaluation masks and tests; later reconcile headline values directly against immutable raw artifacts and cold-read the final graphs.
