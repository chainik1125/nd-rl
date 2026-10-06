---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Live training and secondary benchmark review

## Live integration finding

The joint runner trains the same model's proof-generation tokens, acceptance decisions, and selected-index decisions from final selected-proof correctness minus attempt cost. Its sum of all sampled-token and decision log probabilities, averaged over complete episodes with a leave-one-out reward baseline, matches that stated objective. Generation is no longer frozen in this path. No intermediate correctness label triggers acceptance or enters the model input.

The selected proof is retrieved from `private_attempts[selected_index]`, paired with the original theorem statement, and checked in Lean. I independently rechecked all six completed selected proofs in the eight-episode safe smoke: **zero Lean disagreements**. The saved [audit](live_smoke_independent_audit.json) also confirms 18 attempts, five selected successes, mean reward 0.5875 at cost 0.03, and zero mismatches in reward, generated-token counts, or generator-prefix-token counts. These are training integration results, not estimates of learned performance. No forced episodes occurred, so this smoke alone does not validate actual cap-boundary behavior.

The runner's saved replay diagnostics show zero raw-token log-probability replay error. Sampling and replay both use float32 in this path, unlike the offline collector's bfloat16 diagnostic path. Environment-assigned effective names form subsequent histories; exact raw sampled tokens, including PAD/EOS, form the policy likelihoods.

One concrete bug was reported and fixed: fixed-count training can leave the unused acceptance head's gradient `None`, so an unconditional `.grad.norm()` in metric logging crashes that control. The allocation researcher added a guard and launched its own fixed-count smoke; that run's result is not yet independently reviewed here.

## Limits on interpretation

The selected live configuration uses an 8192-token cap with a checkpoint whose original configured context is 1024. This must be described as an interface/context-extension intervention. Attempts after the first additionally receive prior proof histories, unlike the original model's focused-state interface. Any degradation could result from that distribution shift; it cannot automatically be attributed to a general property of reasoning.

The fixed reserve for a future attempt is selected before observing its actual length, which avoids privileged future-size access. The separate state reserve is a conservative design value, not a mathematical guarantee that every future proof state fits; actual action/context truncations must remain visible in results. Claims about harmful continuation require states that genuinely have room for complete branches, with hard-limit and token-cap failures separated.

A fixed-count control must report actual attempts when context limits prevent the requested count. Fair compute comparisons require generated tokens, repeated generator-state prefills, and chooser prefills. The live and frozen-pool paths answer different questions and should have separate results and checkpoint lineage.

## Secondary textbook benchmark

The retrieved source README names `targets/test_short_prompts.jsonl` and `targets/test_long_prompts.jsonl` as one-run test files. `AGENT_POLICY.md` repeats that restriction. I found no additional restriction specific to the public `textbook72` benchmark in those documents. A frozen secondary evaluation on the already-published textbook benchmark is consistent with the inspected source guidance without accessing either protected test file.

Freeze the trained checkpoint, sampler settings, methods, budgets, and reporting choices before generating benchmark outcomes. Describe the benchmark as historically reused, and disclose the model card's known K12 overlap; do not call it fresh or contamination-free. This use does not alter the protection of unrelated final-only sets.

Use eight-candidate selection as the primary matching task. A 32-attempt independent oracle curve can provide useful context, but choosing among 32 candidates changes the trained eight-index interface. Any grouping, hierarchical selection, or stopping beyond eight must be specified as a transfer protocol before viewing outcomes.

## Evidence

Inspected `allocation/live.py`, `allocation/train_live.py`, retrieved `lean_gate.py`, source `README.md` and `AGENT_POLICY.md`, and the immutable `allocation/live_safe_smoke_cost_s11` configuration, episode log, metrics, and runner snapshot. Rechecking used local CPU Lean only. No GPU run or protected test access was performed by this reviewer.
