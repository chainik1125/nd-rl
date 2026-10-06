---
written_on: 2026-10-06
written_by: agent:codex
status: completed_dev_engineering_diagnostic_independently_audited
---

# Cached confidence: small DEV numerical check

This separately authorized engineering diagnostic leaves every registered model, scorer, policy, source, registry and replay charge unchanged. It is not a speedup experiment. No MAIN or historical TEST outcomes were accessed. The primary panel is 16 hash-selected MAIN DEV theorems, eight per family, with eight original-r16 attempts each, T0.8, batch 128 and the existing 256-action/48-step caps.

The process-local callback calls the pinned `_gumbel_pick` exactly once and returns its untouched selected tokens. It computes FP32 behavior-T0.8 log probability, base-T1 log probability and base-T1 entropy from that call's logits. Original row slots remain stable through compaction. Callback records after each row's first actual EOS are ignored; sampled PAD before EOS is retained. Every retained callback token sequence and count is asserted identical to the returned raw action. Early stopping is exclusively `eos`, avoiding inserted unsampled terminators. Replay uses unchanged `collect.likelihood_batch` on exactly the same raw prefixes and actions, before environment normalization.

All 128 candidates and 1871 action rows completed. A preliminary matched-seed eight-row batch preserves original/instrumented tokens exactly. Across candidates, mean absolute cached-minus-replay differences in mean behavior log probability, base log probability and base entropy are respectively 0.00015055,0.00012573 and0.00007222; maxima are 0.00267573,0.00204995 and0.00055998. Thus cached and BF16 full-replay confidence are not bit-identical. Cached FP32 token values were accumulated in Python FP64, whereas replay reduces each action in FP32, so differences include reduction arithmetic as well as cached/full-forward computation. No sampled PAD occurred in the empirical pilot; a separate synthetic CPU check covers retained PAD, dead-row suppression and original-slot compaction.

A small secondary check uses original r16 in FP32, eight initial DEV action rows from both families. It does not test a live-format checkpoint or complete FP32 proof trajectories. Maximum normalized differences are 1.60e−7,1.56e−7 and1.20e−7. Raw original/instrumented arrays are preserved for both its BF16 identity repeat and FP32 control. This supports only the stated narrow numerical comparison.

Runtime matmul-precision and TF32 flags were not recorded in the completed processes. Autocast modes are documented by the saved source; “FP32” specifies the tensor/computation mode, not a bit-exact hardware guarantee. Per-token callback events and cached logits were not persisted. Executed token/count assertions and saved original/instrumented identity arrays support mapping, while the independent audit reproduces raw action prefixes and action-to-candidate aggregates; it cannot recompute unavailable cached per-token logits.

A CPU-only application of the unchanged frozen feature model reconstructs public candidate records and substitutes only these three aggregates. All other seven features are asserted identical. All 16 fixed-eight argmax tie sets and all 128 crossings of the frozen 0.85 threshold agree, without reading validity labels or refitting. Scores differ: maximum feature-logit change is 0.11558 and sigmoid-probability change 0.00328387. Agreement on this small panel does not establish general decision equivalence, correctness, calibration or a substitute production protocol.

## Timing scope and evidence

The original fields named `process_wall_seconds_monotonic` are preserved, but measure `main()` entry through report assembly, excluding Python/import startup. Recorded scopes are 121.55 seconds for BF16 and 5.44 seconds for the small secondary check, lower bounds on external child occupancy and not GPU-active time. BF16 separately records 114.02 seconds sampling with callback and 2.47 seconds replay. These include synchronization/host overhead and are not isolated kernel benchmarks. Callback overhead prevents a speedup conclusion. The first run's host-UTC guard was supplemented by clock-tool supervision; the secondary uses a clock-tool anchor plus monotonic budget. Both exited successfully, and GPU3 was explicitly released at clock-tool 08:12:21 UTC, before the 08:25 deadline.

Artifacts below are relative to `artifacts/learned_compute_allocation_20261006/selectors/`:

- `cached_confidence_dev16_v1/`: raw per-action prefixes/tokens/cached and replay values, per-candidate differences, preliminary token identity, configuration, source snapshots and synthetic CPU bookkeeping check.
- `cached_confidence_fp32_dev8_v1/`: small FP32 raw action comparison, separate raw BF16/FP32 identity arrays and source/configuration hashes.
- `cached_confidence_feature_dev16_v1/`: label-free candidate score differences, pool tie sets, threshold agreement and input/model/source hashes.
- `cached_confidence_resources_v1.json`: explicitly scoped diagnostic-only timing ledger. These runs are excluded from the final registered 102-process accounting.

The scripts are `cached_confidence_diagnostic.py`, `cached_confidence_fp32_control.py` and `cached_confidence_decision_check.py`. Scoped Ruff passes. One CPU tokenizer-import error was corrected before the successful decision check, with no output or model changes from that failed attempt. The independent CPU audit passed all 1871 reconstructed prefixes, candidate aggregate sums, frozen-feature scores (maximum reconstruction difference 7.11e−15), tie sets, threshold crossings and synthetic bookkeeping. Its evidence is in the [independent audit results](../reviews/cached_confidence_independent_audit.json) and [audit source](../reviews/audit_cached_confidence.py); the [final review note](../reviews/cached_confidence_review.md) records acceptance and limitations. No registered cost reduction follows from these observations.
