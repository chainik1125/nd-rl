---
written_on: 2026-10-06
written_by: agent:codex
---

# DEV-reproduced live aggregation numeric guard mismatch

The frozen `aggregate_live.py` SHA256 `70afd93cf3d62f59d48f305d9d2962fc7e26efa07a388a545d0c6c736071a80d` rejects valid integrated FP32 probabilities within roundoff of one. The independent [DEV reproducer](audit_live_numeric_guard.py) and [exact failing records](live_numeric_guard_dev_reproducer.json) read only the six registered cost/no-cost DEV evaluations. No live TEST outputs were opened for this finding; selection/staged MAIN results had already been inspected under their prior authorization.

Of 2,112 integrated DEV policy/theorem rows, 79 reject. Every rejection is exclusively the `correct_probability <= 1 + 1e-8` condition. Attempts, forced-termination probability, finite/nonnegative work and native reward consistency all pass their original checks. The maximum correctness excursion is 8.62055458e-8, less than FP32 machine epsilon 1.19209290e-7. The frozen path integrator already permits selection normalization error up to 1e-5; the aggregator's 1e-8 bound is inconsistent with that accepted numerical representation.

Recommended correction: a versioned, analysis-only successor allowing correctness probabilities within 1e-6 of [0,1], with no clipping, renormalization or changes to values, rewards, averaging, weighting, bootstrap, models, generation, policy choices or original registries. Keep other guards unchanged unless separately justified. Preserve the original analysis and bind the successor/addendum explicitly to the original registry and source hashes, recording that the issue was reproduced on DEV before live TEST outcome analysis.

This note establishes the defect and minimal correction rationale. Acceptance of a concrete successor requires a separate source-diff and addendum-binding audit; that review is pending at this point.

## Concrete successor acceptance

Reviewed and accepted the bounded successor sources: MAIN `aggregate_live_fp32_v2.py` SHA256 `cddf226e9801dacb14d7d94fc7aa04da2bf893f2297b3e4293fd2b9875260f83`, historical `aggregate_historical_fp32_v2.py` SHA256 `d21eaa8dc6c08332015390998a26134323b8f195b1cf9503a491509df73dd0d9`, and `analysis_addendum.py` SHA256 `3b36c3568e526a238ae0f48c4399768d1446ccf2bbd8cd7867808587b2ce624d`. Full diffs preserve inference sources and estimators. A metrics-function AST comparison confirms that only the correctness-probability bound changes; successor registration/provenance is handled separately.

The [successor audit](audit_numeric_successor.py) and [results](numeric_successor_independent_audit.json) find all 2,033 originally passing DEV estimate arrays bitwise identical and all 79 roundoff rows admitted. Synthetic tests continue to reject probability above 1+2e-6, nonfinite correctness, negative work, inconsistent reward and excess attempts. Addendum tests reject draft status, wrong registry or successor, altered bootstrap settings and requested renormalization. A temporary synthetic frozen addendum passes the metadata guard; no actual addendum was sealed or scientific analysis run by this reviewer.

Root must preserve both original registries and explicitly seal the analysis-only addendum before use. Its rationale should say that no **joint-live TEST outcome analysis** informed the correction: frozen-pool MAIN outcomes were already known. This is a numeric-domain guard fix, not a post-outcome model or statistical-estimator change.
