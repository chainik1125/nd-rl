---
written_on: 2026-10-06
written_by: agent:codex
---

# MAIN frozen feature and staged-policy audit

Accepted the arithmetic in `analysis/main_test_features_v1` and `analysis/main_test_feature_staged_v1` following authorized post-seal MAIN access. The [independent audit](audit_main_feature_staging.py) and [saved results](main_feature_staging_independent_audit.json) use their own public feature construction, fixed-prefix subset enumeration and threshold expectation derivation rather than the production policy integrator. No model fitting, neural inference, policy retuning or historical outcome access was performed.

The frozen feature model and DEV policy hashes match the final registered method. Replayed features and scores on all 400 actual pools reproduce feature best-of-eight accuracy 0.875. Fixed-count selection enumerates every equiprobable k-subset and uniform maximum-score tie. For thresholds, the first qualifying candidate is uniform over qualifiers, each nonqualifier precedes the first qualifier with probability 1/(m+1), and expected attempts are 9/(m+1); if none qualifies all eight are used. These independent calculations and unchanged DEV mixture weights reproduce all 18,400 saved policy rows and summary/reward arithmetic: 131,310 scalar checks, maximum absolute difference 5.55e-17.

At the DEV-chosen threshold 0.85 for cost 0.03, selected accuracy is 0.849485119, mean attempts 2.307160714, and expected unpadded logical model-input work 14,012.7423. This includes 7,022.5305 required feature-replay input tokens; replay is not discarded as diagnostic overhead. Fixed-one avoids scoring; mixtures inherit the exact component-specific fixed-one exemption.

The unchanged DEV work-matched fixed mixture yields accuracy 0.819733518 and TEST work 14,795.4833. The TEST work difference is −782.7410, not zero: DEV matching need not transfer exactly to TEST. Report that residual rather than claiming exact TEST compute matching or recalibrating weights. The paired family-bootstrap aggregate is a separate forthcoming check; this arithmetic audit alone is not a confidence-interval claim.

These are exact expectations for uniformly ordered independent frozen pools. They are not contextual generation policies, actual padded GPU forwards, end-to-end latency, or an implementation-free FLOP metric. The result can support a frozen score-threshold policy comparison with its own prespecified fixed mixtures; it does not establish joint-RL stopping skill or global compute superiority over differently measured systems. Ruff passes for the new audit helper.
