---
written_on: 2026-10-06
written_by: agent:codex
status: development_outputs_verified_guard_hardening_recommended
---

# Staged aggregation review

## Actual development output

Independent reconstruction matched all **40,960 extracted theorem-policy records**, seed-averaged endpoints, and all paired stratified bootstrap intervals with zero discrepancy. Each group/cost contains all 320 theorems; every BCE endpoint uses three seeds per theorem and every feature endpoint uses one scorer. The records agree with the respective frozen v3 reports. All inspected metrics are finite.

At cost .03, the mean BCE accuracy gain over its own logical-work-matched fixed mixtures is .030543, with the saved exploratory interval [.016785, .045375]. The feature gain is .032414 [.018018, .047598]. These intervals condition on the development-selected policies and saved pools; they are not confirmatory evidence after tuning. The separate scorer-specific comparison scope is correct and does not imply a global fixed-budget superiority claim.

The bootstrap first averages seeds within each theorem, differences the paired policies, and resamples whole theorems within family. Its current balanced input therefore preserves the 50/50 family target. It correctly leaves mixture weights fixed within the bootstrap. Report held-out work mismatch rather than refitting mixtures. The TEST registration check occurs before `load_run` reads derived outcome reports.

## Recommended pre-freeze guards

The following are implementation gaps, not defects observed in the supplied development data:

1. **Bind and require the seed inventory.** The code rejects repeated seed aliases but does not require BCE seeds {0,1,2} and one feature run. It also does not bind `training_seed` to a checkpoint/configuration provenance record. The same evaluated run relabeled as three seeds would be accepted. Require the declared inventory, distinct expected checkpoint identities, and verified seed provenance.
2. **Do not average unmatched seed sets.** A `None` matched mixture silently removes that seed's comparator rows while leaving its adaptive rows. If any seed lacks the declared comparison, mark that entire group/comparison unavailable, or explicitly restrict both sides to an identically declared seed set. Never subtract a three-seed adaptive mean from a two-seed comparator mean.
3. **Verify derived-run provenance.** Feature output currently need not match `policy.feature_model_sha256`. Check its recorded feature-model hash and current artifact; verify the BCE prediction/configuration chain and stored critic identity; verify evaluator/metrics snapshot hashes against the policy. Confirm each source report's means agree with its per-theorem rows. Bind the source summary/per-theorem file hashes into the aggregate output for reproducibility. The inventory hash currently binds their paths and policy hash, not their content.
4. **Validate all numeric inputs before averaging.** `stratified_interval` checks finite comparison vectors, but fixed-budget rows outside selected comparisons can carry nonfinite values into the report/plot. Check every consumed accuracy, attempt, work, and derived reward; enforce accuracy in [0,1], attempts in [1,8], and nonnegative work. Require a positive bootstrap count.
5. **Validate public identity sets, not only lengths.** The raw-to-transition compatibility check correctly preserves every original field, but source public rows need explicit unique theorem IDs and exact ID-set equality. Likewise reject duplicate candidate IDs before converting them to a dictionary. As written, repeated identical source rows can evade the length-plus-loop check. The actual supplied public files pass the intended identity comparison.

These guards should reject stale or incomplete analyses before final aggregation; no research method or selection rule needs to change. The aggregation's output currently retains seed-level rows but not a concise per-seed summary; retaining the existing standalone reports is adequate if the final report shows the seed spread.

## Evidence

[Independent audit script](audit_staged_aggregation.py) and [numeric reproduction](staged_aggregation_independent_audit.json). Reviewed `analysis/aggregate_staged.py`, `selection/aggregate_selection.py::stratified_interval`, `analysis/staged_dev_inventory.json`, `analysis/staged_dev_aggregate_v1`, and their frozen v3 sources. No shared implementation was edited and no final-test outcome was opened.
