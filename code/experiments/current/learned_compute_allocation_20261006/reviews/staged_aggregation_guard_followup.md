---
written_on: 2026-10-06
written_by: agent:codex
status: targeted_adversarial_checks_passed
---

# Staged aggregation guard follow-up

The strengthened aggregator accepts all four clean development runs and rejects all six temporary-copy mutations below before creating any aggregate output:

| Mutation | Rejection |
|---|---|
| Remove BCE seed 2 | Required seed inventory incomplete |
| Relabel seed 0's checkpoint/run as seed 1 | Training seed differs from provenance |
| Change fixed-four summary accuracy | Summary differs from theorem records |
| Insert NaN in fixed-four work | Nonfinite policy metric |
| Duplicate aggregation public theorem ID | Duplicate public IDs |
| Duplicate source public theorem ID, updating its hash | Source theorem set differs |

The v2 development estimates and intervals are exactly unchanged from independently verified v1. The new per-seed summaries and provenance fields are additions. These checks address the requested failure modes; they are not a claim of exhaustive validation of arbitrary corrupted artifacts.

Reproduction: [script](audit_staged_aggregation_guards.py), [results with audited source SHA256](staged_aggregation_guard_audit.json). Temporary copies were created under this review directory and removed automatically. No shared implementation, original artifact, or final-test outcome was changed or accessed. The separate approved frozen-head follow-up does not alter these main-method checks.
