---
written_on: 2026-10-06
written_by: agent:codex
---

# Operational selector queue successor

Accepted the latest `selection/run_frozen_test_queue_shared.py` SHA256 `dc6d25de03e9797ef5abe8bc5ed656eb23fd4dcce281dc73e465fd2ab2696c1f`, including the fix resolving relative addendum paths before snapshotting. The original helper remains SHA256 `806ee26742c7c56202452a6d899b8f966fb7c5d69515eb2e6dc2f3d2a9b33e6f`.

The [independent mocked audit](audit_shared_selection_admission.py) and [results](shared_selection_admission_audit.json) confirm that signature-only and inference subprocess calls and the command parser are AST-identical to the original. Actual MAIN/historical entries still match the decision-parent commitments and each retains 37 distinct registered scientific commands. Both registries, the common operational addendum SHA256 `c0176301d56965d4b355c071f27c5df032ba546e9a8bfd2891da7dbaf5e89431`, and the separate selector amendment bind the exact old/new helpers.

Thirteen mocked cases pass: empty/foreign occupancy at the declared free-memory boundary is accepted, while insufficient free memory, same-UID occupancy, GPU0, draft status, wrong registry/helper/common-addendum identities and altered memory policy reject. Admission is AST-identical to the accepted causal wrapper. Each child revalidates registry, entries, addendum and helper identities immediately before a fresh occupancy check and launch. Manifests truthfully record actual successor/original/addendum hashes and foreign occupancy; snapshots include both addenda. Existing-output refusal remains and is strengthened for dangling symlinks.

This is an admission/provenance amendment, not scientific resampling or a new evaluation method. It makes no changes to models, data, inference orders, scores or analysis definitions. Root must seal the separate amendment and retain the all-nine-MAIN completion barrier before dispatching historical inference; this selector wrapper does not itself implement that external scheduling barrier. Co-located wall times remain unsuitable for isolated-latency claims. No GPU query, process launch or historical outcome read occurred in the review. Ruff passes for the audit helper.
