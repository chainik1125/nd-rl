---
written_on: 2026-10-06
written_by: agent:codex
---

# Operational live launcher successor

Accepted `allocation/launch_live_test_shared.py` SHA256 `ef977ee575a632c72252945a103179352d82d6a9656b5366f36bc6d0cc2bde07`. The original launcher remains unchanged at SHA256 `1f96343cdc8092bb30bf0674fcd556d8dce2ebc173a43673fc93b5b6f7e2e19c`.

The [mocked independent audit](audit_shared_live_admission.py) and [results](shared_live_admission_audit.json) verify by AST comparison that original preflight/scientific command construction changes only to validate the operational addendum, check the original helper's registered identity, and replace idle-only admission with the declared shared-memory policy. Its admission function is AST-identical to the accepted causal successor. Twelve mocked admission/binding cases pass, including exact-threshold acceptance and rejection of insufficient memory, same-user occupancy, unclaimed GPU, draft status, mismatched registries/helpers, and altered memory requirements.

Every generation/readout phase verifies unchanged registry and addendum bytes and revalidates both original and successor source identities. Generation takes a fresh occupancy snapshot immediately before launch. The ledger records the actual successor hash, original helper hash, exact addendum hash/path, foreign process/UID information and free memory; it does not falsely claim that the unchanged original wrapper executed. Existing child science guards, models, seeds, methods, output refusal rules and postprocessing commands remain intact.

This approval is an operational admission change only, not a method or estimator change. Root may bind this exact hash and the accepted causal hash in the common operational addendum. Preserve co-location metadata and timing caveats; device sharing prevents interpreting measured wall times as controlled isolated latency. The audit made no GPU queries, launches, process modifications or outcome reads. Ruff passes.
