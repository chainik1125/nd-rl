---
written_on: 2026-10-06
written_by: agent:codex
status: bounded_source_review_accepted
---

# Fixed-panel causal aggregate

The numerical estimand is implemented correctly. For each theorem, the aggregate averages the two declared counts and two checkpoints with fixed equal weights, then bootstraps whole theorem IDs within family. History-removal effects are differenced within the same checkpoint/theorem/count before averaging. A missing required state makes the fixed endpoint unavailable instead of silently changing weights. Count-specific and eligible-state estimates remain separate descriptive summaries. The common-first-16 panel makes the four continuation horizons comparable in theorem composition. Three focused synthetic tests pass independently.

The initial source did not bind seed aliases to training provenance or require a consistent checkpoint across conditions. The updated source now verifies zero-cost learned-arm training seed/configuration, actual checkpoint bytes, distinct checkpoint identities for seeds 101/102, their consistent mapping across all conditions, and identical actual public-panel bytes. These checks address the demonstrated provenance concern without changing the estimand.

Aggregate SHA256 `9c1ce685cbd37e00e926760ac9edd579fb1cc9f85afb61c610724ae17bf30591` implements both remaining checks correctly. It reruns the unchanged audited analyzer in a temporary directory and compares the entire summary and per-state output. It binds all consumed state, attempt, branch, configuration, and completion files. It compares incumbent raw/effective/transition content, action seeds, canonical names, identities, family, hard-limit status, controller lengths and probabilities across conditions on the common first 16 theorems. Probability tolerance is explicitly `1e-7`; future branches are not incorrectly required to agree after interventions diverge.

The TEST aggregate guard precedes per-state and summary reads, validates the aggregate registration, and binds each run's original generation registry and analysis settings. Planned missing runs remain unavailable rather than being dropped from the endpoint. The common-panel horizon figure is explicitly descriptive; it does not claim significance from separate intervals. This source is accepted for registration within this bounded review. The actual complete eight-run aggregate still requires arithmetic verification once available; this is not a claim that an unseen full run has passed. This review does not authorize final-test access.

Evidence: reviewed `context/aggregate_causal.py`; ran `code/tests/test_lca_causal_aggregate.py`, **3 passed**. No real final-test outcomes or new proof generation were accessed.
