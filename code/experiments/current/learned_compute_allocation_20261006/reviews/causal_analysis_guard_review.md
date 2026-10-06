---
written_on: 2026-10-06
written_by: agent:codex
status: targeted_guards_and_registration_verified
---

# Causal analyzer guard audit

The family-stratified theorem bootstrap and explicit available-state versus fixed-panel distinction match the reviewed estimand. The clean forced-four smoke analyzes successfully. Removing one branch while retaining its eligible state is rejected. Removing both the state and branch produces an explicitly incomplete panel, as intended for descriptive available-state estimates.

Temporary mutations against analyzer SHA256 `d3de0afb4b46d6c84c6fbdeb424d0dfbe95eaef43e694788a284e904b055a086` demonstrated acceptance of seven invalid inputs: acceptance probability 2; an invented branch family; a duplicate semantic state under a new state ID; inconsistent completion state count; negative new-attempt count; requested forced count inconsistent with configuration; and replicate ID 999. The duplicate-state mutation reported five analyzed states while still declaring the planned four-state panel complete.

The corrected analyzer SHA256 `9dd4bf4c0da9864fbf61e6ddf00b6c4cd712fc8f536819a9eb1e1716c634a288` now rejects all seven mutations. It implements authoritative joins among public theorem metadata, collected states, and branch records, bounds acceptance probabilities and future counts, checks exact replicate IDs, reconciles completion records, and derives the split from actual public metadata. The clean smoke still passes; removing both a state and its branch still yields an explicitly incomplete descriptive panel.

An additional synthetic registration audit wrapped **copied DEV smoke outcomes** in temporary synthetic TEST metadata and a temporary registry; no actual final-test file was used. The clean synthetic registration passes. Changed bootstrap settings, registry bytes, generation arguments, checkpoint hash, upstream source hashes, helper snapshot, generator snapshot, and a downgraded configuration split all fail **before scored branches are opened**. These targeted checks verify the requested safeguards, not arbitrary-corruption completeness.

Evidence: [reproducible mutation script](audit_causal_analysis_guards.py), [current mutation results](causal_analysis_guard_audit.json), [synthetic registration script](audit_causal_registration.py), and [registration results](causal_registration_audit.json). All mutations used temporary copies of existing DEV smoke records; no final-test outcome, GPU work, or shared-source edit occurred.
