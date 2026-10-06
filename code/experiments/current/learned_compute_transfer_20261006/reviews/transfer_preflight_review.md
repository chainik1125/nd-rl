---
written_on: 2026-10-06
written_by: agent:codex
status: bounded_preflight_complete
---

# Transfer data and model preflight

Read the transfer README, acquisition/preparation scripts, phase-one collection launcher, live launcher and frozen inventories before inspecting any new generated outcomes. The scope preserves the closed sprint and correctly separates ordinary187 from derived225 projections, independent pass1/oracle8 from contextual first/oracle8, and training arms from deployment rules.

CPU checks against actual bytes passed:

- Combined public SHA256 `969d72c5d861f91adcbf238577daf53ee129440010d4adcd2dea5cc99acf8e47`, all 412 unique IDs and exact source-row prompts, partitioned 187/225.
- No `reference_proof`, `reference_output`, `given`, `task_input` or `check` fields in public input records. The models consume the public prompt, not source reference/audit metadata.
- All 12 registered selector checkpoint hashes and five phase-one implementation hashes match the phase-one decisions, SHA256 `fca9e56597b9486f5504a2c2c0017ff714a817f08b61f735f220386d2d3324ad`.
- All nine live entries' base/checkpoint hashes and original DEV configuration hashes match their actual files. Fixed-one is selected at update100; fixed-four and primary RL checkpoints at200; format-only has no RL checkpoint.

Derived records have 164 base-cluster labels: 122 singleton clusters, 28 pairs, nine triples and five groups of four. Preserve the 225-row-weighted point estimate. Whole-base bootstrap draws should resample those 164 clusters and recompute each draw's row-weighted mean with its resulting denominator. Equal averaging of base means is a different estimand and should not silently replace the row-weighted result. Ordinary source-stratified theorem resampling is appropriate for the declared source composition. Intervals remain conditional on recorded samples and fitted checkpoints.

Before inference/analysis sealing, the agents should bind the new launchers, statistical implementation/settings and frozen feature-model coefficients as well as the old neural signatures. `selection_split: dev` in the phase-one decisions appears to be a compatibility field; the dataset rows correctly say `test`, and an explicit transfer seal must guard outcome access rather than relying on that field. Collection currently uses frozen old-code defaults for original checkpoint/source, whose actual checkpoint bytes were independently verified in this review; the final collection configuration must confirm those defaults resolve to the registered source/checkpoint.

The live launcher constructs inference arguments from the original DEV configurations and uses the old evaluator, preserving scientific settings. A small no-overwrite guard improvement was requested before freezing: reject dangling symlinks and an existing `.launch.py` snapshot in addition to output/log/ledger paths. Bind the model inventory in registry parent hashes. These are orchestration checks, not changes to the models or estimates.

No new outcome was opened and no neural inference or training was run by the reviewer. This note does not certify an as-yet-unwritten final transfer analysis or registry; those concrete artifacts require their own source/binding review.
