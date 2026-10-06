---
written_on: 2026-10-06
written_by: agent:codex
---

# Actual final registry and MAIN transcript audit

After root explicitly authorized MAIN outcome access following the 05:33:52 UTC final seal, independently verified the actual parent/collection/public/registry chain and reran literal rendered-proof verification. The [audit script](audit_actual_main_seal.py) and [results](actual_main_seal_independent_audit.json) are new post-seal evidence, separate from earlier synthetic protocol tests and the style-only helper cleanup.

Verified registry SHA256 values:

- Parent: `51c704b30edfe9ff748378a2ec18aaf189da48c36bceaf7ec186bf5f76579a7b`.
- MAIN: `bd289d62de0e5c3395fa10330177385184cc66d67d09fc82cead10a65894a1d2`.
- Historical: `d228e09e16fa86aa69160e84464eca5064fcecde3c057393e7cc0eecba1b2d7d`.

All six parent input files and 212 collection source files still match their frozen hashes. The nine registered live models, fitted policies and selected DEV configurations match their bytes. Collection launch records contain exactly the registered collector/renderer/sealer commands with successful return codes. Both public seals match the parent plans and current public bytes. Reconstructing the final registries from the parent's registered decisions plus only allowed sealing metadata yields exact equality; the historical registry references the exact final MAIN hash. Every registered analysis signature hash is internally consistent. The records attest that private outcomes were not inspected before sealing; hashing can verify these records and their integrity, not independently prove the absence of an unrecorded human read.

The MAIN measurement-audit script and input hashes match the saved `transition_lean_audit.json`. Independent CPU Lean checking of all 3,200 literal public transition transcripts yields 2,291 valid proofs and zero disagreement with private labels. All 2,559 completed transcript strings exactly equal their checked private sources. The complete pooled-length inventory also matches; one of 400 pooled inputs exceeds 8,192 tokens and must retain the registered fallback rather than be excluded. No method or evaluation decision was changed after observing this.

Historical access in this review was limited to registry/launch/seal metadata and hashing public file bytes. No historical verdicts or performance summaries were inspected. The new audit helper passes Ruff; its actual Lean run took approximately 6.21 seconds wall time and 43.75 process seconds.
