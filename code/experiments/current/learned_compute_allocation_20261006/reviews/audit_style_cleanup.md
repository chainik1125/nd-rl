---
written_on: 2026-10-06
written_by: agent:codex
---

# Audit helper style cleanup

Cleaned only the 19 `reviews/audit_*.py` helpers. No frozen experiment implementation or registry was edited. Original executed sources were copied before editing to `artifacts/learned_compute_allocation_20261006/reviews/original_audit_sources_20261006T052503Z/`. Its `manifest.json` SHA256 is `a45cc39e22eeb3cf8936036faee358b1f94e5d6f3263de33f4c6061829f55894` and lists the SHA256 of every original. All 19 originals were checked against that manifest after cleanup.

The complete changed-path list and both original/reformatted hashes are in [audit_style_cleanup_manifest.json](audit_style_cleanup_manifest.json). Historical numerical claims remain tied to the preserved originals. Workspace files are reformatted replay helpers; none was executed to produce new numerical evidence during this cleanup.

Changes comprise statement and import formatting, descriptive comprehension-variable names, lambda-to-function spelling, explicit file reading, and six unused import removals. Imports that intentionally follow runtime source-path initialization retain their location with explanatory E402 exceptions. Removed imports were unused `contextlib`, `io`, `copy`, `hashlib`, `itertools`, and the unused `matched_mixture` symbol (the same module remains imported for `joint_mixture`).

Reviewed semantic lint cases explicitly:

- B023: the registration audit's immediate tracking callback now binds its per-case `opened` list and original loader as default arguments. The metadata canonicalizer binds the current atom mapping similarly. The original callbacks were invoked within their defining iteration; binding preserves that intended behavior and prevents accidental delayed-call capture.
- B905: inspected all 27 original zip sites (paired arrays, candidate/label lists, verifier flags, and atom permutations). Added `strict=False` explicitly, preserving the original truncation behavior rather than adding new shape assertions during a style-only cleanup. Existing audit assertions remain unchanged.
- File reads now close the one directly flagged public file via `read_text().splitlines()`; inputs are the same JSONL records.

Checks: Ruff check passes for all 19 helpers; Ruff format check reports all 19 formatted; Python in-memory `compile(..., 'exec')` passes for all 19 without importing or executing them. The initial scoped check found 876 findings, including the two most recently added helpers; the final scoped check has none. No numerical reruns, GPU jobs, TEST outcome reads, or changes to experiment evidence were performed.
