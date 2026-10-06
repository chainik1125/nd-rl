---
written_on: 2026-10-06
written_by: agent:codex
status: approved_separate_protocol_pending_registry
---

# Historical textbook retention and transfer

This supplementary benchmark reuses the public textbook72 source. It is not a
fresh generalization test. Selection's canonical exposure audit finds one known
Stage-1 pretraining match, and zero matches in the 4,495 EI target rows and the
actual 90,436-row r16 final training mix. The historical source's 14 `rl_train`
roles therefore do not establish that those 14 problems trained this r16.

Use `artifacts/learned_compute_allocation_20261006/data/textbook72_historical_public_v1/public.jsonl`:
72 rows with unchanged proof-state prompts, source roles `textbook_dev` (58) and
`rl_train` (14), and no reference-line information. No reference lengths become
model inputs or sampling limits. All 72 were excluded from MAIN supervision.

Evaluate the six selected primary models plus selected fixed-first, fixed-four,
and format-only controls, after the fresh MAIN evaluation. Freeze a separate
registry before any fresh or historical test outcomes are opened. Use two full
paths per theorem, seed 92006, and the unchanged bounded MAIN inference caps.
Checkpoints and DEV-fitted counter/context/budget policies remain unchanged.
Do not tune on historical outcomes. These settings do not match upstream
pass@256 with action/step limits 512/96.

## Separate aggregation

`aggregate_historical.py` imports the frozen `aggregate_live.py` helpers without
changing the primary analysis. The command requires `--runs`, `--roles`,
`--frozen-protocol`, `--main-frozen-protocol`, `--exposure-audit`, and `--out`.
`--roles` points to the public historical metadata above. `--exposure-audit`
points to `artifacts/learned_compute_allocation_20261006/data/textbook72_metadata_audit/exposure.json`.
The runs manifest uses the same nine arm/seed names as MAIN, plus the newly
collected historical path folders and applications of the original DEV-fit files.

The historical registry retains the primary schema, except for its dataset hash,
seed 92006, and theorem count 72. It additionally contains:

- `main_frozen_protocol_sha256`: exact frozen MAIN registry hash.
- `historical_exposure_audit_sha256`: exact exposure-audit hash.
- `statistical_analysis.script_sha256`: `aggregate_historical.py` hash.
- `statistical_analysis.aggregate_live_dependency_sha256`: locked primary helper hash.
- `statistical_analysis.family_data_sha256`: historical public metadata hash.
- Statistical settings: 10,000 bootstrap samples, seed 102006, confidence 0.95.

Registration checks require all nine model/base/inference and DEV-fit signatures
to agree with MAIN. Primary helpers then check model/data/draw/arm registration
before reading primary outcomes, and verify derived fits before including them.

Report all 72 together and both source roles separately. Average draws within
theorem and trained seeds within arm. Descriptive paired bootstrap intervals
resample whole theorems within source role; they describe benchmark variation
conditional on the fitted models, not fresh-population generalization or new
training seeds. Save each seed, categorical primary and greedy sensitivities,
selected correctness, attempts, native/common-cost utilities, measured work,
fixed/count/context/DEV-mixture contrasts, and finite-path oracle availability.
