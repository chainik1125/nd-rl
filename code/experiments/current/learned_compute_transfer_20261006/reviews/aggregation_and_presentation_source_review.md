---
written_on: 2026-10-06
written_by: agent:codex
status: source_review_with_guard_requests
---

# Transfer aggregation and historical mapping review

Read `live/aggregate.py`, `selection/aggregate_transfer.py`, the new phase-one launcher/seal scripts, and root's historical normalizer and comparison renderer. No new transfer outcome was opened. This is a source review, not acceptance of final numerical outputs.

Both aggregators implement the declared row-weighted derived estimand: whole base clusters are sampled with replacement and each draw recomputes its row denominator. Primary live estimates average the two draws per theorem, then the three fixed training seeds, then rows. Paired native-minus-own-first contrasts share bootstrap weights. Oracle outputs correctly omit fictitious deployment cost. Phase-one exact tie expectation and three-seed averaging are correct. Ordinary phase-one intervals stratify by source role; ordinary live intervals use unstratified base-cluster resampling. These are different declared uncertainty procedures and should not be described as identical.

The following guard requests were sent to the owners before new outcomes were reviewed:

- Phase-one launcher should verify frozen implementation/helper and feature-model bytes per child, not merely record current entrypoint hashes. Its feature scorer has no independent transfer child guard. Its aggregator should finish all launch/config/source/model metadata checks before reading predictions, bind private label bytes, require exactly eight nonduplicate Boolean labels, and validate finite feature accuracies in range. The mathematical estimator does not need changing.
- Live aggregation should additionally verify registered seed/repeats, base hash, exact launch commands and parent hashes before outcome reads. Its inherited inference fingerprint does not bind sampling draw seeds. Independent final metadata verification can cover this without silently changing sealed source. Raw oracle labels also need count/type consistency with saved public paths; a dictionary join alone does not enforce those semantics.

## Historical normalization and figure

Independently checked the normalizer's three closed-source hashes and all live scalar mappings against the old audited historical aggregate. All match exactly. The four headline rows are:

| Arm | Pass1 | Oracle8 | Returned | Attempts |
|---|---:|---:|---:|---:|
| Original generator, pooled fixed8 | .4236111111 | .625 | .5277777778 | 8 |
| Fixed4-trained control, fixed4 readout | .4097222222 | .4236111111 | .4080340357 | 4 |
| Cost-trained, native readout | .4143518519 | .4398148148 | .4140270928 | 1.0388868413 |
| No-cost-trained, native readout | .4143518519 | .4328703704 | .4095101381 | 2.3719608735 |

The new renderer maps phase-one pooled accuracy and live fixed4/native correctness to the intended fields. Its existing fixed4-oracle8 extrapolation caption is important. Requested caption improvements: explicitly call returned scores native categorical expectations and the figure point estimates without intervals; explain original pass1 as the average of eight exchangeable samples versus first-proof success on contextual joint paths. Keep contextual/seed/readout qualifiers in the derived figure too, whose separate footer currently omits several of them.

Neither low stochastic fixed-eight correctness nor the gap to oracle establishes an intrinsic coherence cost. Greedy index selection can differ sharply from native categorical selection. The returned-versus-own-first contrast is the primary learned-allocation comparison; original and joint generators are different systems. Final numerical and graphical review remains pending all completed transfer results.
