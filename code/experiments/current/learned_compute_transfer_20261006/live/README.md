---
written_on: 2026-10-06
written_by: agent:codex
status: frozen_evaluation_running
---

# Transfer of existing joint proof-generation and retry policies

Evaluate the six already-selected cost/correctness-only checkpoints and the trained
fixed-four control on Charles's pinned 2026-10-02 release, without training or tuning.
The fixed-one and format-only models are supplementary controls. Reuse the audited
textbook-72 evaluation rather than generating replacement draws.

The [new protocol](../../../../../artifacts/learned_compute_transfer_20261006/live/protocol.json)
registers all nine inherited model/inference signatures, combined public data,
unchanged historical inference implementation, and new orchestration/statistics.
Only the dataset and prospectively fixed sampling seed change. Two contextual
paths per theorem use seed 95006, FP32, temperature 0.8, maximum eight attempts,
1,024 emitted tokens per attempt, 256 per action, 48 actions, and context 16,384.
The historical evaluation used seed 92006 with the same caps and precision.

There are two separate scopes: 187 ordinary theorem rows are the primary transfer
benchmark; 225 derived rows are sequent-only diagnostic projections. The projection
omits given proof prefixes, task markers, and structural constraints unsupported by
the inherited model interface. It is not a native-task score for Charles's derived
suite, and there is no pooled 412-row native-task headline. Every source component
and derived nuance is retained separately.

Report own first-proof validity, oracle availability after four/eight attempts,
selected correctness under the native policy, expected attempts, forced termination,
and same-checkpoint fixed-one/four/eight readouts. Oracle availability refers to the
first k candidates of a bounded contextual path; those candidates are not iid and
no iid pass@k estimator is applied. Oracle availability is not deployable selection.
The fixed-four model was trained with four attempts; its primary return policy is
fixed-four, while evaluation at eight is secondary. Correctness-only native stopping
can terminate early and had no explicit coherence reward during training.

Analysis averages the two paths within each release row, then the three fixed RL
seeds within primary arm. Point estimates are row-weighted. Paired 95% intervals
resample whole base-problem clusters within each scope with 10,000 draws, seed
192006, and recompute the row-weighted denominator in every draw. The ordinary
scope has 187 unique base clusters; the derived projection has 164. Intervals
condition on these frozen models and recorded paths, not a new training-seed
population. Controls each have one seed and selected checkpoint ages differ.

All registered launch/config/model bindings must pass before aggregation reads
integrated outcomes. Generation snapshots and separate process ledgers preserve
commands, GPU occupancy, exact hashes, and whole-child process times. No closed
sprint file or registry is edited. Latest data provenance/exposure checks are owned
by selection; known source reuse and any training exposure constrain transfer claims.
