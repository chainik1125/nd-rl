---
written_on: 2026-10-06
written_by: agent:codex
status: evaluation_running
---

# Transfer of trained proof-generation and retry policies

This evaluation uses existing checkpoints without training or tuning. The six
joint cost-aware/correctness-only models retain their original development-selected
weights. A model trained to use four attempts is the fixed-budget control; a
fixed-one trained model and format-only initialization are supplementary controls.

Audited results on the reused 72-theorem textbook benchmark are retained unchanged.
On Charles's pinned 2026-10-02 release, 187 ordinary theorem rows form the primary
transfer benchmark. The 225 derived rows are evaluated separately as sequent-only
projections: given proof prefixes, task instructions and structural constraints are
omitted because this model's interface does not implement those tasks. A projected
proof score is not native performance on the derived suite.

Each checkpoint receives two bounded contextual paths per row. We report first-proof
correctness, whether a correct proof exists among the first four/eight attempts,
native selected correctness and expected attempts, and selected correctness at fixed
budgets. Earlier attempts condition later generation; oracle availability is not an
iid pass@k estimate or an attainable selection rule. The fixed-four model's primary
readout uses four attempts; its eight-attempt readout is secondary. The correctness-only
policy may stop early and was not trained with an explicit coherence reward.

The [working protocol and implementation](../../../code/experiments/current/learned_compute_transfer_20261006/live/README.md)
bind model/data/source hashes, fixed sampling settings and uncertainty rules.
The [new frozen registry](../../../artifacts/learned_compute_transfer_20261006/live/protocol.json)
uses seed 95006 with two paths and preserves historical FP32 inference, temperature
0.8, and all generation/context limits. Three fixed training seeds are averaged in
primary arms; controls have one seed. Point estimates weight release rows equally;
intervals resample whole related base-problem clusters and condition on these models.
No pooled 412-row native-task result will be reported.

New live evaluations are running. Existing historical findings and their limitations
remain available in the [audited allocation summaries](../../2026-10-06-learned-compute-allocation/allocation/README.md).
