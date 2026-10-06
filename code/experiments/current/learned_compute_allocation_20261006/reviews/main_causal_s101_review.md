---
written_on: 2026-10-06
written_by: agent:codex
status: main_development_artifact_audit_passed
---

# First main zero-cost continuation probe

Independent checks of `causal_main_nocost_s101_one_extra_v1` reproduce every saved selection-value decomposition across **128 states and 1,280 branches**, and every pooled mean and stratified bootstrap interval, with zero arithmetic discrepancy. The panel contains 64 development theorems, 32 per family, with both requested states eligible. Panel, checkpoint, and saved runner hashes match.

A deterministic sample uses panel positions 0, 1, 32, and 33, both attempt counts, and continuation replicate zero in both history modes where present. **20 CPU controller evaluations** agree with saved GPU probabilities within `1.19e-7`. Independently rechecking **36 literal public proof renderings** with Lean, including a deterministic supplement of valid and invalid proofs, yields zero label discrepancies. This sample verifies recorded evaluations; it is not a new performance estimate.

The observed one-extra correctness change is approximately −.011515. Valid probability mass on retained proofs falls from .904143 to .879164; the new proof contributes .013464. This is selection dilution despite added candidates, not evidence that oracle proof availability declines. Future-proof validity is about .520508 under actual history; the first-16 memoryless contrast improves new-proof validity and reduces the correctness loss. These observations do not alone establish a semantic interference mechanism. The count-plus-incumbent-validity residual acceptance/effect correlation independently reproduces .291773, which has the opposite sign from “accept more when continuation is worse.” It remains an exploratory association.

The negative effect is specifically the expectation under the native **categorical index readout**. It does not establish inevitable harm under greedy deployment. The separately reported greedy readout can retain first-proof correctness by continuing to select the first proof at larger budgets; such a flat result also does not establish useful selection among additional proofs. Keep native categorical, greedy, and oracle-availability results separate. No additional causal readout was introduced after these outcomes.

The larger first-proof correctness than the separate DEV32 path evaluation is not a paired-protocol contradiction. Both use checkpoint SHA256 `cf77abef4d31ec0ca2df52af37f06497e094d9341d079aba98c47a927238ae38`. The causal panel has **58/64** first successes; its overlapping DEV32 theorems have **27/32**, and its added 32 have **31/32**. The separate path evaluation has **25/32**. Panel composition explains .0625 of the difference; the overlapping panel differs by another .0625 under independent draws. The causal seed is 71006 with keyed incumbent seeds, versus path seed 52006 and its path schedule. Do not compare these as identical generated proofs or attribute the difference to an intervention.

Evidence: [reproduction and sample script](audit_main_causal_s101.py), [complete audit results and sample IDs](main_causal_s101_independent_audit.json). Only development outcomes and CPU/Lean checks were used; no GPU job, final-test outcome, or protected benchmark was accessed.
