---
written_on: 2026-10-06
written_by: agent:codex
---

# Cached confidence DEV diagnostic review

Accepted as a bounded numerical engineering diagnostic, with the precision and evidence qualifications below. No registered confidence feature, policy, replay charge or scoring method changes follow. No model inference or verifier run was launched by the reviewer; no MAIN/HIST TEST outcomes or diagnostic validity labels were read.

[Independent CPU helper](audit_cached_confidence.py) and [results](cached_confidence_independent_audit.json) reconstruct all1871 saved raw action prefixes through the public environment, all128 candidate sums and normalized aggregates, and unchanged feature coefficients applied to reconstructed candidates. Candidate sums match exactly; feature scores agree within7.11e−15. All16 full-pool argmax tie sets and128 crossings of threshold0.85 agree. The maximum reported score discrepancies are meaningful: matching decisions on this panel is not numerical equality or a general substitution guarantee.

Source inspection confirms a single untouched original sampler call per callback, no new randomness, original-slot attribution, suppression after first sampled EOS, retained sampled PAD, early=eos, and separate temperature0.8 behavior versus temperature1 base statistics. Each generated action asserts cached-event token and timestep equality with its trimmed raw output. Saved original/instrumented identity arrays agree in both secondary BF16 and FP32 checks. Independent CPU reconstruction confirms each recorded prefix, termination boundary, action count and aggregate token denominator. The empirical pilot contains no sampled PAD; a newly executed synthetic callback exercise independently reproduces retained PAD, dead-row suppression and noncontiguous original slots.

The BF16 maximum candidate-average differences reproduce: .0026757292 behavior log probability, .0020499498 base log probability and .0005599771 base entropy. Cached Python sums use FP64 accumulation of FP32 token values whereas replay reduces each action in FP32. The experiment therefore measures the combined cached/full-forward and reduction discrepancy; it does not isolate their contributions.

The eight-row original-r16 FP32 initial-action comparison reproduces maxima1.5959e−7,1.5593e−7 and1.1973e−7. It does not cover complete trajectories or adapted/joint-live checkpoints. The saved source documents BF16 autocast or disabled autocast, but completed processes did not capture runtime float32 matmul precision or TF32 flags. “FP32” must not be interpreted as a strict IEEE arithmetic or bit-exact guarantee, and a new process cannot retrospectively measure those settings.

Cached per-token event/logit arrays were not persisted for the full pilot. Their identity and accounting are supported by the executed source assertions and saved raw token identity controls, not an independent reconstruction from unavailable token logits. The independent numerical audit starts from saved per-action sums. This limitation does not invalidate the arithmetic comparison, but bounds its evidential granularity.

Accepted claim: cached confidence is numerically close but not identical to the full BF16 replay on this small outcome-free DEV panel, and the tested frozen feature decisions agree there. Rejected extrapolations: general decision equivalence, calibrated correctness, production substitution, live-checkpoint equivalence, or measured speedup. There was no prospectively justified equivalence margin. The already disclosed main()-scope timing excludes import startup and includes callback synchronization; it is not an isolated performance benchmark. Registered replay costs remain authoritative.

The helper passed Ruff and syntax checks after formatting; the CPU arithmetic and synthetic bookkeeping checks were actually executed. No scientific source was edited.
