---
written_on: 2026-10-06
written_by: agent:codex
status: prespecified_secondary_test_method_pending_root_registry
---

# Frozen-trunk secondary evaluation

Root authorized this secondary final evaluation before any MAIN TEST or historical benchmark outcomes. Apply the one terminal head from [the development method freeze](frozen_head_method_freeze.md), without refitting, tuning or checkpoint selection. Head SHA256 is `d15adf65253377f3505cede44c9ba68c27b70a29321de84a4059c5ca5285299f`; original r16 checkpoint SHA256 is `ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24`. The original trunk and language-model head parameter digest is `594173e7d67a64f5c4ffa9ac5176e4390770e44ce7d307590474bed3957c8eb5` before and after feature extraction.

Use [evaluate_frozen_head.py](evaluate_frozen_head.py), with context8192, full public transition view, canonical candidate IDs and batch8. Extract the final LayerNorm vector at the factorized final `<act>` in BF16 model autocast; retain the FP32 vector and apply the saved scalar head in CPU FP64. No normalization, likelihood replay, new objective or policy-head mutation is permitted. For a candidate exceeding context8192, retain it with independent neutral logit0/probability0.5 and no model-forward charge. Report candidate and theorem overflow counts. This rule was authorized before TEST outcomes and aligns with the factorized BCE protocol; it does not use future candidates or verifier labels.

The fresh online evaluator will be checked on development against the original feature cache before signatures are frozen. Test guards validate model, source and sealed public-pool registration before inference or label reads. Separate prediction configuration contains provenance only; evaluated accuracy/log-loss summaries live in a separate post-guard summary file. The development-cache option is forbidden on TEST.

[compare_frozen_head.py](compare_frozen_head.py) reports the matched fully fine-tuned BCE seed0 comparator, and separately the fixed three-seed BCE mean within theorem. It also reports random and the frozen full public-feature predictor. No winner is selected. Theorem-level paired bootstrap preserves family proportions,5000 draws, seed61006, conditioning on the fixed head/checkpoints. Report all family, n-valid, completion and mixed-complete strata.

[frozen_head_thresholds.py](frozen_head_thresholds.py) uses the unchanged root `analysis/threshold_baselines.py` grid, costs, exact candidate-order/tie expectation, reward tie-breaking and development-fixed mixed-count controls. Its terminal policy is calibrated on existing DEV probabilities and frozen before any final outcomes. Charge text-scoring input tokens, no likelihood/entropy replay, and exempt fixed1 from scoring its sole candidate. These are fixed-pool counterfactual policies with additive unpadded cost proxies, not measured online throughput. This secondary result remains separate from the existing verified primary three-seed BCE staged aggregation.

Optimizer limitations remain part of the report: the fixed L-BFGS solve terminated before its iteration/evaluation limits with objective0.2820167704 and max-gradient2.02e−6; gradient tolerance1e−7 was not met. The exact internal stopping branch was not recorded. No remedial refit is permitted.

Root owns both MAIN and historical final registries. This note records decisions; it does not authorize sampling, sealing or TEST access by itself. All method, inference, threshold-policy and statistical signatures must enter the appropriate registry before their final use.
