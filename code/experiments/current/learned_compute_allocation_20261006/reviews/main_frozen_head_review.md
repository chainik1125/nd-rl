---
written_on: 2026-10-06
written_by: agent:codex
---

# MAIN frozen-head secondary audit

Accepted `analysis/final_main/frozen_head_paired` and `frozen_head_thresholds`, with predictions from `selectors/final_main/frozen_head`. The [independent audit](audit_main_frozen_head.py) and [results](main_frozen_head_independent_audit.json) verify the registered head/base/extractor and analysis identities, prediction/input hashes, and exact reuse of the frozen DEV policy. All six paired accuracy comparisons reproduce from candidate logits and private labels. Cached FP32 vectors multiplied by the saved FP64 385-parameter head reproduce logits within 1.07e-14; this is a CPU arithmetic check, not a new trunk forward or training run.

The head achieves 0.84 selected-proof accuracy. Its prespecified matched BCE seed0 comparator achieves 0.8425: difference −0.0025, paired 95% family-bootstrap interval [−0.02, +0.015]. Against public features the difference is −0.035, interval [−0.055, −0.0175]. A nonsignificant head/BCE difference does not establish equivalence or noninferiority; neither a margin nor an equivalence test was specified. Keep the separately labeled three-seed BCE mean separate from the matched seed0 comparison.

Independently checked all 18,400 fixed/threshold/mixture policy rows using uniform-prefix subset enumeration and exact threshold inclusion probabilities. All summary arithmetic agrees within 2.33e-10 (roundoff on large token-squared work quantities). At the DEV-selected threshold 0.65 for cost 0.03, correctness is 0.8198511905, attempts 1.8292232143, logical model-input work 6,048.3525 and reward 0.794974494. Work comprises 5,195.6730 ideal generator input tokens and 852.6795 critic input tokens; no confidence replay is required and fixed1 avoids critic scoring. This reward being numerically close to BCE's three-seed staged reward is a point-estimate observation, not a new equivalence claim or reason to replace the primary comparison.

The saved original-trunk parameter digest before and after MAIN inference is `594173e7d67a64f5c4ffa9ac5176e4390770e44ce7d307590474bed3957c8eb5`, matching the independently checked original initialization and the original base checkpoint hash. The fitted head is separate; no original generator or language-model-head parameter is changed. Under an unchanged sampler/interface, this rules out weight-induced generator drift from fitting the scalar head. It does not demonstrate that every altered context/interface preserves generation quality, or generalize from the separate DEV diagnostic in which full-critic fine-tuning degraded generation. The latter is an observed diagnostic on a particular setup, not a measured MAIN frozen-head generation comparison.

No neural rerun, fitting, post-test method selection or historical outcome access occurred. The helper passes Ruff. No frozen experiment source was edited.
