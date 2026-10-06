---
written_on: 2026-10-06
written_by: agent:codex
status: presentation_check
---

# Reader check of staged stopping and causal DEV figures

Scope: I viewed only `analysis/main_test_staged_aggregate_v1/staged_policies.png` and `report_figures/causal_development_v2/continuation.png` before writing these observations. I did not open their metrics, rendering code or provenance files. I already have extensive sprint context, so this is not a fresh-context audit. No historical outcomes were read, and no scientific sources or metrics were changed.

## Staged stopping

The visual supports a clear descriptive reading: independent attempts improve fixed-budget accuracy; the feature scorer has higher accuracy than BCE at the same attempt counts, while the token-cost view changes their relative cost positions. Both adaptive stars appear above nearby fixed-budget points with roughly two mean attempts. The two panels help show why attempt count alone is an incomplete compute comparison.

The figure itself does not state the number of test theorems, family balance, BCE seed aggregation, or treatment of score ties. The legend's `cost 0.03` could mean money, token cost, or an attempt penalty. Dashed connecting segments could also suggest an achieved continuous frontier rather than discrete fixed policies. Without intervals, readers cannot assess uncertainty or claim a significant advantage from vertical differences. `Logical model input tokens, lower bound` correctly limits the cost claim, but does not explain which forwards are charged, whether padding is included, or why the scorers have different costs. The right panel omits its own legend; shared colors remain readable at this size.

Suggested caption, using prior protocol context: “Fresh MAIN TEST: 400 theorems, equally divided between the two frozen families. Circles show fixed budgets of 1, 2, 4 and 8 independently generated attempts; dashed segments connect discrete policies. Stars show stopping thresholds selected on DEV for reward accuracy minus 0.03 times additional attempts. Feature scoring includes required likelihood/entropy replay; BCE uses text scoring, averaged within theorem over three fixed training seeds. The right axis charges logical model inputs as lower bounds, not elapsed GPU time; fixed-one selection needs no scorer. Candidate-order and tie expectations follow the frozen protocol. These points are descriptive; paired uncertainty is reported separately.”

## Causal development

This presentation is substantially clearer than an internal-condition plot. The left panel explicitly separates loss of probability mass on existing valid proofs, positive mass on new valid proofs, and their negative net effect. The caption makes clear that old candidates remain available, so the negative term does not mean correct proofs were deleted. The right panel explicitly uses the same 16 theorems for every horizon, avoiding a misleading comparison between the original 64/32/16 panels. Labels explain the intervention rather than exposing internal suffixes.

A reader can infer a negative expected-correctness effect for every displayed condition, conditional on the fixed checkpoints and draws. The caption appropriately rules out inference about horizon-to-horizon differences. Remaining ambiguity is the reference for “accuracy change”: accepting/selecting now versus continued generation and selection. “Mass” is categorical selection probability on valid proofs, not the fraction of attempts that pass. This distinction could be stated once in the caption. The two left probability-mass terms share accuracy-change units because their sum is expected correctness, but that reasoning is not immediate for an unfamiliar reader.

Suggested caption addition: “Changes compare continuation with selecting from the candidates already available at the forced state. Correctness is expected under native categorical proof selection. The left decomposition sums the change in selection probability on existing valid proofs and probability on newly generated valid proofs; their sum is the net expected-correctness change.” Preserve the existing DEV label, equal checkpoint/count weighting, family-stratified conditional bootstrap statement, old-candidate availability and no horizon-difference testing caveat.
