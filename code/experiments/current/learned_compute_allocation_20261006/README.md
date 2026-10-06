---
written_on: 2026-10-06
written_by: agent:codex
---

# Learned proof selection and stopping

This working folder contains the 10-hour investigation of Dan's round-16 proof model,
conducted from 01:39 to 11:39 UTC on 2026-10-06. Start with the
[reader report](../../../../experiment-summaries/2026-10-06-learned-compute-allocation/summary.md).
The [kickoff](start.md) fixes the sprint contract; the [manifest](manifest.json) records
current execution status, and the [research log](research_log.md) records decisions.

## Experiments and interpretation

- [Selection method](selection/method_freeze.md): eight independent proofs, a pooled
  neural chooser, per-proof controls, and strong public-feature baselines.
- [Frozen-backbone head](selection/frozen_head_method_freeze.md): a small recognition
  readout that preserves the original generator parameters.
- [Joint allocation method](allocation/main_method.md): independently trained
  [cost-aware](allocation/cost/README.md) and
  [correctness-only](allocation/no_cost/README.md) generator/controllers.
- [Continuation test](context/causal_test_plan.md): acceptance now versus forced
  continuation, keeping all earlier proofs available.
- [Interpretation](stopping_interpretation.md), [proposal coverage](scope_and_limitations.md),
  and [hypotheses](hypotheses.md) distinguish observations from explanations.
- [Next stopping experiment](selection/followup_stopping_design.md) is proposed and unrun.
- [Cached-confidence diagnostic](selection/cached_confidence_note.md) is a small development
  comparison; registered replay charges remain unchanged.

## Evidence and provenance

Raw model outputs, fitted checkpoints, immutable registries, and numerical analyses live in
[the artifact folder](../../../../artifacts/learned_compute_allocation_20261006/).
The MAIN and historical protocols under `final_evaluation/` were sealed before final
inference. Their source/model/data bindings remain unchanged. The separate
`analysis_successors/fp32_probability_domain_v2/`,
`analysis_successors/historical_selection_roles_v2/`, `operational_amendment_v1/`, and
`operational_selection_amendment_v1/` addenda record a reviewed floating-point validity
guard, historical role-schema validation, and shared-GPU launch admission. The historical
successor changes the role assertion without changing any statistical estimator.
Original sources and registries remain available alongside the successors.

Independent numerical and implementation reviews are in [reviews/](reviews/).
The [resource audit](allocation/resource_accounting_review.md) distinguishes final process
timings from incomplete development sections; the [style note](reviews/final_style_exceptions.md)
records two preserved provenance-bound formatting exceptions.
Reader-facing figures are copied into the summary folder with provenance; presentation
renderers do not change estimates. Live and frozen-pool compute measurements have
different accounting scopes and must not be combined into a single performance frontier.

The [runtime snapshot](../../../../artifacts/learned_compute_allocation_20261006/runtime_environment_v1.json)
records Python, package, Lean and GPU versions. Run-local configurations, launch ledgers
and source copies record the actual commands and settings; frozen registries bind the
selected models and analyses. Independent replay helpers and their recorded outputs
are linked from each numerical review. These checks ran in the recorded workspace;
installation and rerunning on a clean machine have not been tested. Large checkpoints,
candidate pools and raw outputs remain local artifacts rather than committed files.

Work is based on branch `dmitry/main`, repository commit
`315f36266bb4dd7feeff87228cb2662f8a36badc`, and upstream proof-model source commit
`4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c`. Sprint changes remain uncommitted.
The source proposal is [Learning to select proofs and allocate attempts](../../../../proposals/learned-compute-allocation/README.md).
