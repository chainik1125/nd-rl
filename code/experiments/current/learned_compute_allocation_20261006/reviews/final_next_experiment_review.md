---
written_on: 2026-10-06
written_by: agent:codex
---

# Final next-experiment recommendation review

Accepted the additions to `selection/followup_stopping_design.md`. The earlier sprint-deferral judgment remains intact. This is a proposed, unrun sequence; no new experiment, inference, fit or endpoint was executed or inserted into the frozen study.

The cheap saved-DEV reranking diagnostic is correctly a prerequisite rather than a replacement for the independent-generation stopping study. It keeps the exact candidate pools and generation fixed, uses a prospectively chosen preserved critic, charges inference, and treats domain shift as a limitation. Recovering selection loss would not repair contextual generation or establish useful stopping; failure would limit the tested critic, not every recognizer. The main next experiment still freezes generator/ranking and tests a small stopping controller with isolated fitting and matched controls.

The calibration paragraph correctly distinguishes objectives from representational classes. A score-monotone count-dependent validity estimate followed by one posterior threshold can induce count-specific raw-score thresholds. Reward improvements from this mechanism remain valid allocation improvements; neither winning nor tying that calibration control establishes or refutes prediction of future improvement by itself. The paragraph leaves this as a later explanatory control rather than adding an unexecuted arm to the present study.

The controller-only scope is now explicit. A positive result would support useful allocation around an unchanged LM-generated-attempt system, not joint LM learning of generation/selection/stopping or cross-task meta-learning. The support/query adaptation stage is appropriately separate.

Before any future reranking execution, its concrete protocol should label native categorical versus greedy readouts separately and include a matching greedy comparison where interpreting recognition quality. Fixed1 requires no scoring for deployment; diagnostic scorer-forward expense should not be silently assigned to that no-choice baseline. These are ordinary implementation details to prespecify later, not a request to expand this completed sprint. No further prose correction is required now.
