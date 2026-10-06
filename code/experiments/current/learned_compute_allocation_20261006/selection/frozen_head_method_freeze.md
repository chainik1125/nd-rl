---
written_on: 2026-10-06
written_by: agent:codex
status: frozen_secondary_development_method
---

# Frozen proof representation with a separate scalar critic

This secondary control was declared after the selected fully fine-tuned binary critic retained only 29/128 clean development generation successes, versus the matched original checkpoint's 93/128. It is not a change to either primary selector, and its development outcome will not choose another architecture or hyperparameter. No final test outcomes have been opened.

Freeze the entire original r16 trunk and original language-model head, starting independently from checkpoint SHA256 `ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24`. Extract the 384-dimensional final LayerNorm state at the last `<act>` position of each public factorized transition input, exactly matching `train_bce.py:model_inputs`. Use the existing schema4 main train/development pools, their immutable canonical candidate IDs, context 8192, and BF16 model autocast. Cache FP32 feature vectors without gradients, with model/source/data hashes and measured compute. Do not standardize features; the frozen final LayerNorm already normalizes each representation. No validity, completion, failure reason, likelihood/entropy replay or constructed reference enters extraction.

Fit one new 385-parameter scalar head, consisting of 384 weights and one bias. Initialize every parameter to zero. Minimize mean unweighted binary cross entropy on all 12800 training candidate validity labels, plus `0.5 * 1e-3 * sum(weight**2)`; the intercept is unpenalized. Use deterministic full-batch L-BFGS, learning rate 1, history 100, maximum 500 iterations and 625 function evaluations, gradient tolerance 1e-7, change tolerance 1e-9, and strong-Wolfe line search. Solve in CPU FP64 from the cached FP32 vectors for numerical stability; retain FP64 coefficients and scalar readout. This precision choice is fixed before extraction/fit. There is no hyperparameter sweep, feature normalization, development-based early stopping or development-based checkpoint choice. Save the one terminal optimizer result and convergence diagnostics even if the iteration/tolerance limit is reached.

The generator checkpoint remains an immutable file and all loaded original parameters have `requires_grad=False`. The new head is separate; it never replaces the original LM output head. Record a pre/post parameter identity digest. Generation preservation follows from exact unchanged original parameters and unchanged generation code; the added readout has no effect on ordinary LM decoding.

Report development candidate log loss/Brier and k8 exact-tie selection accuracy, stratified by family and completed/mixed-validity pools. Compare with the already chosen fully fine-tuned BCE seed 0 and the frozen full public-feature critic, without choosing favorable subsets. The existing original n1-minus-n2 factorized readout is an additional descriptive initialization reference. This is a linear representation probe with supervised verifier labels, not zero-shot recognition or a claim about nonlinear recoverability.

Estimated resource cost before measurement: less than two GPU minutes for one feature pass over train/development and less than one CPU minute for the scalar fit/evaluation. Record actual monotonic timings and batch forward/attention-position counts. This control uses a different optimizer and parameter budget from full fine-tuning; comparisons are diagnostic. A final-test extension requires registering this method, chosen head/source hashes and inference rule in the final freeze before any test outcomes, if the coordinator decides to include it. Existing frozen trainer/evaluator signatures remain untouched.
