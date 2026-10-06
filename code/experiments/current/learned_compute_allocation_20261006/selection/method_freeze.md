---
written_on: 2026-10-06
written_by: agent:codex
status: frozen
---

# Frozen learned-selector methods

The coordinator approved this method configuration at approximately 02:31 UTC, before any main selector training or final test candidate sampling. The main theorem generator mixture was frozen earlier in `data/main_v1/manifest.json`. No final test outcomes have been opened.

The primary method is an eight-proof pooled chooser. An independent copy of the original r16 ALiBiGPT receives the theorem, eight randomly ordered candidate transcripts, and an index prediction position. Its original vocabulary logits for `n1` through `n8` become an eight-way softmax. The control is a factorized ranker: the same original model independently receives each theorem/proof pair and produces a scalar from its vocabulary logit difference `n1` minus `n2`; the eight scalars receive a pool softmax. There are no extra head parameters. Each method starts independently from the original checkpoint for seeds 0, 1 and 2. Neither architecture is chosen after viewing test results; report both, with pooled selection designated primary.

Both methods use the full public transition transcript, replayed deterministically from raw sampled actions. Every environment text delta is preserved, including implicit case closures and branch binders. A failed EOS-ended action retains deterministic normalized attempted text; a truncated action retains raw output. No verifier verdict, failure reason, private completion flag or reference proof enters the model. Effective explicit-action traces remain an auxiliary representation ablation because they can omit necessary implicit binders. Exact completed transcript identity and all-attempt direct Lean agreement were established on the main development pools before training.

The objective is negative log probability mass assigned to any valid candidate. All-invalid pools receive no optimizer update but remain in evaluation. All-valid pools have zero data gradient, so training reports their count separately from mixed-validity pools. For each run: five epochs, learning rate 0.00003, AdamW weight decay 0.01, gradient norm clipping at 1, batch size two theorem pools, BF16 model autocast with FP32 readout/loss, shuffled training candidate order and a fixed evaluation permutation seed 77106. Every epoch is saved. Select each run's checkpoint by highest development exact uniform-argmax-tie expected correctness, breaking equal epoch scores by the earliest epoch.

The forward context cap is 8192 for both architectures. This explicitly extrapolates beyond the checkpoint configuration of 1024. Input text is never truncated. An overflow pool receives uniform candidate probabilities and a seeded uniform argmax choice; exact expected correctness includes it. Non-overflow exact score ties also receive seeded uniform choices, and reporting includes exact uniform-tie correctness. The factorized control's smaller per-forward input sizes will be reported, including the portion within the original 1024-token support.

Executed forward calls and padded forward tokens are measured separately for training and evaluation. These counts exclude backward FLOPs, tokenizer/replay CPU work and private verification. Generator collection additionally reports executed batched sampling and diagnostic likelihood replay counts; per-candidate generation token steps are lower bounds on batch execution, not exact aggregate compute.

The inference boundary remains a new format relative to r16 training. Recognizing generated candidates after adaptation is the experimental question; these methods do not establish that the original checkpoint natively understands eight-proof lists. Claims require the frozen test pools, strong syntax/likelihood/entropy/feature baselines, theorem-level paired uncertainty, independent seeds and separate family results.

Before main training, the coordinator also approved supplementary evaluation permutation seeds 77107 and 77108. The primary seed remains 77106. Report accuracy at all three orders and their mean/range for every frozen checkpoint; never choose a permutation or architecture using test outcomes. Factorized controls additionally receive a canonical-probability equivariance check with explicit floating-point tolerances and exact tie-expected correctness.
