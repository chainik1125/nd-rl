---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - huScalingLLMReinforcement2025
---

# ProRL v2 (NVIDIA blog): a longer recipe, no checkable boundary numbers

Paper: [@huScalingLLMReinforcement2025] (proposed key; Jian Hu et al., NVIDIA Technical Blog, 2025-08-13)
Source: `nvidia-devblog-prorl-v2.txt` (developer.nvidia.com, fetched 2026-10-19), `nvidia-prorlv2-blog-jina.txt` (research.nvidia.com/labs/lpr/prorlv2, truncated render) and the HF model card (`PR_hf-nemotron-research-reasoning-qwen-1.5b-README.txt`, fetched 2026-11-07). Blog figures are images; only captions are in the text.

## Setup card

| | ProRL v2 |
|---|---|
| Model, start | 1.5B; base DeepSeek-R1-Distill-Qwen-1.5B (distilled) [PR-125]; v2 scales v1's steps "from 2000 to 3000" (model card) [PR-145] |
| RL share of pretraining | 4.4 × 10⁻³ – 2.1 × 10⁻² FLOPs (rl-compute-share row "ProRL v2") |
| RL steps | > 3,000, five domains [PR-79, PR-86] |
| Largest k | not stated in text [PR-87] |
| Grading | programmatic verifiers only [PR-126]; no hand check |
| Controls | symmetric budgets no; format no; guessing no; contamination no (none described) |

## Learnings

**Observed (what the text states).**
- Algorithm changes from v1: REINFORCE++ baseline (group-mean subtraction, then batch normalisation) instead of GRPO's group standardisation [PR-80]; clip bounds 0.20 / 0.28 [PR-81] (v1: 0.2 / 0.4 [PR-6]); KL kept, with a k2 estimator [PR-83]; a cyclic cosine length penalty (e.g. 100 updates on, 500 off) [PR-84].
- Reset schedule: every 200–500 steps or on KL spikes / stalled validation, reference reset to the current policy, optimizer state not cleared [PR-82] (v1 reset both, on validation stalls only [PR-4, PR-5]).
- Context: v2 trained at 8K, v1 described as 2,000 steps at 16K [PR-85, PR-86]; the model was "still undergoing continuous training" [PR-90].
- Boundary claims are stated without numbers: pass@1 and pass@k "climb over thousands of RL steps" [PR-87]; "on tasks where base models always failed" ProRL achieves strong pass rates [PR-88, PR-91]; lower n-gram overlap with pretraining data [PR-92]. No k, task list or table appears in the text.
- IF: the blog mentions IFEVAL only in the Figure 1 caption, metric unstated [PR-89]. The model card gives v2 IFEval 70.85 against v1's 66.02, both average scores as in v1 Table 3 [PR-147, PR-24]. No IF pass@k anywhere.
- A "ProRL V2.1 Update — February 9, 2026" exists [PR-93]; its content could not be rendered (jina and WebFetch returned only the header): UNVERIFIED.

**Paper interprets.** Steps, resets and length scheduling sustain improvement, and the improvement "expands the base model's reasoning boundary" [PR-87].

**Our interpretation.** The blog adds a recipe (fixed-interval resets, no optimizer clearing) but no checkable boundary evidence: every boundary statement is either a figure we cannot read in text or a restatement of v1's claim. It is the start checkpoint of BroRL ([hu2025-brorl](hu2025-brorl.md)), whose authors say it had plateaued [PR-127].

## Evidence and limitations

Both blog files read in full (the jina file is a 35-line truncated render). Pass@k figures not inspected (images). No seeds or error bars in the text.

## Connections and questions

- C3, C5: **unrelated**: no IF pass@k; IFEval only in a caption with an unstated metric [PR-89].
- C7: **unrelated**: no k or controls to compare [PR-87].
- Q2: v2 keeps KL with resets every 200–500 steps; Olmo drops KL [PR-137], so ProRL's "KL + resets" lever has no Olmo counterpart.
