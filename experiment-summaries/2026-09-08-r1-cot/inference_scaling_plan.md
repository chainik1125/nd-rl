---
written_on: 2026-09-08
written_by: agent:codex
---

# Predeclared evaluation-only inference scaling

**Prioritize verifier-triggered revision of the model's own failed draft, with shared first attempts and explicit replay costs.** Add bounded scratch continuation as a smaller ablation. Neither intervention changes weights, supplies a reference answer, nor yields an unmodified-policy sample. Intervention traces are ineligible for RL or replay training.

The primary inspiration is s1's decoding intervention: it extends thinking by suppressing termination and adding a continuation cue, and separately studies early termination. Its results also show saturation/repetition when continuation is pushed too far. Our symbolic `<REVISE>` cue is an adaptation, not a replication of natural-language “Wait.” [s1 paper](https://arxiv.org/html/2501.19393v1), [authors' implementation](https://github.com/simplescaling/s1).

R1 motivates measuring learned reasoning before the final answer, but increased length alone does not establish useful inference scaling. [R1 report](https://arxiv.org/html/2501.12948v1). Sequential revision and independent sampling should be compared under common budgets because their relative effectiveness can depend on difficulty. [Snell et al.](https://arxiv.org/html/2408.03314v1).

## Frozen comparisons

Use the same selected checkpoint, tokenizer, theorem population, temperature 0.8, and target-level seeds. Repeat with a second declared seed if time permits. Validate controller mechanics only on synthetic inputs before release. Final textbook checkpoints, configurations, and selectors freeze before any final-only inference.

| Condition | Intervention | Purpose |
|---|---|---|
| Independent sampling | Original prompt on every attempt; no token intervention | Parallel compute baseline |
| Larger allowance | Ordinary generation at 512, 1,024, and 1,536 response tokens | Distinguish truncation relief from learned continuation |
| Failed-draft revision | At most one retry using the previous failed final certificate | Test whether feedback/history beats an independent retry |
| Scratch continuation | Replace the first one or two proposed `</THINK>` tokens with `<REVISE>` | Test whether extra sequential work helps the correction model |

Use per-target charged allowances of 2,048 and 4,096 tokens for the repeated-attempt comparisons. The last attempt receives only the remaining allowance. Never silently exceed the checkpoint context window. Log the effective per-target cap, including reductions caused by prompt length or batch padding. Smaller-budget truncations remain failures. Retain the actual cost alongside every score; equal allowances do not guarantee equal consumption or FLOPs.

## Failed-draft revision, precisely

1. Generate an ordinary first attempt with cap 512. The revision and independent-retry conditions share this exact saved attempt. If it proves the original theorem, stop both conditions successfully.
2. A failure is revision-eligible only if its final suffix ends in one terminal QED, parses as a proof, and fails the strict checker. No gold proof or target-specific hint is consulted. An ineligible failure gets an independent retry in both conditions; retain it in the full denominator.
3. Start a fresh response to the **unchanged original prompt**. Require its first `<THINK>` to be model-generated. Then insert `<DRAFT>`, the failed final suffix's exact saved token IDs, `<ERROR>`, and `<REVISE>`. This matches the correction cold-start structure. The error marker means only that this actual draft failed; it supplies no corrected step.
4. Continue generation. The model must emit `</THINK>`, `<FINAL>`, and final QED itself. The strict checker judges only the new final suffix against the original theorem. No successful prefix, forced proof line, corrected citation, or rewritten theorem is accepted in its place.
5. Freeze a 1,024-token continuation cap, reduced if needed by remaining charged budget/context. Charge the initial attempt, copied draft, inserted markers, and repeated prompt. Compare with an independent retry charged under the same rule. Report the unconditional improvement and the paired difference on the shared initial-failure subset.

## Bounded continuation, precisely

For the 46-token correction model only, intercept a proposed `</THINK>` **before feeding it back**. For at most `r∈{1,2}` occurrences, replace it with the forced token `<REVISE>` and resume. Save both proposed and effective token streams. Do not insert `<ERROR>` here: no completed draft has yet been checked.

Stop overriding when scratch reaches 768 tokens or fewer than 512 context positions remain for a final proof. Stopping overrides merely restores ordinary sampling; it does not inject closing delimiters. If the model never closes its thought, the attempt fails at its response/context cap. Include `r=0` as the unchanged-policy control and an optional one-token `;` replacement as a cue-content control. No intervention is allowed after `<FINAL>`.

The 43-token planning model has not learned `<REVISE>`. Do not add that token at evaluation or map it to another ID. Its primary comparisons remain independent sampling and the larger ordinary token cap. Pure delimiter suppression could be a separately named later diagnostic, not this preregistered repair-cue treatment.

## Accounting, API, and reporting

The charged budget is **sampled token proposals + externally inserted/replayed tokens + prompt tokens on each new prefill**. A discarded proposed closing token still counts as a proposal; its replacement also counts as a forced token. Previously generated text copied into a new call is charged again. Maintain separate totals for accepted sampled tokens, discarded proposals, forced markers, copied draft tokens, prompt prefill, scratch, and final proof. Also log actual active/padded prefill and decode positions, wall time, and verifier calls. These expose token-count versus hardware-cost differences.

Minimal new API: an evaluation sampler with `(prompt, tokenizer_state, seed, max_new, injected_prefix, before_feed_hook)`; the hook can inspect a proposed token and return a replacement **only in scratch**. Each token/decision records its origin and any replaced proposal. An immutable controller manifest pins policy/codec/checker hashes, budgets, condition, and all seeds. Record `training_admissible=false`; forced actions have no behavior-policy log-probability. Base-model log-probabilities may be logged diagnostically and must not be treated as sampled-action likelihoods.

Report exact textbook solve counts by reference-length bin, shortest strictly verified compressed witnesses as upper bounds, and paired target-bootstrap differences at the two fixed allowances. Plot both the common allowance and actual charged consumption. Keep ordinary-policy curves and intervention curves visibly distinct. A controller gain is evidence for a **model-plus-controller** capability; a longer trace without additional verified held-out successes is not a positive result.

Scope: first run a 96-target synthetic pilot for eligibility, formatting, and throughput. Then evaluate only the frozen selected correction checkpoint plus its ordinary controls, reserving at most one hour across the two allocated GPUs for this secondary grid. Do not consume the final evaluation reserve by expanding the grid after seeing final results. This document is a proposal; no intervention sampler or GPU experiment was implemented for it, and no textbook contents were read for this subtask.
