---
written_on: 2026-10-06
written_by: agent:codex
status: implementation
---

# Sequential allocation interface audit

The actual sprint base is Dan's `la_T1_best12_s0_r16.pt`, SHA256
`ba5d03de5dba890be643da82f9edca826d254c862cdcd33c221954b70918da24`,
loaded from clean source commit `4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c`.
It is ALiBiGPT, six layers of width 384, with the 111-token `lean_staten` tokenizer.
The existing untracked `proof-state` source is older and remains read-only.

The proof-state environment computes focused hypotheses/goals and applies grammar and
scope constraints. It does not typecheck atomic proof terms; Lean verifies completed
proofs. Grammar acceptance nevertheless shapes the generated trajectory. The original
sampler resets `Env` for every attempt; its history option carries only within-attempt
history. Across-attempt conditioning must therefore be introduced explicitly.

The sampler's `LEANREJ` and `LEANPARSE` strings reveal checker/parse outcomes and must
never enter controller prompts. Raw model outputs can also contain introduction names
that the environment overwrites under `lean_staten`; the controller must see coherent
executed action text, with the final attempted malformed action retained, while raw
emissions and their probabilities remain available for accounting. Metadata fields
such as `valid`, `ended_reason`, and `parse_complete` are private measurements.

## First pilot

A trainable copy of the exact same base processes the theorem and numbered effective
proof-token histories. New zero-initialized heads factorize a Bernoulli acceptance
choice and a categorical prior-attempt index. Cost and no-cost conditions load the
base independently; neither inherits the experiment-1 selector. Uniform initialization
is independent of correctness and context occupancy. Full-episode REINFORCE uses final
selected-proof correctness minus optional additional-attempt cost. It sums decision
log probabilities within each episode and averages over episodes, with an independently
sampled leave-one-out reward baseline. There is no supervised stopping, entropy reward,
KL penalty, discount, or weight decay.

This first pilot freezes independent proof generation. It therefore tests learned
selection and allocation only; it cannot establish joint proof generation/controller
RL or accumulated-context harm in generation. Additional candidates cannot reduce
oracle availability. Any deterioration is selection-relative.

Context admission uses a fixed prospective per-attempt bound independent of future
sample text. The default checkpoint configuration states 1024, but ALiBi admits longer
inputs. Main development occupancy audits motivate an explicit 16384-token context,
with 1024 emitted tokens per attempt and 1024 prospective state reserve. This extends
the base checkpoint's nominal context configuration and must be reported.
No history is silently clipped. Forced acceptance contributes selection probability
but no stopping probability/gradient. Exact candidate IDs retrieve the original stored
proof; verifier labels are accessed only after a sampled final selection.

## Verification so far

Fourteen meaningful protocol and likelihood tests pass, including exact retrieval against
the original theorem, private-field isolation, no correctness-triggered termination,
forced-limit inference, attempt/decision alternation, context reservation, and zero
forced-stop gradient. Targeted Ruff and shared-code Pyright pass. Full-repository checks
are blocked by thousands of preexisting Ruff findings in the unrelated untracked
proof-state source; no fixes were made there.

## Live generation and remaining interface confounds

The live runner now scores all raw generation tokens plus voluntary stopping and
selection decisions in a complete-episode outcome-only REINFORCE loss. Forced stop
decisions retain only selection likelihood. Independent format adaptation uses
teacher-distribution KL without private outcomes or stop targets; acceptance and
selection heads start uniformly after adaptation.

Format replay currently supports zero, one, and four earlier frozen-generator proof
histories. Actual eight-attempt deployment can contain up to seven earlier proofs
sampled from the adapted or subsequently trained model. Both the unsupported counts
and the changed history distribution are remaining interface confounds. Development
full-path audits show substantial generation decay after several histories even when
first-attempt quality is retained. Early stopping alone cannot distinguish avoidance
of this interface shift from recognition of reasoning completeness.

The full-path evaluator couples all stopping methods on one generated contextual
path and integrates stop and index probabilities exactly. Every private proof outcome
is checked after path generation. Expected policy work and the actual expense of
measuring eight-attempt paths are separate quantities. Fixed-count and count-only
hazard controls remain necessary before attributing benefits to proof-aware stopping.

Earlier proofs remain selectable throughout a live episode, so additional attempts
cannot reduce oracle availability. A drop in realized correctness can arise from
categorical selection dilution or changing scores for old candidates. Negative
continuation value must therefore be identified relative to the frozen learned
readout and separated from new-proof quality. On the chosen initialization's small
development panel, oracle availability rises only from 0.78125 after one attempt to
0.8125 after three through eight, while individual new-proof quality falls. This is
an interface-limited development result, not a general claim that more reasoning
reduces available correct answers.
