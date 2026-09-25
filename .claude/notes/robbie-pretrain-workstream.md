---
written_on: 2026-09-22
written_by: agent:codex
---

# Robbie's workstream: pretraining ablations, week of September 21, 2026

## Summary

Source: Robbie's explicit scope clarification and pasted planning notes on September 22, 2026.
This is an agent-organized record of his instructions, not a new agent research proposal.

**Robbie is ablating pretraining only. Keep the initial dataset/data step and downstream RL
pipeline fixed across those ablations. Judge pretraining choices by downstream RL results,
not by pretraining loss alone.** Use autoresearch to hill-climb proof-length performance.

This clarification governs Robbie's idea backlog and weekly work. Older broad proposals and
agent-generated idea lists do not expand this scope. It does not redefine other people's
workstreams or authorize changing the frozen experiment harness.

## Scope and priorities

- Explore the pretraining setup and how the resulting model responds to the fixed RL pipeline.
  Architecture, positional encodings, looped transformers, optimizer, learning-rate schedule,
  initialization and pretraining objectives are candidate directions within the existing contract.
- Measure whether pretraining loss correlates with downstream RL success. Loss is a diagnostic;
  a lower loss alone is not the success criterion.
- Keep dataset generation, dataset selection/composition and the RL algorithm out of the
  ablation loop. Do not silently classify data augmentation, resampling or curricula as in scope
  just because they could be implemented in pretrain.py; settle that boundary with Robbie first.
- GPU speed work in `pretrain.py` is **encouraged** (Robbie, 2026-09-25, reversing the earlier
  "resist speed optimization" line): pretraining is wall-clock bound (300 s), so more steps per
  second is free compute, and faster models also sample faster in RL. Score, not throughput,
  still decides keeps.
- Build an idea backlog around pretraining mechanisms and predicted downstream RL effects.
  Data-generator projects, RL-recipe comparisons and new evaluation benchmarks are separate work.

## Open questions and inspiration

- **Tokenization:** potentially in scope; Robbie mentioned "try Lean" and noted that ownership
  may belong to Dan. This is an open boundary, not authorization to change the proof environment,
  verifier, dataset or RL pipeline.
- **Architecture leads:** positional encodings, looped transformers, and Robbie's note
  "deepseek hypermanifolds." Preserve the latter as a research lead to identify, not a verified
  paper title or technical claim. Read relevant DeepSeek and OLMo work for further inspiration.
- **Size/compute constraint:** the planning dump asks whether to fix model size, FLOPs, speed,
  or a combination of size and speed. The active loop already has concrete constraints in its
  harness; use those until Robbie deliberately revisits them. Do not reopen the budget per idea.
- **Interaction between stages:** independence of pretraining and RL is a hope/hypothesis, not
  an assumption established by evidence. Data and RL may also interact. Holding them fixed is
  how this workstream isolates pretraining effects; it does not prove the stages are independent.

## Baseline-selection context

The planning dump proposes reading Dmitry's work and spending two or three runs to choose a
strong baseline for the data/RL setup before starting the pretraining ablations. Robbie prefers
a starting point with enough successes to discriminate improvements: "20 new proofs up from 15"
is more informative for his purpose than "3 new proofs up from 2." These are illustrative numbers,
not measured results. This planning note does not instruct an agent to restart baseline selection.

The dump also lists a separate RL-pipeline choice for fixed pretraining/data: an off-the-shelf
system (Tinker was mentioned) versus building something internally ("PPO, GAPO" in the source
notes). These are baseline/other-workstream options, not additional variables for Robbie's loop.
The names and suitability of these options have not been checked here.

## Group and infrastructure context

As described by Robbie in the dump, Dan uses DigitalOcean VPSs for Claude Code, RunPod GPUs,
and a Hugging Face repository for model checkpoints. His orchestration includes proposal agents,
runners, preregistration/hypothesis gates, a librarian agent, and phone alerts. Experiments get
separate branches merged into a Dan branch; multiple VPSs host multiple agents. This records
reported context, not independently checked infrastructure or a requirement to copy it locally.

## Evidence and next steps

- The source of the workstream boundary is Robbie's message, not an inference from code.
- [Active loop instructions](../../code/experiments/current/autoresearch/program.md) and the
  [frozen harness](../../code/experiments/current/autoresearch/harness.py) specify the current
  execution protocol. This note records scope, not another copy of changing run constants.
- The [earlier constraints proposal](../../docs/proposals/2026-09-20-pretrain-workstream-constraints.md)
  is historical planning context. Its broader options are not blanket approval for this week.
- Next: develop a pretraining-only idea backlog, labeling hypotheses, expected RL effects,
  implementation cost, and any unresolved scope boundary. No new experiment is launched by
  writing this note.

## 2026-09-24: the fixed baseline moved to Dan's Lean format

On Robbie's instruction ("get Dan's best tokenizer and dataset, and Leon's best RL, plug them
in"), the frozen harness moved to fork `51604b3` with Dan's `lean_seq` tokenizer and a Lean 4 ∧
nd_verify reward, from autoresearch run 098. Dan's best cap-6 dataset is the `train.jsonl` the
loop already used, so data is unchanged. Leon had no RL work to adopt yet, so expert iteration
is unchanged. Data and RL stay fixed *within* the new protocol; scores from 000–097 are not
comparable. Details: `code/experiments/current/autoresearch/program.md`.
