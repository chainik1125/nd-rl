---
written_on: 2026-10-06
written_by: agent:codex
status: running
---

# Sprint kickoff — learning to select proof attempts and stop

## Time and objective

- Start: 2026-10-06 01:39 UTC.
- Deadline: 2026-10-06 11:39 UTC (10 hours of wall-clock time).
- Writing-only block: 2026-10-06 10:39–11:39 UTC.
- Objective: test whether a model initialized from Dan's latest expert-iteration proof model can recover the gap between single-attempt success and oracle pass@k by learning to select its own attempts and decide when to stop.

The shared design is [the proposal](../../../../proposals/learned-compute-allocation/README.md). Its three experiments start independently from the same base checkpoint. Begin with selection from eight frozen-generator rollouts, then test cost-aware stopping and, if the environment exposes the phenomenon, stopping under correctness-only reward. Format differences between the proposal and Dan's proof-state model must be audited before training.

## Hypotheses and falsifiers

| Hypothesis | Prediction | Falsifier or narrowing observation |
|---|---|---|
| H1: learned recognition recovers useful oracle headroom | Held-out selected-proof accuracy exceeds the strongest non-oracle baseline on identical candidate pools | No reproducible improvement; apparent gain is explained by leaked verifier outputs, malformed-proof cues, or overlapping theorems |
| H2: learned stopping improves allocation with attempt costs | Higher held-out correctness-minus-cost than development-selected fixed counts and compute-matched mixtures | Gains disappear under fair selection and compute matching, or actions reduce to a fixed counter |
| H3: harmful accumulated context supports correctness-only stopping | Continuation has negative measured value before hard limits and acceptance tracks that value | No degradation, only truncation, or acceptance does not predict continuation harm |

Keep observed results, interpretations, and post-hoc hypotheses separate. Negative outcomes are valid sprint findings.

## Priority and evaluation

1. Audit checkpoint lineage, tokenizer, proof-state feedback, verifier, data exposure, and available context; reproduce a small baseline.
2. Generate fresh theorem-level train/development/test splits with structural overlap checks; freeze test choices before opening outcomes.
3. Run the smallest end-to-end learned-selector experiment with shuffled candidates and fair baselines; scale and replicate only after the audit.
4. Run an independently initialized sequential attempt controller; distinguish a frozen-generator controller from joint generator/controller training explicitly.
5. Test context harm and correctness-only stopping only when protocol and capacity support an interpretable test.

Do not repurpose protected final evaluations as development data. Existing published benchmark numbers are prior evidence, not newly reproduced results. Primary comparisons use identical theorem pools, paired theorem-level uncertainty, and independent training seeds when feasible. A reduced pilot must be called a pilot rather than completion of the full proposal.

## Repository and team

- Repository: `/data/users/dmitry/ndrl`.
- Branch: `dmitry/main`; base commit: `315f36266bb4dd7feeff87228cb2662f8a36badc`.
- Keep edits in new sprint-specific paths on the current branch; no pushes or unrelated branch changes.
- Preexisting untracked directory: `code/experiments/current/proof-state/`. Preserve it; inspect read-only where useful.
- Agents: two `gpt-6.1-sol` researchers (selection/data; sequential allocation), one `gpt-6-astra` reviewer, root responsible for synthesis and coordination.
- Required instructions: repository `AGENTS.md`, the proposal, and `/data/users/dmitry/simplex-research/.agents/skills/run-research-sprint/SKILL.md` plus its four references.
- The documented `docs/NOTE_TEMPLATE.md` has moved to `docs/project_strategy/NOTE_TEMPLATE.md`; use that existing template's conventions.

## Compute, budget, and external actions

- Use existing local CPUs and available H200s after checking occupancy and claims. Coordinate devices centrally; never stop another user's process.
- Additional paid-compute cap: $0; reserve: $0. Do not launch Modal/RunPod or paid APIs under this kickoff.
- Reading/downloading source and checkpoints is in scope. Keep secrets out of outputs.
- Push commits, publish reports, open PRs, and upload model/data artifacts: not authorized by this kickoff. Preserve artifacts locally and record a proposed durable destination.
- Stop only processes launched by this sprint. Record PIDs, commands, configurations, expected durations, and recovery paths.

## Deliverables and completion

- `research_log.md`: timestamped decisions, hypotheses, dead ends, resources, provenance, and elapsed/remaining time.
- `manifest.json`: machine-readable sprint state and artifact index; each run gets its own immutable configuration/results.
- Separate experiment working folders and corresponding summary folders, linked from the sprint report.
- Reader-facing [summary](../../../../experiment-summaries/2026-10-06-learned-compute-allocation/summary.md), quantitative figures, uncertainty, limitations, and the best next experiment.
- Relevant protocol/regression tests and code checks, including unresolved failures.
- Astra review of the design, load-bearing evidence, and final report; correct concrete overclaims or implementation failures.

Work for the requested wall-clock duration, pivoting toward informative tests as evidence arrives. Reserve the last hour for report verification and review. Completion requires an evidence-backed report and accounted-for resources; merely starting jobs is not completion. If a proposal component cannot be tested, state why and leave a concrete next step.
