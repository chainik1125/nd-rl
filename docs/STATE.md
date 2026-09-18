---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

The latest experiment code and artifacts are on simplex1 in `/data/users/dmitry/nd-rl/take-home/`, especially `window_sprint_20260906`, `transfer_sprint_20260907`, and `current_state_20260909`. The local earlier-iterations directory predates these experiments.

The [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md) completed its first independent pair audit. No endpoint expanded the observed proof-length window beyond six reference lines; the 73 final-only problems remained unopened in the local record. Next step: reconcile the remaining remote run state on simplex1 and decide whether to audit another frozen pair or close the sprint without opening the final-only split.

A [literature review for learning beyond initial training data](../astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.

The [novelty campaign](../experiment-summaries/2026-09-15-novelty-campaign/README.md) ([proposal](proposals/2026-09-15-novelty-campaign.md); take-home fork, branch `dan_novelty`) completed 2026-09-16 and was reviewed. Result: expert iteration produces depth-3 nested boxes at the 10%-coverage rate with zero depth-3 proofs in pretraining (frozen control and base sampling: none) — composition of a learned move; derived-ORE is amplification of a ≈10⁻³ base probability; the reductio dial was uninformative. Textbook transfer stays confined to schemata the base assigns ≥10⁻⁴. Next step: more seeds, a reductio pool whose double negation must be derived, and a cap-8 arm. The campaign tested whether RL creates or elicits capabilities: base-model log-probabilities and pass@10⁴ reachability of every RL-found proof, then pretraining sets with a proof pattern's frequency dialled from 0 to 10⁻¹ followed by identical expert iteration. It also records how the take-home run compares with the R1-CoT sprint: both found no textbook transfer, for what looks like the same reason (the generator's theorem-shape distribution).
An [automated research orchestration proposal](proposals/2026-09-16-automated-research-orchestration.md) (2026-09-16, design only) describes four agent roles (proposer, executor, reviewer, librarian), the gates between them, the hard constraints the reviewer enforces, and the infrastructure Dan would need to approve; its worked example is the ignition study on the `nd-takehome` fork.
[Five proposals](proposals/2026-09-17-proposals-round-2.md) follow the novelty campaign: cast ND into Lean and measure novelty by model scale (Dmitry's idea, with a Qwen3-Coder in-context v0), test the structural-vs-rule-sequence rule of thumb on new patterns, quantify the precursor effect, GRPO vs expert iteration at f = 0, and target pools that require a pattern.

Proposal 6 ([RL technique ladder](proposals/2026-09-17-rl-technique-ladder.md)) returns to the original L − P objective: true proof length from the bounded prover, a fixed transfer pool, equal sample budgets, and a leaderboard of RL techniques (expert-iteration variants, GRPO, preference training, search) with pre-registration per rung.
