---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# InternLM2.5-StepProver: critic-guided best-first search, preference-trained critic, large-scale EI
Paper: InternLM2.5-StepProver: Advancing Automated Theorem Proving via Critic-Guided Search, Wu et al., 2024
Source: https://arxiv.org/abs/2410.15700v2
## Learnings
- Baseline BF (§2.1): expand states by average tactic log-prob. Authors: "using best-first-search with log-probability scores seldom leads to deep proofs".
- Critic-guided (CG) search (§2.1): a separate critic V(s) ∈ ℝ picks the state to expand. It is trained as a preference/reward model, "instead of binary targets", on (i) path pairs (a state closer to no_goals beats its ancestor; up to n-choose-2 pairs per path) and (ii) sibling pairs (on-path state beats an off-path sibling).
- Critic: InternLM2-Chat-1.8B, 454K pairs; pair accuracy on 6510 miniF2F-test pairs "78.0%" (App. A).
- miniF2F-test (Table 1, S = 32, K = 600): BF 47.3 / 57.3 / 59.4% at 1 / 16 / 256 passes; CG 43.0 / 61.7 / 65.6%; BF+CG 65.9% at 256. CG is *worse* at 1 pass: "with low sampling budgets, CG may overlook simpler proofs that are trivial for BF" (§3.1). ProofNet: BF 22.3, CG 23.9, combined 27.0 (§3.1).
- Proof length (§3.2): "The average length of solutions found by the BF method is 1.66, whereas the same indicator is 4.44 for the CG method"; Fig. 3 says CG finds proofs with >9 tactics.
- EI (§2.2): ~21,364 CPU days; 17.0% of Lean-Workbook-plus proved or disproved; "only about 1.5% of CPU resources are used to solve 17.0% of the problems" (Table 4 discussion). After each round they re-search solved problems for shorter proofs, and the critic ranks unsolved statements (the top 50% are kept).
## Evidence and limitations
- The critic adds a forward pass per state; BFS-Prover notes it "effectively doubles the number of inference calls", so CG vs BF in Table 1 is not compute-matched.
- No ablation of search-expert vs sampling-expert EI, and no measurement of whether the trained prover improved because of CG data vs BF data.
- The authors say they "do not have a stable metric to measure critic models" (Limitations).
## Connections and questions
- B1: the strongest published claim that a value/critic beats pure log-prob best-first on *proof length*. It supports adding a value to any step-env search aimed at L*.
- Preference-style critic training needs only successful paths plus siblings, both already produced by a step-env search. Path pairs amount to "closer to the goal is better", i.e. a learned steps-to-go ranking; compare Polu 2022's proofsize buckets and AlphaProof's −steps value.
- Smallest test: train a small value head on path + sibling pairs from state-env EI data; compare CG vs log-prob BF at equal *model calls* (count critic calls) on the long pool's bins 13–16.
