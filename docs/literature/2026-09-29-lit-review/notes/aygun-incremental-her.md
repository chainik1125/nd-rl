---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Aygün et al.: hindsight experience replay for a clause-scoring prover, bootstrapped without a curriculum
Paper: Proving Theorems using Incremental Learning and Hindsight Experience Replay (Aygün, Orseau, Anand, Glorot, Firoiu, Zhang, Precup, Mourad; arXiv 2021; ICML 2022, PMLR 162, author list there also includes Mcaleer)
Source: https://arxiv.org/abs/2112.10664v1 (the only arXiv version; the PMLR version, https://proceedings.mlr.press/v162/aygun22a.html, was not read and per its landing page evaluates on MPTP2078/M2k/Mizar40 and builds on E — unverified)

## Learnings
- Setting (Abstract; §2): first-order logic without equality, given-clause saturation prover with a learned clause-scoring network (transformer on clause graphs), trained per TPTP domain from random init; no human proofs.
- HER variant (§2.2): every non-input clause D generated during a failed refutation is a "hindsight goal"; the ancestors of D are positives and other generated clauses negatives for the relabelled problem C_s ∪ {¬D}. "while the original version of HER … only uses the last reached state as a single hindsight goal, we use all intermediate clauses".
- Subsampling (§2.3, Alg. 2, App. C): the n² (positive, goal) pairs are far too many; goals are sampled with a heavy-tailed weight by clause size, w_size = 1/ln(size+e) − 1/ln(size+e+1), favouring small goals "since the empty clause (which is the true target) has size 0".
- "Incremental learning" (§2.3): all conjectures are retried under the Uniform Budgeted Scheduler with time budgets "3s, 6s, 12s, …, 3072s"; search continues after a proof is found to get "often shorter" proofs. The curriculum is implicit — whichever targets become provable.
- Results (§3, Table 1): IL w/HER "proved 1.94 times as many problems as the basic prover"; "14 (1.5%) more conjectures than E 1h and 23 (2.42%) fewer … than E 7d". Ablation: "IL w/o HER … failing to prove 198 (20.9%) of the conjectures that can be proven by IL w/HER", failures concentrated in "hard" domains where "IL w/o HER stalled". Shorter proofs than E on "921 conjectures (97.9%)" of 941.

## Evidence and limitations
- The ablation is the clean HER evidence; ten models × 1000 actors × 7 days; no seed spread reported in text.
- The learned object is a scorer (classifier) inside a complete search procedure, not a generative policy.

## Connections and questions
- How it differs from our hindsight: (1) they also use negatives (non-ancestor clauses) — we train only on positive proofs; (2) explicit goal-size weighting toward targets that look like the real target — we do not weight relabelled theorems by closeness to the ladder rung (e.g. by L_true or formula size); (3) relabelled goals are subsampled to a learner-throughput budget. (4) Their prover keeps searching for shorter proofs after success (cf. Polu 2022 shortest-proof rule, already proposed).
- Smallest test for us: weight hindsight-relabelled examples by their L_true relative to the current frontier rung (up-weight long ones) instead of uniform — addresses B1 if hindsight data is dominated by short trivia.
