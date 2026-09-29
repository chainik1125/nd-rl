---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# STP: a conjecturer trained on barely-provable statements beats expert iteration on a stalled pool
Paper: STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving (Dong & Ma, 2025)
Source: https://arxiv.org/abs/2502.00212v4

## Learnings
- One LLM plays two roles: a conjecturer that, given a seed theorem, its proof and a lemma used in it, proposes a "new, related conjecture", and a prover that runs "standard expert iteration" on dataset statements plus conjectures (Abstract; Fig. 1; §3.2).
- "Barely provable" is operational (§3.2, Step 4): empirical pass rate P̂(c) from K proofs; the conjecturer is trained on conjectures with "P̂(c_i) ∈ (0,1/4]", a correct proof that uses the seed lemma, then an elegance filter that removes "conjectures whose minimum proof length divided by the length of the conjecture is in the lowest 20%", then a Wasserstein re-weighting toward the embedding distribution of still-unproved statements (to avoid mode collapse; App. A.5 reports that collapse happened in early runs).
- Prover data (§3.2): only proofs of statements with pass rate "below 1/2" ("We consider other correct proofs trivial"), replay buffer of last three iterations, per-statement weight reciprocal to the number of proofs, length penalty γ^L.
- Final model is re-trained from base on SFT data plus all proofs of statements with pass rate ≤ 1/4 (§3.3).
- Budget (§4.1): "we sample K=32 proofs per conjecture/statement. For the expert iteration and parallel sampling, we use K=64 … STP has the same sample budget as the baseline methods per iteration."
- Headline: "STP proves 28.5% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration" (Abstract). NOTE: 13.2% is Wu et al. 2024's number, not an equal-budget run; the equal-budget comparison is only graphical (Fig. 2/3, Fig. 5 left, Fig. 4 left for Isabelle, branching from STP checkpoints).
- Mechanism evidence (§4.4; Fig. 4 right): at a checkpoint with 11.4% cumulative pass rate, "only 131 out of 2.5M generated proofs of the unproved statements are correct", whereas "at least 47% of the generated conjectures in STP training are successfully proved".
- Re-training with conjecture proofs adds "about 2-3% performance gain (for pass@128)" on miniF2F/ProofNet vs re-training on LeanWorkbook proofs only (§4.4; Table 2, App. B.1).

## Evidence and limitations
- Scale: DeepSeek-Prover-V1.5-SFT (7B), 48 iterations, "3.6M conjectures, 241M proofs, and 51.3B tokens" (§4.2). Isabelle run from Llemma-7b, 58 iterations.
- No numeric table for STP vs EI at equal proofs; I read the claim from captions/text only. Only one seed.
- LeanWorkbook ceiling: of 20 unproved sampled statements "only 7 statements are provable" (§4.2), so EI's plateau is partly unprovable targets.

## Connections and questions
- B1/B2: our EI stalls at L*=12 on a pool where the frontier rungs have near-zero pass rates; STP's diagnosis (EI's samples on hard targets yield ~no signal) matches. What we lack: any conjecturer. Our generator plays the role of an unconditioned conjecturer, but targets are not chosen by the current prover's pass rate.
- Cheapest STP-flavoured test without a learned conjecturer: each round, generate candidate targets (generator, or mutations/extensions of solved proofs), estimate P̂ with K≈32 attempts from the current policy, keep those with P̂ in (0, 1/4] as EI targets, and drop proofs of targets with P̂ ≥ 1/2 from EI training data. Our fast sampler makes K=32 cheap; Lean checking is the cost (B5).
- A learned conjecturer head for a 3.2M model is speculative; STP's conjecturer relies on a 7B pretrained LM that already writes Lean statements.
