---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# BFS-Prover: length-normalised best-first search + expert iteration with beam filtering + DPO on compiler errors
Paper: BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving, Xin et al. (ByteDance), 2025
Source: https://arxiv.org/abs/2502.03438v3
## Learnings
- Search (§2.2, Eq. 1): priority queue over proof states scored by score(s_L) = Σ log p(a_t|s_t) / L^α, α ∈ [0,1]. "increasing α and/or reducing the expansion width drives the search system toward exploring deeper paths". Motivation (§1): cumulative log-prob "intrinsically penalizes longer paths".
- EI loop (§2.3): (1) beam-search filtering: statements provable by beam search (width 32) are removed and their proofs "deliberately not added" to training; (2) BFS with temperature sampling on the rest; all (state, tactic) pairs on successful paths are added; (3) SFT from the base model on the accumulated data; or (4) DPO with pairs (tactic on the proof path, sibling tactic that caused a Lean error).
- Settings (§3.1): α = 0.0 in EI "to minimize inductive bias"; eval uses α = 0.5, temperature 1.1, width 2 (§3.4.1).
- Results: miniF2F-test 70.83% ± 0.89 at 2048×2×600 and 72.95% accumulated over α ∈ {0, 0.5, 1} (Table 1). SFT 64.58% → 70.38% and SFT+DPO 64.98% → 70.83% from pass@64 to pass@2048 (Fig. 4, §3.4.2).
- Proof length rises over EI rounds: mean 10.2 → 16.7 tactics (Fig. 2, a "sample" of an early and a later iteration).
## Evidence and limitations
- There is no ablation of α, of the beam filter, or of BFS vs sampling-based EI. The α benefit is argued, not measured; the 72.95% "accumulative" mixes three α values and so conflates α with extra attempts.
- The DPO gain at pass@2048 is 0.45 points, inside the reported ±0.89 band.
- The proof-length shift (Fig. 2) mixes a harder remaining corpus (easy statements are filtered out) with policy change.
- 7B Qwen2.5-Math base; ~900k autoformalised statements; 8×A100 nodes.
## Connections and questions
- B1: length normalisation is the one knob in this cluster that targets the length frontier directly. With α = 0 a 13-step proof competes against short partial paths purely on summed log-prob; α = 1 (mean log-prob) removes that. Cheap to implement and to ablate in our step env.
- Beam filtering ≈ dropping theorems the policy already solves greedily; close to our ladder/curriculum filters (already done). The "don't add easy proofs" rule is a data-selection variant we have not tried.
- DPO on invalid steps: our state env already rejects invalid steps syntactically, so negatives come free; the reported effect size is too small to justify it on its own.
- Smallest test: best-first over the state env with α ∈ {0, 0.5, 1} at matched expansions; record the L_true distribution of found proofs and L*.
