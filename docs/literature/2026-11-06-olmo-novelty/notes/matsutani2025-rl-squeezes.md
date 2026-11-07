---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - matsutaniRLSqueezesSFT2025
---

# RL Squeezes, SFT Expands (v2): trajectory diversity at 256 samples, no instruction following

Paper: [@matsutaniRLSqueezesSFT2025] (proposed key, as in the rl-compute-share note)
Source: https://arxiv.org/abs/2509.21128v2 (v2, 2026-05-27, the latest; ICLR 2026). Body and appendices B–C read.
Extends `nd-rl/docs/literature/2026-10-19-rl-compute-share/notes/matsutani2025-rl-squeezes.md`. Ids are in
`ledger/IL.tsv`.

## Setup card

| item | value |
|---|---|
| models, start | 1.5B / 7B / 14B quartets: pretrained base, zero-RL (Oat-Zero, SimpleRL-Zoo), SFT (R1-Distill) and SFT+RL (ProRL, AceReason) [IL-83, IL-74]. Exceptions are Llama-3.1-8B and AceReason-1.1 [IL-77, IL-78] |
| RL share of pretraining | not computed for this paper (the rl-compute-share table has ProRL v1 at 2.6 × 10⁻³ – 1.2 × 10⁻² and SimpleRL-Zoo at ≤ 1.1 × 10⁻³) |
| RL steps | re-analysis only; ProRL is described as "2500 steps", revision not stated [IL-74] |
| domains | maths (AIME24/25, AMC23) and code (HumanEval) [IL-83] |
| largest k | 256 for maths, in a figure only [IL-71, IL-72]; 128 for code [IL-81] |
| grading | verifiable answer, using Yue et al.'s implementation [IL-193] |
| controls | symmetric budgets **partly** (16,000 tokens, but 4,096 for the Qwen2.5-Math base and Oat-Zero) [IL-75]; format tolerance **no**; guessing **no**; contamination **no**; training data not controlled [IL-80] |

## Learnings

**Observed.**
- RL, from a base or from an SFT model, cuts the number of distinct incorrect trajectories (chrF + UPGMA clusters)
  [IL-69, IL-73]. It also cuts distinct correct trajectories [IL-68]. SFT adds correct trajectories [IL-70].
- Exceptions: Llama-3.1-8B RL expands correct paths [IL-77], as does AceReason-1.1-7B on AIME24/25 [IL-78]. One SFT+RL
  pair shows no RL accuracy gain [IL-76].
- The length limit matters: a 4,096-token limit "causes substantial accuracy degradation" in distilled models
  [IL-199]. The paper keeps 4,096 for the Qwen2.5-Math base and Oat-Zero [IL-75].
- Nothing on instruction following. The only "instruction" in the text is an appended "Always respond in English
  only." (text search).

**Paper interprets.** "RL compresses incorrect trajectories, whereas SFT expands correct ones" [IL-67]. The pass@1 gain
is redistribution of probability mass [IL-82]. The loss of correct trajectories explains why the base wins at large k
[IL-68]. It reads Invisible Leash as a theorem that RLVR "cannot exceed the support of Base model" [IL-79]. Invisible
Leash v5 itself now concedes that exact support is never empty [IL-3].

**Our interpretation.** The paper measures solution *diversity* inside solved problems, not problem-level reach. It
gives no reach or reverse counts and stops at k = 256, so it can neither support nor refute an RL-only-problem claim.
The reading of Invisible Leash it cites [IL-79] is the strict-support reading that Leash v5 has dropped.

## Evidence and limitations

- It is cross-sectional: different recipes, data and bases, and no run is followed over steps [IL-80].
- Pass@k appears only in figures [IL-72]; the unequal length limits [IL-75] confound the zero-RL comparisons.

## Connections and questions

- **C1, unrelated:** it counts trajectory diversity, not problems. The Llama-3.1-8B exception (RL expands correct paths
  from a pretrained base) is the closest case to our RL-Zero result [IL-77].
- **C3, unrelated:** no IF, no reach counts, k ≤ 256 [IL-72, IL-83].
- **C8, replicates** (the confound only): too short a length limit depresses distilled models' pass@k [IL-199]. This is
  the mechanism behind C8 Test 1.
- See [wu2025-invisible-leash.md](wu2025-invisible-leash.md) and
  [yang2026-passk-crossovers.md](yang2026-passk-crossovers.md).
