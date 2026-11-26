---
written_on: 2026-11-26
written_by: agent:claude
status: proposal 49, written at Dan's request 2026-11-26 (the low-rank add-on paused from proposal 47, with the truncation test). Dan approved it on 2026-11-26, up to $80; launched as `olmo-lowrank` on agent2 (brief `followups/BRIEF_olmo-lowrank.md`). No result yet.
---

# Proposal 49: is RL's change to the weights low-rank, and is the low-rank part what matters? (`olmo-lowrank`)

## The short version

Every weight matrix's change can be split into rank-1 pieces, sorted by size: the singular value decomposition. The
list of sizes is the change's **spectrum**. It can take three shapes:
- **low-rank:** a few large pieces and then a cliff;
- **noise:** a slow, smooth decline with no cliff;
- **signal plus noise:** a few spikes standing above a flat floor.

A rank-16 LoRA is a change forced to have at most 16 pieces in every matrix.

[`olmo-weights`](../../experiment-summaries/2026-11-25-olmo-weights-rl-total-move-31-185x-below-sft-its-8-9x-lr-per-step-holds-only-at-25-50-step-spacing/README.md)
left two puzzles this can resolve.

1. **The LoRA points along RL's change, but only a little.** `olmo-ifladder`'s `lora_main` was trained on 16 RL
   trajectories and recovers 40–41 of the 48 IF reach prompts. Its cosine with RL's change (DPO → step 2,300, over the
   448 attention and MLP matrices) is 0.0052. That is about 1,000 random standard deviations above chance and 28× the
   control LoRA, yet only half a percent of perfect alignment. Two readings fit:
   - **(a) RL's change is genuinely high-rank,** and the LoRA reproduces a small slice of it that suffices for the
     behaviour.
   - **(b) RL's change is a low-rank signal under a high-rank floor of noise,** and the noise dilutes the LoRA's cosine.
2. **RL wanders.** Its straightness is 0.10 (32B) and 0.07 (7B), and consecutive changes have cosine ≈ −0.2, so most
   of each step is undone later. Is the back-and-forth high-rank noise, sampling jitter plus bf16 rounding (65 % of the
   32B weights end RL within one bf16 step of DPO), with a low-rank drift underneath?

**Prior work.** Mukherjee et al. (NeurIPS 2025, [arXiv 2505.11711](https://arxiv.org/abs/2505.11711)) report that
RL updates only 5–30 % of parameters, yet the updates to almost every matrix are **nearly full-rank**: reading (a).
They measured rank, not whether the low-rank part carries the behaviour.

**What this proposal adds:** a **truncation test**. Rebuild the model as DPO plus only the top-k pieces of RL's
change, and see how much of RL's IFBench gain survives. It cross-tests the spectra behaviourally: a change can have a
spike-and-floor spectrum and still need the floor, or a full-rank spectrum whose top pieces do all the work.

## Part 1: spectra (32B, plus 7B as a cheap second model)

**Changes to decompose,** for each of the 448 attention and MLP matrices (q, k, v, o, gate, up and down × 64 layers;
7B: its own count):

| change | 32B | 7B |
|---|---|---|
| RL net, DPO → step t | t = 50, 100, 500, 750, 1,100, 1,600, 2,300 | t = 25, 100, 500, 1,000, 1,375 |
| single RL intervals | three 50-step intervals (early, mid, late) | three 25-step intervals |
| a mid-run RL chord | 1,100 → 2,100 | 500 → 1,375 |
| SFT total | base `main` → SFT `main` | same |
| DPO total | SFT `main` → DPO | same |
| RL from the base | — | RL-Zero-IF, base → `main` |

**Lineage:** the 32B RL checkpoints at steps 150–200 lie on an abandoned branch (`olmo-weights`), so stay on the
lineage. The 32B SFT `main` is a soup of two runs; treat it as the SFT end point.

**Baselines,** so a spike can be told from noise:
- **Random:** an i.i.d. Gaussian matrix of the same shape, scaled to the same Frobenius norm.
- **bf16 rounding:** round to bf16 a synthetic update with the same per-element size as RL's 50-step updates, applied
  to the DPO weights, and subtract. This gives the floor that rounding alone produces. Pre-register the exact
  construction.

**Measures, per change and matrix,** aggregated by module type and depth:
- the share of ‖ΔW‖² in the top 1, 4, 16, 64 and 256 pieces;
- the stable rank, ‖ΔW‖² ÷ σ₁²;
- the **signal rank:** the number of singular values above the rounding baseline's largest.

**How the RL net change develops with steps:** does the top-16 share grow, and do the spikes grow while the floor
stays put?

**Overlap with the LoRA** (32B): for each matrix, the overlap of RL's top-16 left and right singular subspaces with
`lora_main`'s column and row spaces (B and A). The scale runs from 0 to 1; random is ≈ 16 / n. Compare with
`lora_ctrl` and with a random rank-16 baseline.

## Part 2: the truncation test (32B)

**Models,** each scored at `olmo-ifcurve`'s decoding:

| arm | weights | what it tests |
|---|---|---|
| T_k, for k = 1, 4, 16, 64, 256 | DPO + the top-k pieces of RL's change in each of the 448 matrices; everything else at DPO | how much of RL's gain the top-k pieces carry |
| F448 | DPO with the 448 matrices replaced by RL step 2,300's; everything else at DPO | whether the attention and MLP matrices carry the gain (embeddings, norms and head left out) |
| C16 | F448 minus the top-16 pieces: the floor only | whether the floor alone does anything |
| stored | `dpo` and `s2300` from earlier runs | the end points |

**How to build them without rounding the change away.** RL's change is mostly at the bf16 rounding scale, so adding a
truncated change to DPO and storing the result in bf16 would erase much of it. Instead:
- **T_k as a rank-k LoRA:** U_k Σ_k V_kᵀ is exactly a rank-k LoRA (B = U_k Σ_k, A = V_kᵀ), so serve it on the DPO base
  through vLLM's LoRA path, as `olmo-ifladder` did.
- **F448:** build it by replacing tensors, which is exact.
- **C16:** serve it as F448 plus a negative rank-16 LoRA.
- **Fallback:** if vLLM's maximum LoRA rank is below 256, run T_256 at the highest supported rank and say so.

**Prompts and draws:**
- **Reach prompts:** `olmo-ifladder`'s 48 instruction-following reach prompts, 16 draws per arm.
- **Aggregate curve:** the 218 IFBench prompts of `olmo-ifcurve`'s arm A, 4 draws per arm.

**Grading:** the strict checker, plus the blinded hand gate on every deciding reach success, under the rules of
`olmo-ifcurve`, including its dated addenda. Report the gamed share per arm.

## Readings (thresholds to pre-register)

Let G be an arm's recovered share of RL's gain: (arm − `dpo`) ÷ (`s2300` − `dpo`). Compute it on the reach prompts'
genuine rate and on the 218 prompts' strict pass@1.

| reading | the arms show | spectra expected |
|---|---|---|
| **low-rank carries it,** reading (b) | G(T₁₆) ≥ 0.5 and G(C16) ≤ 0.2 | spikes above the rounding floor in RL's net change |
| **high-rank,** reading (a) | G(T₂₅₆) < 0.5 while G(F448) ≥ 0.75 | no clear spikes |
| **outside the 448 matrices** | G(F448) < 0.75: embeddings, norms or head matter | |

**Otherwise:** report k₅₀, the smallest k with G ≥ 0.5, and compare it with the signal rank from Part 1.

**My guesses, not readings:**
- RL's net change shows spikes above the rounding floor, and single 50-step intervals are mostly floor.
- G(T₁₆) is between 0.3 and 0.7.
- `lora_main` overlaps RL's top-16 subspace far more than its 0.0052 cosine suggests.

## What it does not show

- **Uniqueness:** a change that works when truncated is one sufficient change, not necessarily the one RL "meant".
- **Generality:** one RL run, one benchmark family for the behaviour test, and no truncation test at 7B (it has no
  reach set).
- **Per-matrix rank:** truncation applies the same k to every matrix; matrices may differ in how many pieces they need.

## Cost

- **Downloads:** about 14 checkpoints at 32B (≈ 0.9 TB) and 10 at 7B (≈ 0.15 TB).
- **Part 1, the SVDs:** a few thousand decompositions; randomized top-256 for all, full SVD for a seeded subset of
  matrices. A few GPU-hours on an H200: ≈ $10–15.
- **Part 2:** 7 new arms × (48 × 16 + 218 × 4) ≈ 11,500 draws at ≈ $0.003–0.0035 each ≈ $35–40, plus model loads.

**Estimate ≈ $50–65; budget $80** (Dan). Re-cost at stage 0.
