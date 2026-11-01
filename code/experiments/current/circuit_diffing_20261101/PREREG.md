---
written_on: 2026-11-01
written_by: agent:claude
---

# Pre-registration: does GRPO create features absent from the Stage-1 base?

Question (Leon, 2026-11-01): does RL in our high-RL regime create new features and circuits that
the pretrained base lacks, or does it only reweight the base's own? Elicitation predicts that only
existing features are up- or down-weighted. Capability gain allows new ones.

Each stage's section is written and hashed before that stage runs. The sha256 and PT time of
each version are in [`log.md`](log.md). Sections are appended later and never edited once hashed.
Deviations are recorded in `log.md` and in the README.

## Stage 0: weight diffing (CPU)

**Checkpoints.** The seed-0 chain: base `S1g3_s0`; the low-RL arm `kp16_st{1,2,4,8,16,32}` and
`kdense_st{1,2,4,8,16,32}`; `mini_it{2,4,8}`; L2 end; L3 `it5…it35`; L3 end; ki78. Also L2 and L3
ends for seeds 1 and 2 against their own bases. R is taken from `lowrl_compute.tsv` or recomputed
with the same rule (`common.rl_R`).

**Matrices.** Every 2-D weight: `qkv` split into q, k and v (384 × 384 each), `proj`, `fc1`
(1280 × 384) and `fc2` (384 × 1280) in each of the 6 blocks, plus `emb` and `head`.

**Metrics.** ΔW = W_ckpt − W_base.
- `rel`: ‖ΔW‖_F / ‖W_base‖_F.
- `erank`: the effective rank exp(H(σ/Σσ)) of ΔW (Roy & Vetterli 2007), divided by min(n, m).
  `r90`: the fraction of singular values needed for 90% of ‖ΔW‖²_F.
- `inside`: the share of ‖ΔW‖²_F in the base's top-k left *and* right singular subspaces,
  ‖U_kᵀ ΔW V_k‖²_F / ‖ΔW‖²_F, with k = 10% of min(n, m). The random expectation is
  k² / (n·m), which is 1% for square matrices. `align` = inside / random expectation. `outside` =
  1 − inside. One-sided left and right shares are also reported, with expectations k/n and k/m.
- Update sparsity, as in Mukherjee et al. 2025: the share of entries with |ΔW| ≤ 1e-6 · max|W_base|.

**Predictions.**
- **P0.1.** The median `rel` over block matrices rises monotonically with R along the seed-0
  chain (Spearman ρ ≥ 0.9). It is below 0.2% at `kp16_st1` and between 1% and 30% at the L3 end.
- **P0.2.** ΔW is high-rank everywhere. At the L3 end, at least 75% of block matrices have
  `erank` ≥ 0.5. Reason: AdamW normalises each coordinate, and Mukherjee et al. report
  near-full-rank RL updates.
- **P0.3.** At the L3 end ΔW is mostly outside the base's top subspaces: median `align` is between
  0.5 and 5, and median `outside` is at least 90%.
- **P0.4.** Seeds 1 and 2 at the L3 end are within ±30% (relative) of seed 0 on median `rel`,
  `erank` and `outside`.
- **P0.5.** Sparsity is below 5% at the L3 end. Weights are fp32, so AdamW moves almost every
  entry. Mukherjee's 70–95% sparsity comes from bf16 rounding.

**Pre-registered reading.** Stage 0 is descriptive. It cannot separate new features from
reweighting: a high-rank update outside the base's top subspaces fits both, and noise-like AdamW
updates look like that too. The one informative outcome would be ΔW confined to the base's top
subspaces (median `align` ≥ 10, inside ≥ 30%). That would lean towards reweighting the base's
dominant directions.

## Stage 1: crosscoder model diffing (GPU)

Written after Stage 0's results were seen ([`stage0.md`](stage0.md)) and before any crosscoder
was trained.

**Data.** For a pair (A = base `S1g3_s{seed}`, B = an RL checkpoint of that seed), paired
residual-stream activations on identical token sequences from three sources: B's own rollouts,
A's own rollouts and Stage-1 pretraining pairs (`rollouts.py`). Rollouts are 1,200 problems from
the L2 and L3 curriculum pools (the RL training distribution), 4 attempts each, plain arm,
budgets 0 / 64 / 256, temperature 1.0. Each environment step is one sequence: state, effort tag,
think block and action. The pretraining sample is 60,000 pairs, about 6.0M tokens. Charles149,
13–20, 21–30 and textbook72 are not used. 3% of each source's sequences are held out for
evaluation.

**Crosscoder.** BatchTopK (Bussmann et al. 2024; as in Minder et al. 2025), 8,192 latents, k = 32
per token, 40M training tokens. Each model's activations are scaled by one scalar to E‖a‖² = d.
AuxK keeps latents alive. One residual layer per crosscoder: layer 3 (after block 3 of 6) is the
primary layer, and layers 1 and 5 are added for the main pair and the null.

**Pairs.**
- Seed 0: base vs `kp16_st1` (the null, R = 7.6e-5), `kp16_st{2,4,8,16,32}`, `kdense_st{8,32}`,
  `mini_it8`, L2, `L3_it20`, L3 end and ki78.
- Seeds 1 and 2: base vs L2 and L3 end.
- Replicates with crosscoder seed 1: the null and the seed-0 L3 end.

**Classification.** Latents that never fire on the held-out data are dead and excluded.
- Δnorm_j = ½·((‖d_jᴬ‖ − ‖d_jᴮ‖) / max(‖d_jᴬ‖, ‖d_jᴮ‖) + 1), as in Minder et al.
  - RL-only (B-only): Δnorm < 0.1, i.e. the base decoder is under 20% of the RL decoder.
  - base-only: Δnorm > 0.9.
  - shared: 0.4 ≤ Δnorm ≤ 0.6.
- **Latent Scaling** (Minder et al.). For each latent and its RL decoder d = d_jᴮ, fit
  β = Σ f_j ⟨d, y⟩ / (‖d‖² Σ f_j²) on the held-out data, for two targets in each model:
  - the error target y = a − â + f_j d_jᵐ;
  - the reconstruction target y = â − b_dec.

  ν^ε = β_base^ε / β_RL^ε, and ν^r = β_base^r / β_RL^r.
- **Strict RL-only**: Δnorm < 0.1 and ν^ε < 0.3 and ν^r < 0.3, with sensitivity at 0.2 and 0.5.
  Base-only is defined symmetrically, with the roles swapped.

**Metrics.**
- N_RL: the number of strict RL-only latents.
- E_RL: the share of the RL model's held-out activation variance carried by strict RL-only
  latents, Σ_x Σ_j f_j² ‖d_jᴮ‖² / Σ_x ‖a_B − ā_B‖², overall and per source.
- N_base and E_base, defined the same way for base-only latents.
- Firing rates of each latent on each source.
- For shared latents: the decoder norm ratio and the decoder cosine.

**Quality bar.** For every pair, FVU ≤ 0.15 for both models and next-token loss recovered ≥ 80%
when the layer is spliced with the reconstruction. If a crosscoder misses this bar, its counts
are not used. Any change to the recipe to meet the bar is logged and applied to every pair.

**Gate.** The method separates the hypotheses only if both hold:
- N_RL(null) ≤ max(5, 0.2 · N_RL(L3 end));
- E_RL(null) ≤ 0.2 · E_RL(L3 end).

If the gate fails, stop and report. If the L3 end itself has N_RL ≤ 5, the gate does not fail.
The finding is then "no RL-only features at this resolution", and Stage 2 is changed to test
reweighting only.

**Predictions.**
- **P1.1.** Quality: FVU ≤ 0.10 for both models at layer 3, and loss recovered ≥ 90%.
- **P1.2.** Null: N_RL ≤ 3 and E_RL < 0.1%. At least 99% of live latents are shared.
- **P1.3.** L3 end, seed 0, layer 3: N_RL between 5 and 100 (0.06–1.2% of latents), E_RL between
  0.5% and 5%, and N_base ≤ N_RL.
- **P1.4.** N_RL and E_RL rise with R. At `kp16_st{2,4}` (R ≤ 1.6e-4) they are within 2× the
  null plus 3 latents. Following Stage 0, they track optimizer steps more than R:
  `kdense_st32` is within 2× of `kp16_st32` although its R is 17× larger.
- **P1.5.** The L3 ends of seeds 1 and 2 are within 2× of seed 0 on N_RL and E_RL.
- **P1.6.** The crosscoder-seed replicate of the L3 end has N_RL within 1.5× of seed 0. At least
  50% of seed-0 strict RL-only latents have a match in the replicate (RL-decoder cosine ≥ 0.7)
  that is itself RL-only by Δnorm.
- **P1.7.** RL-only latents fire at least 2× more often on RL rollouts than on base rollouts.
  Their firing is concentrated on sampled tokens (think and action) rather than state tokens.

**Planned weight-drift control** (an addition to the brief). Any weight change moves
activations, so RL-only latents could reflect the size of the drift and not new computation. A
"CPT" model continues Stage-1 training of `S1g3_s0` on its own pretraining pairs (`train.finetune`:
AdamW with betas 0.9 / 0.95, weight decay 0, grad clip 1, 312 steps of about 400 records, lr
1.5e-4 with cosine decay to 30%). This matches the GRPO chain's optimizer, step count and tokens
per step, but leaves the training distribution unchanged.
- If the median `rel` is not within 0.7–1.4× of the L3 end's 6.5%, the lr is raised (to 4.5e-4,
  then 1e-3) to match.
- The crosscoder for base vs CPT then runs on the same recipe. B's rollouts come from CPT.
- Pre-registered reading: if N_RL(CPT) ≥ 0.5 · N_RL(L3 end) at matched `rel`, RL-only latents are
  not specific to RL, and the count is not a usable capability metric.

**Pre-registered reading.**
- If the gate passes, the L3 end clearly exceeds both the null and CPT, and E_RL ≥ 1%: RL writes
  directions into the residual stream that the base does not have at that layer, at least
  linearly. These are "new features" in the crosscoder sense. That is not yet evidence of new
  circuits; Stage 2 tests whether they matter.
- If the L3 end has few RL-only latents but its shared latents have changed decoder norms or
  firing rates: consistent with reweighting at that layer.
- The threshold between "new" and "strongly upweighted" is a choice. The base-to-RL strength
  ratios (Δnorm, ν) are reported as distributions, not only counts.

## Stage 2: causal tests, revised to the reweighting branch (GPU)

Written after these Stage 1 results were seen, and before any Stage 2 sampling.
- The primary crosscoder (base vs L3 end, seed 0, layer 3, pre-registered recipe) has 0 strict
  RL-only latents and 0 base-only latents. The null also has 0.
- The planted positive control recovers 4 of 6 planted features as strict RL-only, with no
  false positives.

As pre-registered, Stage 2 therefore tests reweighting, not RL-only ablation. The ablation of
RL-only latents (and its matched random control) is dropped because the set is empty.

**Setup.** Crosscoder `L3_s0_L3_n8192_k32_seed0`.
- Problems: `stage2_problems.json`, 80 RL-gained problems (G) and 121 shared ones (S).
  - G: 20 from Charles149 and 60 from 13–20.
  - S: 72 from Charles149 and 49 from 13–20.
  - Neither set was used for crosscoder training.
- 32 attempts per problem, at the reads' setting: plain arm, B64, temperature 0.8,
  max_action 512, max_steps 96. Every attempt is judged by Lean.
- Patching runs inside the sampler, at every token (`patching.PairPatched`).

**Conditions.**
- `rl_clean` and `base_clean`: the references.
- `rl_recon`: the RL model with layer 3 replaced by its crosscoder reconstruction, a fidelity
  check.
- `base_replace`: the base with layer 3 replaced by the RL model's layer-3 residual on the same
  tokens. This is the positive control for any layer-3 intervention.
- `base_transplant`: the base with every live latent switched to its RL version (norm and
  direction), keeping the base's reconstruction error.
- `base_rescale`: the base with every shared latent rescaled to the RL model's strength along
  the base's own direction. This is pure reweighting, with no rotation.
- `rl_revert`: the RL model with every live latent switched back to its base version.
- `rl_unscale`: the RL model with shared latents scaled back to base strength along its own
  directions. This undoes the reweighting only.

**Metrics.**
- Solve %: the mean over problems of n_ok / 32, for G and S.
- Gap recovered for base-driven conditions: (cond − base_clean) / (rl_clean − base_clean).
- Gap removed for RL-driven conditions: (rl_clean − cond) / (rl_clean − base_clean).
- Relative drops and selectivity (drop G − drop S).
- 95% paired problem-bootstrap intervals (`stage2_analyze.py`).

**Predictions.**
- **P2.1.** `base_clean` and `rl_clean` match the reads' category definitions:
  - `base_clean`: G ≤ 5% and S ≥ 40%;
  - `rl_clean`: G ≥ 30% and S ≥ 60%.
- **P2.2.** `rl_recon` keeps at least 80% of `rl_clean`'s solve % on both G and S.
- **P2.3.** `base_replace` recovers 20–80% of the RL − base gap on G.
- **P2.4.** `base_transplant` recovers at least half as much of the G gap as `base_replace`.
- **P2.5.** `base_rescale` recovers under 15% of the G gap.
- **P2.6.** `rl_revert` removes at least 30% of the G gap, with selectivity > 0. `rl_unscale`
  removes under 15%.

**Pre-registered reading.**
- **Rotation, not new features.** Suppose `base_replace` and `base_transplant` recover a large
  share of the gap but `base_rescale` does not, and `rl_revert` removes it but `rl_unscale` does
  not. Then the part of RL's gain carried at layer 3 lives in changed write directions of
  features the base already has. That is neither new features nor pure magnitude reweighting.
- **Change downstream.** If `base_replace` recovers little, the gain depends on changes after
  layer 3, which a layer-3 crosscoder cannot speak to. That leaves room for new downstream
  circuits that read existing features differently.
- **Reweighting.** If `base_rescale` recovers most of the gap, that supports magnitude
  reweighting of existing features.

## Stage 2 addendum: the same tests at layer 5 (GPU)

Written after the layer-3 conditions `rl_clean`, `base_clean` and `base_replace` were seen, and
before any layer-5 sampling.
- `rl_clean`: G 67.1%, S 93.0%.
- `base_clean`: G 2.0%, S 70.9%.
- `base_replace` at layer 3: G 13.9%, S 81.3%. That is 18% of the G gap, just under P2.3's 20%.

So most of RL's gain on G does not pass through the layer-3 residual alone. Block 4–6 changes
matter too. The layer-5 crosscoder (`L3_s0_L5_n8192_k32_seed0`) also has 0 strict RL-only
latents.

**Conditions.**
- Run `base_replace`, `base_transplant`, `base_rescale` and `rl_revert` with the layer-5
  crosscoder: same problems, attempts and settings.
- `rl_clean` and `base_clean` do not depend on the crosscoder. They are copied from the layer-3
  run, with the same seed and the same samples.

**Predictions.**
- **P2.7.** `base_replace` at layer 5 recovers more of the G gap than at layer 3, at least 40%.
- **P2.8.** `base_transplant` at layer 5 recovers at least half of what `base_replace` at layer 5
  recovers on G.
- **P2.9.** `base_rescale` at layer 5 recovers under 15% of the G gap.

**Reading.** If P2.7 and P2.8 hold and P2.9 holds, then most of the RL − base difference that
layer 5 carries is expressed through latents the base already has. Those latents are rotated and
rescaled, not new. Pure magnitude reweighting is not enough.

## Stage 2 addendum: a matched control for `rl_revert`

Written after the layer-3 results for `base_transplant`, `base_rescale`, `rl_revert` and
`rl_unscale` were seen, and before this control ran.
- `rl_revert` removes 41% [34, 47] of the G gap, with a relative drop of 40% on G against 6% on S.
- `rl_unscale` removes −1% [−5, 3].

Harder problems may simply be more fragile under any perturbation of this size: the edit is 34%
of the residual norm. So a matched control is added.

**Condition.** `rl_revertperm`: the same revert edit, but each latent's difference vector
(d^RL / s_RL − d^base / s_base) is permuted among the live latents (seed 0). The set of
difference vectors is the same, but each one is applied when the wrong latent fires. It runs at
layer 3, and at layer 5 with the layer-5 conditions.

**Prediction.**
- **P2.10.** `rl_revertperm` removes less of the G gap than `rl_revert` and has lower selectivity
  (relative drop G − S).

**Reading.**
- If `rl_revertperm` hurts G as much as `rl_revert` does, the revert effect is a generic
  perturbation effect. It is then not evidence that RL's gains run through those particular
  feature rotations.
- If `rl_revertperm` hurts less, the specific directions matter.
