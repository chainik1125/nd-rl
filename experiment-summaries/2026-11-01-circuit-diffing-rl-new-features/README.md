---
written_on: 2026-11-01
written_by: agent:claude
---

# Circuit diffing: does GRPO create features the Stage-1 base lacks?

Status (2026-11-01, written by agent:claude for Leon):
- Stages 0–2 are done for the seed-0 primary pair, base `S1g3_s0` against the L3 end.
- Replicates are queued but not run: seeds 1 and 2, the crosscoder-seed replicate, the
  low-RL sweep, the CPT crosscoder and Stage 2 at layer 5. Other users hold 3 of serrano's GPUs,
  the power-supply limit.
- Stage 3 (edge-level circuits) was not attempted.

Working folder: [`code/experiments/current/circuit_diffing_20261101/`](../../code/experiments/current/circuit_diffing_20261101/)
(branch `leon/circuit-diffing`). Pre-registration, with the sha256 of each version in the log:
[`PREREG.md`](../../code/experiments/current/circuit_diffing_20261101/PREREG.md). Running
record: [`log.md`](../../code/experiments/current/circuit_diffing_20261101/log.md).

## Question

Leon (2026-11-01) asked for a mechanistic version of the paper's claim that RL creates
capability rather than eliciting it. The test: does post-training create circuits the
pretrained base lacks? Elicitation predicts no new circuits, only upweighted ones.

The models are the ~10M-parameter thinking prover (6 Peri-LN blocks, d = 384) and its GRPO
continuation. RL ends at about 0.29× Stage-1 FLOPs. On Charles149 the L3 end beats its base by
+10.7 points at pass@1024.

## Bottom line

**Observed.**
1. **No RL-only features.**
   - BatchTopK crosscoders (8,192 latents, k 32) on paired residual streams of the base and the
     L3 end find 0 RL-only latents and 0 base-only latents at layers 1, 3 and 5. Every live latent
     is shared.
   - The behaviourally identical null (R 7.6e-5) also has 0.
   - A planted positive control recovers 4 of 6 synthetic RL-only features, with no false
     positives. A new feature carrying ≥ ~0.01% of RL activation variance in one latent would
     have been found.
2. **RL changes existing features' write directions, not their magnitudes.**
   - Shared latents' decoders rotate: the energy-weighted decoder cosine between base and RL is
     0.945 at the L3 end, 0.968 at the L2 end and 1.000 for the null.
   - Decoder norms barely move: latents whose norm changes more than 2× carry 0.03% of RL
     variance.
   - The decoder bias (the mean residual) also shifts, by about 20% of a typical activation norm.
3. **The rotations, not the magnitudes, matter causally.** These tests patch inside the sampler
   at layer 3, on 80 RL-gained (G) and 121 shared (S) problems.
   - Pure magnitude reweighting does nothing in either direction:
     - the base with RL's decoder norms recovers 0% [−1, 1] of the G gap;
     - RL with the base's norms loses −1% [−5, 3] of it.
   - Switching every latent to its other-model version (direction and norm) matters:
     - in RL, it removes 41% [34, 47] of the G gap (G 67.1% → 40.5%; S 93.0% → 87.3%);
     - in the base, it recovers 12% [8, 15].
   - Replacing the base's whole layer-3 residual with RL's recovers only 18% [14, 23]. Most of
     the gain also needs RL's changes in blocks 4–6.
4. **Weight diffing cannot tell RL from drift.** Continued pretraining on the base's own data,
   at matched optimizer, steps and weight change ("CPT"), leaves the same ΔW signature as RL:
   the same effective rank and the same alignment with the base's top singular subspaces. ΔW
   size tracks AdamW steps, not RL FLOPs.

**Interpretation** (kept apart from the results above).
- At the resolution of residual-stream crosscoders, GRPO here does not add features the base
  lacks. It re-encodes features the base already computes.
- The causal tests also rule out the simplest "upweighting" reading: rescaling shared features
  does nothing.
- So the capability gain looks like new computation over existing features (changed write
  directions and downstream use), not new features. That is a middle position: it is neither
  "new circuits = new features" nor "elicitation = upweighting".
- **For the paper, the count of RL-only crosscoder latents is not a usable "new circuits"
  metric.** It is 0 at every R we tested, including where the behavioural gain is large.

## Setup

**Models.** All checkpoints were read-only.
- Base: `S1g3_s{0,1,2}`.
- RL, seed 0: L2 end (`grpo_think_vs_s0_L2` it39, R 0.146) and L3 end
  (`grpo_think_vs_mb_kkl0_s0_L3` it39, R 0.289).
- Low-RL checkpoints `kp16_st{1..32}`, `kdense_st{1..32}` and `mini_it{2,4,8}`, copied from ynez.
- Seeds 1 and 2, L2 and L3 ends.
- R comes from `high_rl_passk_20261030/lowrl_compute.tsv`, or is recomputed with the same rule.

**Stage 0** (`stage0_weights.py`). Per 2-D matrix:
- relative ‖ΔW‖;
- effective rank;
- the share of ΔW energy in the base's top-10% singular subspaces, against the random
  expectation;
- update sparsity.

**Stage 1** (`rollouts.py`, `crosscoder.py`, `stage1_analyze.py`).
- **Data.** Paired activations of base and RL on identical token sequences, about 19M tokens
  from three sources:
  - each model's own rollouts on 1,200 L2 and L3 pool problems (4 attempts each, budgets
    0 / 64 / 256, temperature 1);
  - the base's rollouts on the same problems;
  - 60,000 Stage-1 pretraining pairs.

  Charles149, 13–20, 21–30 and textbook72 were not used.
- **Crosscoder.** One residual layer each. BatchTopK, selecting on f · (‖dᴬ‖ + ‖dᴮ‖), with
  AuxK; 8,192 latents; k 32; 40M training tokens; each model scaled to E‖a‖² = d.
- **Classes.** Minder et al.'s Δnorm (RL-only < 0.1), plus Latent Scaling, ν^ε and ν^r < 0.3.
- **Quality.** At layer 3, FVU is 0.008 for both models. Splicing the reconstruction keeps
  next-token loss at 100% recovered.
- **Planted control** (`--plant`). Six synthetic features, each a random function of token and
  position mod 29 or 331, are added to the RL model's activations only. They fire on 0.2–4.7% of
  tokens with magnitude 2–8; a typical activation norm is 19.6.

**Weight-drift control** (`cpt.py`). `S1g3_s0` trained further on its own pretraining pairs
with GRPO's optimizer: AdamW, 312 steps, about 260k tokens per step, lr 1.5e-4. Its median
relative weight change is 5.6%, against 6.5% for the L3 end.

**Stage 2** (`problems.py`, `patching.py`, `stage2.py`).
- **Problems.**
  - G is 80 problems the base practically never solves but RL does: base rate < 1% in 1,024
    attempts on Charles149 (20 problems), or 0 of 12 on 13–20 (60), with RL ≥ 10% or ≥ 8 of 16.
  - S is 121 problems both solve at ≥ 30% (Charles149: 72; 13–20: 49).
- **Sampling.** 32 attempts per problem at the read setting (plain arm, B64, temperature 0.8),
  all judged by Lean.
- **Patching.** The crosscoder runs inside `sample.think_generate` at every token. The other
  model runs alongside on the same tokens, to supply its half of the encoder input.

## Results

### Stage 0: weight diffing

![Stage 0](../../code/experiments/current/circuit_diffing_20261101/figs/stage0_weights.png)

Medians over the 36 block matrices; numbers in
[`stage0.md`](../../code/experiments/current/circuit_diffing_20261101/stage0.md).

| model | R | relative ‖ΔW‖ | effective rank / full | energy share in base top-10% (random 1%) |
|---|---|---|---|---|
| `kp16_st1` (1 AdamW step) | 7.6e-5 | 0.047% | 0.65 | 4.8% (6.1×) |
| `kp16_st32` | 1.8e-3 | 2.0% | 0.44 | 6.4% (7.4×) |
| `kdense_st32` | 3.0e-2 | 2.0% | 0.44 | 6.4% (7.7×) |
| L2 end | 0.146 | 5.1% | 0.41 | 6.3% (7.2×) |
| L3 end, seeds 0 / 1 / 2 | 0.28–0.29 | 6.5 / 6.6 / 6.7% | 0.42 / 0.42 / 0.43 | 6.3 / 6.4 / 6.1% |
| CPT (no RL) | — | 5.6% | 0.42 | 6.7% (8.0×) |

- ΔW size follows AdamW steps, not R. One step gives 0.05% in both low-RL arms, although their
  R differs 21×; 32 steps give 2.0% in both, although their R differs 17×.
- ΔW geometry is fixed from the first step on: about 7× random alignment with the base's top
  singular subspaces, and an effective rank near 0.42.
- CPT, which trains only on the base's own data, has the same geometry.

### Stage 1: crosscoders

![Stage 1](../../code/experiments/current/circuit_diffing_20261101/figs/stage1_L3.png)

Numbers in [`stage1.md`](../../code/experiments/current/circuit_diffing_20261101/stage1.md).

| pair (seed 0) | layer | live latents | RL-only candidates (Δnorm < 0.1) | strict RL-only | base-only | Δnorm range | energy-weighted decoder cos |
|---|---|---|---|---|---|---|---|
| null `kp16_st1` | 3 | 7,083 | 0 | 0 | 0 | 0.48–0.52 | 1.000 |
| L2 end | 3 | 7,125 | 0 | 0 | 0 | 0.20–0.73 | 0.968 |
| L3 end | 3 | 7,093 | 0 | 0 | 0 | 0.20–0.82 | 0.945 |
| L3 end | 5 | 7,918 | 1 (removed by Latent Scaling) | 0 | 0 | 0.08–0.89 | 0.925 |
| L3 end | 1 | 4,204 | 0 | 0 | 0 | 0.27–0.71 | 0.967 |
| L3 end + 6 planted features | 3 | 7,097 | 4 | **4, all planted** | 0 | 0.00–0.82 | 0.934 |

**Planted control** (`eval.json`, `planted`). Features 1, 2, 3 and 5 are each recovered by one
latent with Δnorm ≤ 0.02, precision ≥ 0.998 and recall 0.79–1.0.
- Feature 4 is the weakest: 0.2% of tokens at magnitude 2, about 0.002% of variance. Its latent
  has Δnorm 0.11, just over the threshold.
- Feature 6 is split over 3 redundant latents, each with Δnorm 0.13–0.21. Latent Scaling alone
  (|ν| < 0.2) flags 7 of the 8 planted latents.

**What changes, if not new features.**
- The decoder cosines of shared latents spread out: 5% are below 0.51 at the L3 end, against
  1.00 for the null.
- Latents with cosine < 0.5 carry 0.24% of RL variance. The well-aligned latents (cosine ≥ 0.8)
  carry over 70%.
- Latents that fire only on RL rollouts or only on base rollouts are as common in the null as at
  the L3 end (137 / 113 vs 126 / 130). They are rare latents subject to sampling noise.

**Exploratory.** 11 latents of the L3-end crosscoder have their RL direction absent from the
base by Latent Scaling alone (|ν^ε| and |ν^r| < 0.3); the null has 0. All have Δnorm 0.3–0.55,
so they are shared latents with a rotated decoder, not RL-only ones.
- Interpretation ([`interpret_L3_s0_L3_n8192_k32_seed0.md`](../../code/experiments/current/circuit_diffing_20261101/interpret_L3_s0_L3_n8192_k32_seed0.md)):
  they fire on token-level syntax in proof states. Examples are an atom inside a hypothesis
  formula, the `:` after a hypothesis name, name tokens such as `n23`, and `<act>` after
  pretraining states.
- They fire at the same rates on RL and base rollouts, and each carries ≤ 0.003% of variance.
- The most-rotated high-energy latents are similar: `:` after hypothesis names, name tokens, and
  `:=` in actions. None looks like an RL-specific behaviour.

### Stage 2: causal tests (layer 3)

![Stage 2](../../code/experiments/current/circuit_diffing_20261101/figs/stage2_L3.png)

Numbers in [`stage2.md`](../../code/experiments/current/circuit_diffing_20261101/stage2.md).
Solve % is the mean over problems of the success rate in 32 attempts, with 95% paired problem
bootstrap intervals.

| condition | G solve % | S solve % | share of the RL − base gap on G |
|---|---|---|---|
| base | 2.0 [1.2, 2.9] | 70.9 [67.0, 74.7] | — |
| base, shared latents at RL norm | 2.0 [1.1, 3.1] | 67.2 [62.9, 71.6] | recovers 0% [−1, 1] |
| base, every latent switched to its RL version | 9.6 [7.2, 12.2] | 80.3 [76.3, 84.3] | recovers 12% [8, 15] |
| base, layer-3 residual replaced by RL's | 13.9 [11.1, 16.9] | 81.3 [77.5, 84.9] | recovers 18% [14, 23] |
| RL (L3 end) | 67.1 [62.0, 72.3] | 93.0 [90.6, 95.2] | — |
| RL, shared latents at base norm | 68.0 [63.0, 72.9] | 92.8 [90.4, 95.1] | removes −1% [−5, 3] |
| RL, layer 3 replaced by its reconstruction | 69.0 [63.8, 73.9] | 91.8 [89.3, 94.1] | removes −3% [−7, 1] |
| RL, every latent switched to its base version | 40.5 [35.2, 46.1] | 87.3 [84.1, 90.3] | removes 41% [34, 47] |
| RL, the same edits permuted among latents (control) | 10.4 [6.4, 15.0] | 38.4 [32.8, 44.3] | removes 87% [80, 93] |

- **The crosscoder is faithful inside the sampler.** Running RL on its reconstruction alone
  keeps its solve rate.
- **Magnitude reweighting is inert.** Every shared latent set to the other model's decoder norm
  changes neither model's behaviour.
- **Direction changes carry part of the gain.**
  - Putting every latent back to its base version removes 41% of RL's G gain but only 26% of its
    S gain.
  - The relative drops are 40% on G and 6% on S, a selectivity of +33.5 [+27.0, +40.1] points.
  - The same edit vectors, applied to the wrong latents, are far more destructive on both G and
    S (selectivity +25.8 [+17.6, +34.2]). So the real edits stay on-manifold.
- **Layer 3 alone is not enough.** Replacing the base's whole layer-3 residual with RL's
  recovers 18% of the G gap and 47% of the S gap.

### Pre-registered predictions

| prediction | outcome |
|---|---|
| P0.1: `rel` rises with R (ρ ≥ 0.9); < 0.2% at `kp16_st1`; 1–30% at L3 end | holds (ρ 0.91; 0.047%; 6.5%) |
| P0.2: ≥ 75% of matrices have effective rank ≥ 0.5 | **fails**: 28% (median 0.42) |
| P0.3: median alignment 0.5–5×, outside share ≥ 90% | **half**: alignment 7.3× (fails), outside 93.7% (holds) |
| P0.4: seeds within ±30% | holds |
| P0.5: sparsity < 5% | holds (0.01%) |
| P1.1: FVU ≤ 0.10, loss recovered ≥ 90% | holds (0.008; 100%) |
| P1.2: null N_RL ≤ 3, ≥ 99% shared | holds (0; 100%) |
| P1.3: L3 end N_RL 5–100 | **fails**: 0, so the pre-registered "no RL-only features" branch applies |
| P1.4–P1.7: N_RL vs R, seeds, crosscoder seed, firing | moot at N_RL = 0; seeds and replicate pending (GPU) |
| P2.1: references match the categories | holds |
| P2.2: reconstruction keeps ≥ 80% | holds (103% G, 99% S) |
| P2.3: `base_replace` recovers 20–80% of the G gap | **fails, narrowly**: 18% [14, 23] |
| P2.4: `base_transplant` ≥ half of `base_replace` | holds (12% vs 18%) |
| P2.5: `base_rescale` < 15% | holds (0%) |
| P2.6: `rl_revert` removes ≥ 30% with selectivity > 0; `rl_unscale` < 15% | holds (41%; +33.5; −1%) |
| P2.10: `rl_revertperm` removes less of the G gap than `rl_revert` | **fails**: it removes more (87%), with lower selectivity |

## Interpretation and open hypotheses

- **Supported, at this resolution:** GRPO's capability gain is not carried by new features at
  layers 1, 3 or 5. Features the base already has are rotated and re-read, and the mean is
  shifted. That is a statement about linear features in single residual layers. It is not a
  statement about circuits in general.
- **What "new circuits" could still mean here.** RL may compute new functions *of* existing
  features: new combinations, and changed downstream reads in blocks 4–6. The 18% ceiling of
  layer-3 replacement says most of the gain needs those downstream changes. A residual-stream
  crosscoder cannot see a new connection between existing features. Edge-level attribution
  (Stage 3) is the tool for that, and it was not run.
- **Elicitation, read strictly as upweighting, fits these data badly.** Rescaling shared features
  to RL's norms does nothing. Direction changes are needed. But direction changes of existing
  features are also a form of re-use, not creation, so this does not show capability creation
  either.
- **Open: is the rotation RL-specific?**
  - The CPT crosscoder is pending. If CPT, at the same weight change, rotates shared latents as
    much (energy-weighted cosine ≈ 0.95), then rotation is generic drift. Its causal share would
    then still matter, but the rotation index would not mark capability gain.
  - Stage 0 already shows CPT and RL are indistinguishable in weight-space geometry.

## Is this a usable "new circuits" metric for the paper?

**Not as originally framed.** The RL-only latent count is 0 everywhere, including the large-gain
end of the R axis, so it cannot track capability gain. The method is sensitive enough: it finds
planted features.

What the paper can use:
- **A negative mechanistic result.** "Crosscoders find no RL-exclusive features, but causal
  patching shows the gain is not magnitude reweighting either." This sits better with "new
  computation over existing features" than with either pole.
- **A functional measure.** The share of the RL − base gap removed by reverting latent
  directions (41% at layer 3).

Both need the queued replicates before they go in a paper: seeds 1 and 2, the CPT crosscoder,
and the layer-5 Stage 2.

## Limitations and next steps

- **Seeds.** One base and RL seed for Stage 1 classification at layers 1 and 5 and for Stage 2.
  Seeds 1 and 2 have Stage 0 only; their crosscoders are queued.
- **Single layers.** Per-layer crosscoders cannot see a feature the base computes at a different
  layer, or a feature spread across layers. An acausal multi-layer crosscoder would.
- **Detection floor.** About 0.01% of variance for a one-latent feature. A new feature split
  over several latents can evade Δnorm, as planted feature 6 did. Latent Scaling alone catches
  it, and found 11 such latents for real; all looked like ordinary syntax features.
- **Stage 2 caveats.**
  - G's 13–20 problems are defined from only 12 base attempts.
  - The relative-drop selectivity partly reflects how fragile harder, longer problems are.
  - The permuted control is an off-manifold perturbation, so it is not a perfect magnitude match.
  - The `edit / ‖h‖` numbers in `stage2.md` include padded positions and are indicative only.
- **GPU.** Serrano had 3 other-user GPU jobs from 11:33 am to 1:05 pm and again from 3:43 pm, so
  this run had 0–1 GPUs. Queue G (`/scratch/cowifornia/nd-rl/circuits/logs/jobs_gpuG.txt`) holds
  the remaining jobs and waits for a slot.
- **Next.**
  1. Finish queue G: seeds 1 and 2, CPT crosscoder, layer-5 Stage 2, replicate, R sweep.
  2. Run an acausal multi-layer crosscoder on the L3 end.
  3. Stage 3: edge attribution (EAP-IG) on next-rule choice. Compare base and RL top edges over
     shared latents.

## Reproduce

From the worktree root, with `ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1` and the main checkout's
interpreter; replace `M` with `code.experiments.current.circuit_diffing_20261101`:

```bash
python -m M.stage0_weights                                  # CPU, 40 s
python -m M.rollouts --gpu 0 --models base:0,L3:0,kp16_st1:0   # ~2 min per model
python -m M.rollouts --pretrain
python -m M.crosscoder --gpu 0 --b L3:0 --layer 3           # ~8 min on a 3090
python -m M.crosscoder --gpu 0 --b L3:0 --layer 3 --plant --out <dir>_plant
python -m M.stage1_analyze
python -m M.problems
python -m M.stage2 --gpu 0 --cc <crosscoder dir> --conditions rl_clean,base_clean,...
python -m M.stage2_analyze && python -m M.figures
```

Artifacts are in `/scratch/cowifornia/nd-rl/circuits/`: `seqs/`, `crosscoders/`, `stage2/`,
`ckpts/` (the low-RL copies and CPT) and `logs/`.
