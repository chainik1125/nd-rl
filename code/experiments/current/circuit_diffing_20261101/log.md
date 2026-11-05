---
written_on: 2026-11-01
written_by: agent:claude
---

# Log: circuit diffing, base vs GRPO

Branch `leon/circuit-diffing` (from `leon/high-rl-paper` at `1abc0ee`), worktree
`/scratch/cowifornia/nd-rl/worktrees/circuit-diffing`. Artifacts go to
`/scratch/cowifornia/nd-rl/circuits/`. Times are PT.

## 2026-11-01

- **11:00 am.** Set up the worktree and the artifacts directory.
  - GPU state: another user's process holds GPU 2. GPUs 0, 1, 3 and 4 are free, so this
    experiment uses at most 2 GPUs.
  - Copied 12 low-RL checkpoints read-only from ynez with rsync (439 MB), to
    `/scratch/cowifornia/nd-rl/circuits/ckpts/lowrl/`: `kp16` and `kdense` at
    `st{1,2,4,8,16,32}`.
  - The base sha256 is the same on ynez and serrano (`8012222f…`).
- **11:06 AM.** PREREG.md with the Stage 0 section, before Stage 0 ran: sha256 `69273f6971354f5d13a270656dfdaf8825450a9297ffacf202b974fc52b1731a`.
- **11:16 AM.** PREREG.md with the Stage 1 section added, before any crosscoder was trained: sha256 `a22483bb6acba8b1892ce58227bb25cb9df116cd5212b6e810b2dc2bd59f0856`.
- **11:07 AM.** Stage 0 ran on CPU in 40 s. Results are in [`stage0.md`](stage0.md) and
  `stage0.json`.
- **11:12 AM – 11:33 AM.** Rollouts ran on GPUs 0 and 1 (`rollouts.py`, 1,200 problems × 4
  attempts per model): 19 models plus the pretraining sample.
- **11:33 AM.** Another user (`despair8`) started vLLM jobs on GPUs 3 and 4, about 11:16 am. With
  `cipeng`'s job on GPU 2, that makes 3 GPU jobs from other users, the serrano PSU limit. I
  stopped my remaining GPU job (the GPU 1 rollout queue). Only `L2_s2` is missing; it was
  killed mid-run. No GPU work runs until fewer than 3 of their jobs remain.
- **11:40 AM.** The GPUs are still blocked: 3 jobs from other users, at full power. As a hedge,
  I started reduced-size crosscoders on CPU (8 threads, nice): 4,096 latents, k 32, 6M training
  tokens, layer 3. Pairs: base vs L3 end, then base vs `kp16_st1`. They go to
  `/scratch/cowifornia/nd-rl/circuits/crosscoders_cpu/`. This deviates from the pre-registered
  recipe (8,192 latents, 40M tokens, GPU). If the GPUs free up, the pre-registered GPU runs
  replace them. Timing: 2.2k tokens/s for crosscoder steps and 5.9k tokens/s for paired
  activations on 8 threads.
- **11:52 AM.** The CPU job had opened a CUDA context on GPU 0, which nvidia-smi listed as a
  fourth GPU process. Its GPU load was zero, but I killed it anyway. It restarted at 11:53 am
  with `CUDA_VISIBLE_DEVICES=` (empty), so no CUDA context can open. The cause was not found:
  model loading, activation capture and crosscoder encoding do not initialise CUDA on their own.
- **12:44 PM.** The CPU crosscoder for base vs L3 end finished (layer 3, 4,096 latents, 6M
  tokens). The base vs `kp16_st1` null started.
  - **Quality.** Held-out FVU 0.019 (base) and 0.018 (RL). Next-token loss recovered is 99.9–100%
    for both models on both rollout sources. L0 is 31.8.
  - **Live latents.** 1,823 of 4,096 fire on the held-out data; 55% are dead.
  - **Classes.** 0 RL-only candidates (Δnorm < 0.1) and 0 base-only. Every live latent has
    Δnorm between 0.31 and 0.76, so the decoder-norm ratio is within 2^±1.1. 1,727 latents
    (95%) are shared (0.4–0.6).
  - **Decoder cosines.** Median 0.87; 5% are below 0.55, and a few are anti-aligned (minimum
    −0.82).
  - **Source-specific firing.** 30 latents fire only on RL rollouts and 20 only on base rollouts.
  - So at this resolution, PREREG's branch "L3 end has N_RL ≤ 5" applies.
  - Next steps: (i) a planted-feature positive control (`--plant`: 6 synthetic features added
    to the RL model's activations only, frequencies 0.3% and 3.4%, magnitudes 2, 4 and 8 in
    scaled units) to show that the method can find a model-exclusive feature at all; (ii) the
    pre-registered GPU recipe once a GPU is free.
- **1:05 PM.** One of `despair8`'s vLLM jobs ended. Other users now hold 2 GPUs (2 and 4), so
  this experiment can use one: GPU 0.
  - I stopped the CPU null run, since its GPU counterpart replaces it and the 8-core limit
    applies.
  - The pre-registered crosscoder queue started on GPU 0 (`queue.sh`, job list
    `/scratch/cowifornia/nd-rl/circuits/logs/jobs_gpuA.txt`). Priority order: L3 end, the null,
    the planted control, the R sweep, seeds 1 and 2, the crosscoder-seed replicate, layers 5
    and 1, then the remaining sweep points.
  - Each GPU crosscoder takes about 7 minutes (500 steps per 17 s).
- **01:32 PM.** PREREG.md with the Stage 2 section added (the reweighting branch), before any Stage 2 sampling: sha256 `04faf02fc3600afcf7dc66089c73e1bc91672edbc89daa2f290c592db87cfc3c`.
- **1:13–1:30 PM.** The first three GPU crosscoders finished, all at layer 3 with the
  pre-registered recipe (8,192 latents, k 32, 40M tokens).
  - **L3 end vs base.**
    - Quality: FVU 0.008 for both models, loss recovered 100%, 7,093 live latents (13% dead).
    - **0 strict RL-only latents and 0 base-only.** Δnorm ranges 0.20–0.82, so no latent's base
      decoder is under 40% of its RL decoder.
    - Energy-weighted decoder cosine 0.945. Latents with cosine < 0.5 carry 0.24% of RL
      variance; latents reweighted more than 2× carry 0.03%.
  - **Null (`kp16_st1`).** 0 RL-only latents; Δnorm 0.48–0.52; every decoder cosine is 1.00.
    The counts of latents that fire only on RL or only on base rollouts are the same as for the
    L3 end (137 / 113 vs 126 / 130), so those are sampling noise in rare latents.
  - **Planted control** (L3 end with 6 synthetic RL-only features).
    - **Strict rule.** Flags exactly 4 latents, all of them planted, with no false positives.
      They recover planted features 1, 2, 3 and 5: precision ≥ 0.998, recall 0.79–1.0,
      Δnorm ≤ 0.02.
    - **Misses.** Feature 4 is the weakest (0.22% frequency, magnitude 2; about 0.002% of
      variance); its latent has Δnorm 0.11, just over the threshold. Feature 6 is split over 3
      redundant latents with Δnorm 0.13–0.21.
    - **Latent Scaling alone** (|ν^ε|, |ν^r| < 0.2, exploratory) catches 7 of the 8 planted
      latents.
    - **Detection floor.** A new feature with ≥ ~0.01% of RL variance, carried by one latent, is
      found.
  - **Gate.** It passes trivially: null and L3 end are both 0. PREREG's branch "L3 end N_RL ≤ 5"
    applies, so Stage 2 switches to reweighting tests.
  - **Exploratory.** Latent Scaling alone, with |ν| < 0.3 on both targets, flags 11 latents of
    the real L3 crosscoder, against 0 for the null. They have Δnorm 0.3–0.55 and decoder
    cosine ≈ 0: the latent fires in the same contexts in both models, but the RL model writes a
    direction the base does not have there. Interpretation is pending (`interpret.py`).
- **1:31 PM.** I stopped the first queue after its L2 job and started queue B (`queue2.sh`, job
  list `/scratch/cowifornia/nd-rl/circuits/logs/jobs_gpuB.txt`). Order: L3 end at layers 5 and 1;
  CPT at lr 1.5e-4, 4.5e-4 and 1e-3; Stage 2; seeds 1 and 2; the R sweep; the crosscoder-seed
  replicate; the null at layers 5 and 1.
- **1:46 PM and 1:55 PM.** The L3-end crosscoders at layers 5 and 1 finished, each with 0
  strict RL-only latents.
  - **Layer 5.** FVU 0.019 / 0.017. 1 candidate has Δnorm < 0.1 (0.08), but Latent Scaling
    removes it at every threshold.
  - **Layer 1.** FVU 0.001 / 0.002; 4,204 live latents.
  - The L2-end crosscoder at layer 3 also has 0.
- **2:00 PM.** CPT (`train.finetune`) runs at about 4.6 s per step, so 24 minutes per lr. To save
  GPU time I stopped queue B after its first CPT run (lr 1.5e-4) and dropped the other two lrs.
  They run only if the first CPT misses the pre-registered `rel` band. Queue C (job list
  `jobs_gpuC.txt`) starts when that CPT ends: Stage 2, interpretation, seeds 1 and 2, the R
  sweep, the crosscoder-seed replicate, and interpretation at layer 5.
- **2:19 PM.** CPT at lr 1.5e-4 finished.
  - Median `rel` is 5.65%, 0.87× the L3 end's 6.50%, inside the pre-registered 0.7–1.4× band.
    So lr 1.5e-4 is the matched control, and the other two lrs are not run.
  - **Stage 0 on CPT** ([`stage0.md`](stage0.md)): `erank` 0.423 (L3 end 0.416), `r90` 0.145
    (0.139), `inside` 6.68% (6.27%), `align` 8.0 (7.3). Continued pretraining on the base's own
    data leaves the same weight-diff signature as RL. So the Stage 0 statistics reflect the
    optimizer, not capability.
- **2:26–2:38 PM.** Stage 2 at layer 3: `rl_clean` G 67.1%, S 93.0%; `base_clean` G 2.0%,
  S 70.9%; `base_replace` G 13.9%, S 81.3%.
- **2:41 PM.** Reordered the queue to queue E (`jobs_gpuE.txt`), which waits for the running
  Stage 2 job. Order: seeds 1 and 2, CPT rollouts and crosscoder, the crosscoder-seed
  replicate, then Stage 2 at layer 5 (addendum below), then the planted control at layer 5, then
  the R sweep. Queue D was cancelled, and its jobs were moved into E.
- **02:42 PM.** PREREG.md with the Stage 2 layer-5 addendum added, before any layer-5 sampling: sha256 `d4600d4ad8ae6dbbcc3c1b5040b45fcc954c2722218801f07284be73117d9b65`.
- **2:47–3:14 PM.** Stage 2 at layer 3. The share of the RL − base gap is in brackets (95% paired
  problem bootstrap).
  - `base_transplant`: G 9.6%, S 80.3% (recovers 12% [8, 15] of the G gap).
  - `base_rescale`: G 2.0%, S 67.2% (0% [−1, 1]).
  - `rl_revert`: G 40.5%, S 87.3% (removes 41% [34, 47]; selectivity +33.5 [+27.0, +40.1]
    points).
  - `rl_unscale`: G 68.0%, S 92.8% (removes −1% [−5, 3]).
  - `rl_recon` was cut by the job's 1-hour timeout and is rerun first in queue F.
- **03:15 PM.** PREREG.md with the matched-control addendum (`rl_revertperm`) added, before it ran:
  sha256 `ac72f04412facb2557dcfae1af702e0c063b082b8aa2bd2d33bfc6799e08a380`. Queue E was replaced by queue F (`jobs_gpuF.txt`), the same list with
  `rl_revertperm` added to both Stage 2 jobs.
- **3:31 PM and 3:43 PM.**
  - `rl_recon`: G 69.0%, S 91.8%, the same as `rl_clean`. The layer-3 reconstruction is
    behaviourally faithful inside the sampler.
  - `rl_revertperm`: G 10.4%, S 38.4%. The permuted edit wrecks both categories: it removes 87%
    of the G gap and pushes S below the base. Its selectivity is +25.8 [+17.6, +34.2] points,
    against +33.5 for `rl_revert`.
- **3:43 PM.** Another user (`despair8`) started two new GPU jobs (GPUs 1 and 3), so other users
  again hold 3 GPUs. The GPU guard refused every remaining job in queue F, and the queue drained
  without running anything. `interpret.py` runs on CPU instead. GPU jobs wait for a free slot.
- **3:47 PM.** `interpret.py` ran on CPU for the layer-3 L3-end crosscoder. The 11 exploratory
  latents and the 5 most-rotated high-energy latents all fire on proof-state syntax (atoms, `:`
  after hypothesis names, name tokens, `:=`), at equal rates on RL and base rollouts. None looks
  RL-specific.
- **3:52 PM.** All 5 GPUs are now held by other users (`despair8` 4, `cipeng` 1). Queue H
  (`jobs_gpuH.txt`) waits for a slot through `gpu_gate.sh`. Its order puts the CPT crosscoder
  first, because whether rotation is RL-specific is the main open question.
- **4:15 PM.** The GPUs are still all taken. The reduced CPU crosscoder for base vs L3 end
  (4,096 latents, 6M tokens) has the same rotation index as the GPU recipe: energy-weighted
  decoder cosine 0.946 vs 0.945. So a reduced CPU crosscoder can answer the open question of
  whether rotation is RL-specific.
- Started on CPU (7 threads, no visible GPU): base vs CPT, reduced recipe.
  - Deviation: CPT has no rollouts of its own (sampling needs a GPU), so the base's rollouts
    stand in for B's (`--b_roll_from base:0`).
  - Output: `crosscoders_cpu/cpt_lr0.00015_0_L3_n4096_k32_cpu_baseroll`.
  - The GPU version in queue H, with CPT's own rollouts, replaces it if a slot opens.
- **5:21 PM.** The CPU CPT crosscoder finished (reduced recipe, base rollouts standing in for
  CPT's own). FVU 0.019 / 0.019; 1,801 live latents; 0 CPT-only and 0 base-only.
  - Energy-weighted decoder cosine 0.953, against 0.946 for the L3 end on the same recipe.
  - Cosine 5% quantile 0.66 against 0.55; rotated-latent energy 0.04% against 0.10%.
  - So most of the rotation seen for RL is generic weight drift. RL's tail is heavier.
  - Numbers: `stage1_cpu.md`.
- **5:25 PM.** Wrapped up and handed back. Queue H was stopped before it ran anything,
  because no GPU slot opened between 3:43 pm and 5:25 pm. Its job list
  (`/scratch/cowifornia/nd-rl/circuits/logs/jobs_gpuH.txt`) holds the remaining replicates.
  No job of this experiment is running.

## 2026-11-04 (Stage 3; times PT, Z in brackets)

- **4:28 pm (00:28Z).** Stage 3 brief from Leon, via the coordinator.
  - Part A: a drift control for `rl_revert`.
  - Part B: edge-level attribution.
  - One GPU, ynez GPU 2 (`GPU-d6327ec5…`), reserved for `circuit_stage3` in
    `knockout_danbase/sup/reserved.json` since 00:30Z. At most 12 CPU cores.
  - Deadline 3 am PT (11:00Z), hard limit 8 am PT (16:00Z). Part B is reduced to fit, and
    pre-registered that way.
- **4:38 pm (00:38Z).** nvidia-smi on ynez shows no process on GPU 2 (4 MiB used).
  - Copied to ynez `/scratch/cowifornia/nd-rl/circuits/`: the base, L3 end, `kp16_st1` and CPT
    checkpoints, the L3-end crosscoders at layers 3 and 5, the `kp16_st1` layer-3 crosscoder,
    the `base_s0`, `kp16_st1_s0` and `pretrain` sequences, and the curriculum pools.
  - Everything goes under `runs/`, a mirror read through `CIRCUITS_RUNS`. ynez's own `think/`
    tree is an older copy.
  - sha256 matches serrano for every checkpoint and crosscoder: base `8012222f…`, L3 end
    `ca264766…`, CPT `6f418716…`, `kp16_st1` `e756b72d…`.
- **4:40 pm (00:40Z).** PREREG.md with the Stage 3 section added, before any Stage 3 job: sha256 `d123304a5db57cd2f56087e903deb2bbbc17ed6fff80b8fb6e8ffb6e9c867c3b`.
- **4:42 pm (00:42Z).** Queue A started on ynez GPU 2 (`queue3.sh`, job list
  `logs/jobs_s3A.txt` on ynez). Order:
  1. CPT rollouts;
  2. the base-vs-CPT crosscoder at layer 3;
  3. Stage-2 sampling on CPT: `rl_clean`, `rl_revert`, `rl_revertperm` and `rl_recon`, all
     driven by CPT;
  4. layer-5 crosscoders for CPT and `kp16_st1`.

  `gpu_guard.sh` checks GPU 2 every 2 min for processes outside the queue. The watchdog covers
  STALL, EXIT and OVERRUN.
- **4:47 pm.** CPT rollouts done: 4,800 attempts, 54.0% done, 71,504 sequences, 7.72M tokens.
- **4:52 pm.** Base-vs-CPT crosscoder at layer 3 trained. Final training FVU 0.008, the same as the
  L3-end crosscoder; L0 32.
- **4:50–5:05 pm.** Part B code written: `traces.py`, `edges.py` and `stage3_analyze.py`.
  - **CPU smoke test** on serrano, on synthetic sequences cut from the L3 pool rollouts (no
    Stage 3 data):
    - the functional blocks used for forward-mode AD match the model's own blocks to 2e-6;
    - the metric matches a full forward;
    - the joint latent code matches `Crosscoder.encode` exactly;
    - the JVP matches central finite differences to 0.4% of its scale;
    - the clean references reproduce exactly.
  - **Additions to the pre-registered design.** These are descriptive and not part of the
    decision rule:
    - an `allU` patching reference, which patches every U3 × U5 edge;
    - 3 → 5 counts with unweighted edges (Σ dpre over the receiver's active positions);
    - counts on the S set.
- **4:58 pm.** **CPT's own solve rates (A2):** G 29.6%, S 87.1%. For reference, the base is at
  G 2.0%, S 70.9% and the L3 end at G 67.1%, S 93.0%.
  - By set (base / CPT / RL):
    - Charles149 G: 0.0 / 10.0 / 73.4%. CPT still solves 0 of 32 on 10 of the 20 problems.
    - 13–20 G: 2.7 / 36.1 / 65.1%.
  - Continued pretraining on the base's own data, with no RL, is therefore not base-like. It
    recovers about 43% of RL's G gain on these problems. The Stage 1 and 2 write-up had *assumed*
    CPT was base-like; that assumption is wrong.
  - CPT is still the matched drift control (same optimizer, steps and weight change), but it is
    not capability-free.
  - Messaged the coordinator. Added one **exploratory** Part A statistic to
    `stage3_analyze.py`: the share of each model's own G gap over the base that its revert
    removes, CPT against RL. The pre-registered reading (relative drop on S) is unchanged.
- **5:00 pm.** Three CPU checks on the CPT gain, requested by the coordinator.
  1. **Compute.**
     - CPT trained on 124,800 distinct S1g3 records, 0.81 of an epoch: 82.1M tokens, 263k per
       step. `train.finetune` with seed 0 draws them deterministically from one shuffle.
     - So R_CPT = 6 × trained / (6 × Stage-1 tokens) = **0.186**, with no sampling.
     - The L3 chain trained on 84.0M tokens (L2 44.0M + L3 40.0M), so its trained-only R is
       0.191. With sampling (129.8M sampled tokens, 34% of its R), its full R is 0.289.
     - CPT therefore matches RL in trained tokens (0.98×) and in AdamW steps (312), and has
       0.64× RL's full R.
  2. **Learning-rate history.**
     - Stage 1 (`train.py`, recipe "best", 24,000 steps, 4.33 epochs, 440.6M tokens) uses Muon
       for the block matrices (peak 0.01) and AdamW for the rest (peak 1e-3, weight decay 0.1).
       The schedule is a 200-step warmup, then a step cosine to a floor of 30% of peak
       (`MIN_LR_FRAC` 0.3).
     - It ran its schedule to the end; it was not cut off. But it ended at 30% of peak (AdamW
       3e-4, Muon 3e-3), not annealed towards 0, and it is not a WSD schedule. Validation loss over
       the last 4,000 steps is flat and noisy (0.058–0.071).
     - CPT re-warms with AdamW on every parameter (10 steps to 1.5e-4) and decays by cosine to
       4.5e-5 over 312 steps, with weight decay 0. Its final LR is below Stage 1's AdamW floor.
     - GRPO does the same: AdamW at 1.5e-4 with a 10-step warmup; L2 constant, L3 WSD down to
       30% (4.5e-5).
     - So both CPT and RL add a low-LR AdamW phase that ends below where Stage 1 stopped.
     - Likely story: **the base is un-annealed, so it is effectively undertrained.** A short,
       lower-LR continuation on the same data recovers part of what RL gains. This is an
       interpretation; no annealing ablation was run.
  3. **Contamination.** Checked by atom-renaming class (`canon_key`) and the premise-order
     invariant `pkey` (ndbench `core.keys`), against all 154,630 S1g3 records:
     - 0 of the 80 G problems and 0 of the 121 S problems share a class with any record, so none
       is in CPT's trained subset either;
     - S1g3's exclusion list holds the OOD sets, textbook72 and the K12 held-out set.
- **5:06 pm.** **`cpt_revert` (A3):** G 16.9% [13.2, 20.9], S 85.0% [81.7, 88.1]; edit / ‖h‖
  0.30 (`rl_revert`: 0.34).
  - Relative drop on S: 2.5% [−0.1, 5.1]. That is 3.6 [0.3, 7.1] points smaller than
    `rl_revert`'s 6.1%.
  - cpt_revert's S stays 14.1 [10.9, 17.3] points above the base.
  - **Pre-registered reading: "benign".** The revert edit does not damage a drifted model in
    general, so the Stage 2 `rl_revert` result is not fragility. But the rule's conclusion,
    "RL's rotations are special", rested on the premise that CPT gained nothing, and that premise
    failed (4:58 pm).
  - **Exploratory:** the share of the model's own G gap over the base that its revert removes:
    - CPT 46.0% [36.0, 55.1];
    - RL 40.9% [34.4, 47.4];
    - difference +5.1 [−6.0, +15.7] points.

    Reverting the layer-3 rotations removes about the same share of whatever either kind of
    further training gained. Messaged the coordinator.
- **5:16 pm.** **`cpt_revertperm`:** G 6.1%, S 60.4%, edit / ‖h‖ 0.41. The permuted edit
  damages CPT, as it did RL (`rl_revertperm`: G 10.4%, S 38.4%).
- **5:20 pm.** `cpt_recon` ran out of GPU memory: a 6.4 GiB allocation in `PairPatched.edit`
  during the prefill at batch 256. The other conditions were already written.
  - It is rerun at the end of queue B with `--batch 64`. A different batch changes the sampling
    stream but not the setting.
- **5:20–5:30 pm.** Layer-5 crosscoder for CPT training. Queue B (`logs/jobs_s3B.txt` on ynez)
  waits for queue A to finish. Its order: `traces`, `edges score` for L3, CPT and `kp16`,
  `edges patch`, then the `cpt_recon` rerun.
- **5:25 pm.** The coordinator asked for a CPT control on the paper's evals, in a new folder
  [`cpt_control_20261105`](../cpt_control_20261105/). Its PREREG was hashed in its own log, and
  its runner (`ynez:/scratch/cowifornia/nd-rl/cpt_control/logs/run_cptc.sh`) waits for queue B.
- **5:26 pm and 5:31 pm.** Layer-5 crosscoders done for CPT (FVU 0.020 / 0.020) and `kp16_st1`
  (0.019 / 0.019). Queue A ended at 5:31 pm, and queue B started `traces`.
- **5:35 pm.** `stage1_analyze.py` rerun with the three new full-size crosscoders (Stage 1
  recipe; `stage1.md` regenerated).
  - **Base vs CPT, layer 3, full size (A1):**
    - FVU 0.008 / 0.008, loss recovered 100%, 7,317 live latents;
    - 0 CPT-only and 0 base-only latents;
    - energy-weighted decoder cosine **0.948**, against 0.945 for the L3 end;
    - 5% quantile of the cosine 0.51 (L3 end 0.51);
    - rotated-latent energy 0.12% (L3 end 0.24%); reweighted > 2×, 0.06% (L3 end 0.03%).

    The reduced CPU version had suggested that RL's tail is heavier (5% quantile 0.55 against
    0.66). At full size and with CPT's own rollouts, that difference is gone at layer 3.
  - **Layer 5:** CPT 0 / 0 latents, cosine 0.938 (5% quantile 0.60), against the L3 end's 0.925
    (0.50). `kp16_st1` is at 1.000.
- **5:33 pm.** `traces`: L3 G solve 66.9% (Stage 2: 67.1%), S 93.8%.
  - Every G problem has a kept attempt: 156 G and 159 S attempts, 6,951 sequences, 998k tokens.
  - 0 actions mismatch their sampled tails.
- **5:48 pm.** `edges score --x L3` took 15.6 min.
  - U3 = 369 and U5 = 374 latents, covering about 80% of |A| in both models.
  - First look at the L3 pair alone, before the nulls:
    - EAP 3 → 5: N_new 57, m_new 6.2% [4.4, 10.4].
    - Agreement between base and L3 on EAP edges is near zero: Spearman 0.01, Pearson −0.37,
      and 590 of the top 1,000 flip sign.
    - The **unweighted** edges agree almost perfectly: Pearson 0.99, sign agreement 98% in
      L3's top 1,000, median base/L3 magnitude ratio 0.97. These measure how much latent i at
      layer 3 drives latent j's pre-activation at layer 5 through each model's own blocks 4–5.
    - Node attributions are anti-correlated: Pearson −0.58 at layer 3, −0.39 at layer 5.
    - Split-half reliability over G problems is 0.87 (L3) and 0.92 (base), so the EAP disagreement
      is not noise.
  - **Reading (interpretation):** the latent-to-latent wiring through blocks 4–5 is essentially
    unchanged. What differs is the downstream gradient dM/df5, that is, how layer-5 features are
    read out into L3's chosen actions. On L3's own proof tokens, the two models use the same
    features towards different actions.
  - Added to `stage3_analyze.py`, as descriptive output: Pearson r, sign agreement and the
    base/X magnitude ratio for EAP edges, unweighted edges and node attributions. The decision
    rule is unchanged.
- **6:03 pm.** `edges score --x cpt` took 14.2 min.
  - CPT's m_new is 5.4% [3.7, 7.0], against L3's 6.2% [4.4, 10.4]. L3 − CPT = +0.8 [−1.3, +5.6]
    points, so criterion 1 of the decision rule (≥ 2× the drift null) fails.
  - Unlike L3, CPT's EAP edges agree with the base's: Pearson 0.88, 94% sign agreement. Its node
    attributions do too: 0.90 and 0.91.
  - Unweighted wiring agreement with the base is 0.99 for both.
- **6:05 pm.** Added two **exploratory** reference conditions to `edges.py patch`, before the patch
  job started. Both are cheap teacher-forced forwards, labelled exploratory, and outside the
  decision rule.
  - `swap13`: L3 with the base's embedding and blocks 1–3.
  - `swap6h`: L3 with the base's block 6, `ln_f` and head.

  Reason: the L3-specific disagreement sits in the downstream gradient, not in the 3 → 5 wiring,
  so these locate where the teacher-forced gap lives. Smoke-tested on CPU and synced to ynez.
