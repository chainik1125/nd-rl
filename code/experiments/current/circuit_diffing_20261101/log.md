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
